import json
import torch
import logging
import re
import os
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ======= Logging: JSON + Sanitization + Dedup =======
class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%d %H:%M:%S,%f"),
            "name": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class SanitizeTextFilter(logging.Filter):
    _allowed = tuple([chr(i) for i in range(32,127)]) + tuple([chr(i) for i in range(1024,1104)])  # ASCII printable + basic Cyrillic
    def _sanitize(self, s: str) -> str:
        # Strip control chars; collapse whitespace
        s = "".join(ch for ch in s if ch in self._allowed)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str) and ("Generated thought:" in record.msg or "Generated response:" in record.msg):
                head, sep, tail = record.msg.partition(":")
                if sep:
                    tail = self._sanitize(tail)
                    record.msg = f"{head}:{sep}{tail}"
                else:
                    record.msg = self._sanitize(record.msg)
        except Exception:
            pass
        return True

class DedupFilter(logging.Filter):
    def __init__(self, window: int = 50):
        super().__init__()
        self.window = window
        self.recent = []
    def filter(self, record: logging.LogRecord) -> bool:
        sig = (record.levelno, record.name, record.msg)
        if sig in self.recent:
            return False
        self.recent.append(sig)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        return True

def configure_logger(logfile: Optional[str] = None):
    logger = logging.getLogger("AEON-Delta")
    logger.setLevel(logging.INFO)
    # Clean existing handlers to avoid duplicates
    logger.handlers.clear()

    fmt = JSONLogFormatter()
    filters = [SanitizeTextFilter(), DedupFilter(window=200)]

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    for flt in filters: sh.addFilter(flt)
    logger.addHandler(sh)

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        for flt in filters: fh.addFilter(flt)
        logger.addHandler(fh)

    return logger

# ======= Imports from core =======
from core import (
    ThoughtEncoder, ThoughtDecoder, AEONConfig, AEONTrainer,
    ThoughtAETrainer, AEONDelta, device, logger as core_logger
)

# ======= Tokenization =======
def tokenize_text(text: str, max_len=64, vocab_size=50000, device=None):
    if not isinstance(text, str):
        return None
    device = device or torch.device("cpu")
    tokens = [ord(c) % vocab_size for c in text[:max_len]]
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long, device=device)

# ======= Improved KL (pred || target) =======
def kl_diag_gaussians(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Treat features of each sample as a diagonal Gaussian; compute per-sample KL(pred || target).
    pred, target: [B, D]
    Returns scalar KL averaged over batch.
    """
    mu_p = pred.mean(dim=-1)
    var_p = pred.var(dim=-1, unbiased=False) + eps
    mu_q = target.mean(dim=-1)
    var_q = target.var(dim=-1, unbiased=False) + eps

    # KL for 1D Gaussians per sample
    kl = 0.5 * ( (var_p/var_q) + ((mu_q - mu_p)**2)/var_q - 1.0 + torch.log(var_q/var_p) )
    return kl.mean()

# ======= Gradient-safe surrogate for entanglement (real-only, no inplace) =======
def cosine_spread_surrogate(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Minimize average pairwise cosine similarity in batch (decorrelate representations).
    Fully differentiable, no complex numbers.
    """
    if z.size(0) < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    z_centered = z - z.mean(dim=1, keepdim=True)
    z_norm = F.normalize(z_centered, dim=1, eps=eps)
    sim = z_norm @ z_norm.T  # [B,B]
    B = sim.size(0)
    offdiag_sum = sim.sum() - torch.diag(sim).sum()
    offdiag_count = B * (B - 1)
    return -offdiag_sum / (offdiag_count + 1e-9)

# ======= Customized Trainer for Z-dynamics with fixed KL and robust batch handling =======
class FixedZDynamicsTrainer(AEONTrainer):
    def train_step(self, batch):
        """
        batch is expected as [B, 2, hidden_dim] tensor saved by this script.
        We slice inside to be robust to DataLoader batching.
        """
        import builtins as _bi
        encoder = getattr(_bi, "encoder", None)
        if encoder is not None:
            encoder.eval()

        self.model.train()

        assert batch.dim() == 3 and batch.shape[1] == 2, f"Invalid batch shape: {batch.shape}"
        assert batch.shape[2] == self.config.hidden_dim, f"Feature dim {batch.shape[2]} != hidden_dim {self.config.hidden_dim}"

        # Ensure contiguous tensors to avoid AsStridedBackward issues from views
        z_t  = batch[:, 0, :].to(self.device).float().contiguous()
        z_t1 = batch[:, 1, :].to(self.device).float().contiguous()

        pred_z = self.model.rssm(z_t).contiguous()

        mse = F.mse_loss(pred_z, z_t1)

        # Random mask created independently of input strides + safe indexing
        mask = torch.rand(z_t.shape, device=self.device) < 0.2
        if mask.any():
            idx = mask.nonzero(as_tuple=True)
            mask_loss = F.mse_loss(pred_z[idx], z_t1[idx])
        else:
            mask_loss = torch.zeros((), device=self.device, dtype=pred_z.dtype)

        # KL (pred || target)
        kl = kl_diag_gaussians(pred_z, z_t1)

        # Safe entanglement surrogate, mapped to 'ent' to preserve API
        ent = -cosine_spread_surrogate(pred_z)

        # Original sign convention preserved: +0.05 * (-ent)
        loss = mse + 0.2 * mask_loss + self.config.kl_weight * kl + 0.05 * (-ent)

        # Backward
        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        logging.getLogger("AEON-Delta").info(f"Phase B step: mse={mse.item():.6f}, kl={kl.item():.6e}, ent={ent.item():.6f}")

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected")

        return {'total_loss': float(loss.detach().item()), 'mse': float(mse.detach().item()), 'kl': float(kl.detach().item()), 'ent': float(ent.detach().item())}

    def fit(self, z_pairs_path: str, epochs: int = 6):
        data = torch.load(z_pairs_path, map_location=self.device)
        if not isinstance(data, torch.Tensor) or data.dim() != 3 or data.shape[1] != 2:
            raise ValueError(f"Expected z_pairs tensor of shape [N,2,D], got {type(data)} shape={getattr(data, 'shape', None)}")
        loader = DataLoader(data, batch_size=256, shuffle=True)
        for epoch in range(epochs):
            progress_bar = tqdm(loader, desc=f"Phase B (Z-dyn) Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                loss_dict = self.train_step(batch)
                # display some key metrics
                postfix = {
                    'loss': f"{loss_dict.get('total_loss', 0.0):.4f}",
                    'mse': f"{loss_dict.get('mse', 0.0):.4f}",
                    'kl': f"{loss_dict.get('kl', 0.0):.2e}",
                    'ent': f"{loss_dict.get('ent', 0.0):.4f}",
                }
                progress_bar.set_postfix(postfix)

# ======= Safe wrapper over ThoughtAETrainer to avoid KeyError on 'total_loss' =======
class SafeThoughtAETrainer(ThoughtAETrainer):
    """
    Overrides .fit to NOT assume 'total_loss' is present in loss_dict.
    It calls the inherited .train_step and computes/infers total loss if missing.
    Also provides required aug_tokens for cores that expect two inputs.
    """
    def _to_float(self, v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().item())
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def _estimate_total(self, metrics: dict):
        if 'total_loss' in metrics:
            return self._to_float(metrics['total_loss'])
        if 'loss' in metrics:
            return self._to_float(metrics['loss'])
        # Compose from components if available
        recon = self._to_float(metrics.get('recon', 0.0))
        nce   = self._to_float(metrics.get('nce', 0.0))
        kl    = self._to_float(metrics.get('kl', 0.0))
        beta  = getattr(self.config, 'kl_weight', 1.0)
        return recon + nce + beta * kl

    def _make_loader(self, tokens: torch.Tensor, batch_size: int):
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"Expected tensor for tokens, got {type(tokens)}")
        if tokens.dim() != 2:
            raise ValueError(f"Tokens must be [N, L], got shape {tokens.shape}")
        # DataLoader expects tensors on CPU; move in train_step
        tokens = tokens.detach().cpu().long()
        ds = TensorDataset(tokens)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    def fit(self, output_dir: str, epochs: int = 30, curriculum: bool = True, batch_size: int = 256):
        log = logging.getLogger("AEON-Delta")
        short_path = os.path.join(output_dir, "short.pt")
        full_path  = os.path.join(output_dir, "full.pt")
        if not os.path.exists(short_path) or not os.path.exists(full_path):
            raise FileNotFoundError(f"Expected short.pt/full.pt in {output_dir}")
        short_tokens = torch.load(short_path, map_location="cpu")
        full_tokens  = torch.load(full_path,  map_location="cpu")

        loader_short = self._make_loader(short_tokens, batch_size)
        loader_full  = self._make_loader(full_tokens,  batch_size)

        use_short_epochs = max(1, int(0.3 * epochs)) if curriculum else 0

        for epoch in range(epochs):
            use_short = (epoch < use_short_epochs)
            loader = loader_short if use_short else loader_full
            phase = "short" if use_short else "full"

            progress_bar = tqdm(loader, desc=f"Phase A (AE) Epoch {epoch+1}/{epochs} [{phase}]")
            for batch in progress_bar:
                # Unwrap from TensorDataset
                tokens = batch[0] if isinstance(batch, (tuple, list)) else batch

                # Move to device
                tokens_dev = tokens.to(self.device, non_blocking=False)

                # ----- Provide aug_tokens required by some core versions -----
                # Fallback policy: identity augmentation (no change).
                # This satisfies signature ThoughtAETrainer.train_step(tokens, aug_tokens)
                # without altering training semantics unexpectedly.
                aug_tokens_dev = tokens_dev

                # Call inherited train_step (does its own optimizer/AMP/etc. inside core)
                try:
                    metrics = self.train_step(tokens_dev, aug_tokens_dev)
                except TypeError:
                    # If core variant actually expects only (tokens), fall back gracefully.
                    metrics = self.train_step(tokens_dev)

                # Normalize metrics to floats
                fm = {k: self._to_float(v) for k, v in (metrics or {}).items()}
                total = self._estimate_total(fm)

                # Logging + progress
                log.info(f"Phase A step: recon={fm.get('recon', 0.0):.6f}, nce={fm.get('nce', 0.0):.6f}, kl={fm.get('kl', 0.0):.6f}")
                postfix = {
                    "loss": f"{total:.4f}",
                }
                if "recon" in fm: postfix["recon"] = f"{fm['recon']:.4f}"
                if "nce" in fm:   postfix["nce"]   = f"{fm['nce']:.4f}"
                if "kl" in fm:    postfix["kl"]    = f"{fm['kl']:.2e}"
                progress_bar.set_postfix(postfix)

# ======= Pipeline =======
def main(
    json_path: str = "/Users/vasapupkin/AEON/AEONSTART/combined.json",
    output_dir: str = "/Users/vasapupkin/AEON/AEONSTART/data/processed/",
    epochs_A: int = 30,
    epochs_B: int = 6,
    log_path: Optional[str] = "./aeon_training.log",
):
    # Configure logger (overrides handlers from core)
    configure_logger(log_path)
    logger = logging.getLogger("AEON-Delta")

    # Config
    config = AEONConfig(seq_length=64, vocab_size=50000, z_dim=256, hidden_dim=256)
    config.kl_weight = 0.1  # default; can be annealed externally if needed

    # ===== Step 1: Load JSON (robust to NDJSON and concatenation) =====
    logger.info("Loading JSON dataset...")
    texts = []
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and "text" in sub:
                            texts.append(sub["text"])
                        elif isinstance(sub, str):
                            texts.append(sub)
                logger.info(f"Parsed line {line_num} successfully")
            except json.JSONDecodeError as e:
                logger.warning(f"Error in line {line_num}: {str(e)}. Attempt split...")
                parts = re.split(r'(?<=})\s*(?={)', line)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        item = json.loads(part)
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                        logger.info(f"Parsed part in line {line_num}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed part in line {line_num}: {part[:80]}...")

    if not texts:
        raise ValueError("No texts in JSON — check file/format")

    logger.info(f"Loaded {len(texts)} texts. Example: {texts[0][:50]}...")

    # ===== Step 2: Tokenize =====
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing"):
        t = tokenize_text(text, max_len=64, vocab_size=config.vocab_size, device=device)
        if t is not None:
            tokenized.append(t)

    if not tokenized:
        raise ValueError("No valid tokens after tokenization")

    # ===== Step 3: Short vs full =====
    short_tokens = []
    for t in tokenized:
        non_pad_len = int((t != 0).sum().item())
        if non_pad_len <= 32:
            short_tokens.append(t)
        else:
            short_t = torch.cat([t[:32], torch.zeros(64 - 32, dtype=t.dtype, device=t.device)])
            short_tokens.append(short_t)

    full_tokens = tokenized

    short_tensor = torch.stack(short_tokens)
    full_tensor  = torch.stack(full_tokens)

    assert short_tensor.shape[1] == 64 and full_tensor.shape[1] == 64
    assert torch.all(short_tensor >= 0) and torch.all(short_tensor < config.vocab_size)
    if torch.isnan(short_tensor).any() or torch.isinf(short_tensor).any():
        raise ValueError("NaN/Inf in short_tensor")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(short_tensor, os.path.join(output_dir, "short.pt"))
    torch.save(full_tensor,  os.path.join(output_dir, "full.pt"))
    logger.info("Saved short.pt and full.pt")

    # ===== Step 4: Phase A training (AE) =====
    logger.info("Phase A training (Thought AE) starting...")
    # Recreate encoder/decoder to ensure local scope uses trainable ones
    from core import ThoughtEncoder as CoreThoughtEncoder, ThoughtDecoder as CoreThoughtDecoder
    enc = CoreThoughtEncoder(config.vocab_size, emb_dim=256, z_dim=256).to(device)
    dec = CoreThoughtDecoder(config.vocab_size, emb_dim=256, z_dim=256).to(device)
    import builtins as _bi
    _bi.encoder = enc
    _bi.decoder = dec

    # Minimal wrapper model to pass params into AE trainer
    class DummyModel(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        def to(self, dev):
            self.encoder.to(dev); self.decoder.to(dev); return self
        def parameters(self):
            return list(self.encoder.parameters()) + list(self.decoder.parameters())
        def named_parameters(self):
            for n,p in self.encoder.named_parameters(): yield f"encoder.{n}", p
            for n,p in self.decoder.named_parameters(): yield f"decoder.{n}", p

    model_A = DummyModel(enc, dec)
    # Use the safe wrapper to avoid KeyError('total_loss') and to pass aug_tokens
    trainer_A = SafeThoughtAETrainer(model_A, config)
    trainer_A.fit(output_dir, epochs=epochs_A, curriculum=True)

    # ===== Step 5: Build z_pairs using the trained encoder =====
    logger.info("Building z_pairs from encoder outputs...")
    enc.eval()
    z_list = []
    with torch.no_grad():
        for tokens in tqdm(tokenized, desc="Encoding z"):
            z = enc(tokens.unsqueeze(0))
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
            z_list.append(z.squeeze(0))
    z_tensor = torch.stack(z_list)  # [N, 256]

    if z_tensor.size(0) < 2:
        raise ValueError("Need at least 2 texts to make z_pairs")

    pairs = []
    for i in range(z_tensor.size(0) - 1):
        pairs.append(torch.stack([z_tensor[i], z_tensor[i+1]], dim=0))
    z_pairs = torch.stack(pairs)  # [N-1, 2, 256]
    z_pairs_path = os.path.join(output_dir, "z_pairs.pt")
    torch.save(z_pairs, z_pairs_path)
    logger.info(f"Saved z_pairs at {z_pairs_path}")

    # ===== Step 6: Phase B training (Z-dynamics) =====
    logger.info("Phase B training (Z-dynamics) starting...")
    model_B = AEONDelta(config)
    trainer_B = FixedZDynamicsTrainer(model_B, config)
    trainer_B.fit(z_pairs_path, epochs=epochs_B)
    logger.info("Phase B done.")

if __name__ == "__main__":
    # Optional CLI
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", type=str, default="/Users/vasapupkin/AEON/AEONSTART/combined.json")
    p.add_argument("--output_dir", type=str, default="/Users/vasapupkin/AEON/AEONSTART/data/processed/")
    p.add_argument("--epochsA", type=int, default=30)
    p.add_argument("--epochsB", type=int, default=6)
    p.add_argument("--log", type=str, default="./aeon_training.log")
    args = p.parse_args()
    main(args.json_path, args.output_dir, args.epochsA, args.epochsB, args.log)
