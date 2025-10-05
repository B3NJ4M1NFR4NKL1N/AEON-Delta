import json
import torch
import logging
import re
import os
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import argparse

# Импорт токенизатора
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Logging: JSON + Sanitization + Dedup
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
    _allowed = tuple([chr(i) for i in range(32,127)]) + tuple([chr(i) for i in range(1024,1104)])
    def _sanitize(self, s: str) -> str:
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
    logger.handlers.clear()
    fmt = JSONLogFormatter()
    filters = [SanitizeTextFilter(), DedupFilter(window=200)]
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    for flt in filters: sh.addFilter(flt)
    logger.addHandler(sh)
    if logfile:
        dirn = os.path.dirname(logfile) or "."
        os.makedirs(dirn, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        for flt in filters: fh.addFilter(flt)
        logger.addHandler(fh)
    return logger

# Imports from core
from core import (
    ThoughtEncoder, ThoughtDecoder, AEONConfig, AEONTrainer,
    ThoughtAETrainer, AEONDelta, device
)

# Tokenization
def tokenize_text(text: str, tokenizer=None, max_len=64, device=None):
    if not isinstance(text, str):
        return None
    device = device or torch.device("cpu")
    if tokenizer:
        return tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0).to(device)
    else:
        vocab_size = 50000
        tokens = [ord(c) % vocab_size for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long, device=device)

# Improved KL & Entanglement Surrogate
def kl_diag_gaussians(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu_p = pred.mean(dim=-1)
    var_p = pred.var(dim=-1, unbiased=False) + eps
    mu_q = target.mean(dim=-1)
    var_q = target.var(dim=-1, unbiased=False) + eps
    kl = 0.5 * ((var_p / var_q) + ((mu_q - mu_p)**2) / var_q - 1.0 + torch.log(var_q / var_p))
    return kl.mean()

def cosine_spread_surrogate(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if z.size(0) < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    z_centered = z - z.mean(dim=1, keepdim=True)
    z_norm = F.normalize(z_centered, dim=1, eps=eps)
    sim = z_norm @ z_norm.T
    B = sim.size(0)
    offdiag_sum = sim.sum() - torch.diag(sim).sum()
    offdiag_count = B * (B - 1)
    return -offdiag_sum / (offdiag_count + 1e-9)

# KL warmup
def _kl_coeff(step: int, warmup: int = 500, max_w: float = 0.1) -> float:
    return min(1.0, step / float(max(1, warmup))) * max_w

# Customized Trainers
class SafeThoughtAETrainer(ThoughtAETrainer):
    def _to_float(self, v):
        return float(v.detach().cpu().item()) if hasattr(v, 'detach') else float(v)

    def _estimate_total(self, fm: dict) -> float:
        return sum(fm.values()) if 'total_loss' not in fm else fm['total_loss']

    def train_step(self, tokens, aug_tokens=None):
        if aug_tokens is None:
            aug_tokens = tokens.roll(1, dims=1)
        try:
            metrics = super().train_step(tokens, aug_tokens)
        except TypeError:
            metrics = super().train_step(tokens)
        fm = {k: self._to_float(v) for k, v in (metrics or {}).items()}
        total = self._estimate_total(fm)
        log = logging.getLogger("AEON-Delta")
        log.info(f"Phase A step: recon={fm.get('recon_loss', 0.0):.6f}, nce={fm.get('info_nce', 0.0):.6f}, kl={fm.get('kl', 0.0):.6f}")
        return fm

    def _info_nce(self, z, z_pos, temperature: float = 0.07):
        # z, z_pos: [B, D]
        sim = F.cosine_similarity(z.unsqueeze(1), z_pos.unsqueeze(0), dim=-1)  # [B, B]
        labels = torch.arange(z.size(0), device=z.device)                      # [B]
        return F.cross_entropy(sim / temperature, labels)

    def _kl_diag(self, z: torch.Tensor, eps: float = 1e-6):
        # KL( N(mu, var) || N(0, I) ) агрегировано по признакам
        mu  = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False).clamp_min(eps)
        return -0.5 * torch.sum(1 + var.log() - mu.pow(2) - var)


# Patch for warmup
try:
    _orig_train_step = SafeThoughtAETrainer.train_step
    def _patched_train_step(self, tokens, aug_tokens=None):
        if not hasattr(self, "global_step"):
            self.global_step = 0
        if aug_tokens is None:
            aug_tokens = tokens.roll(1, dims=1)
        self.model.train()
        tokens = tokens.to(self.device, non_blocking=False)
        aug_tokens = aug_tokens.to(self.device, non_blocking=False)
        z = self.model.encoder(tokens)
        z_pos = self.model.encoder(aug_tokens)
        logits = self.model.decoder(z, tokens)
        recon = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), tokens.reshape(-1))
        nce = self._info_nce(z, z_pos)
        kl = self._kl_diag(z)  # ✅ корректный вызов
        kl_w = _kl_coeff(self.global_step, warmup=getattr(self.config, "kl_warmup", 500), max_w=0.1)
        loss = recon + 0.3 * nce + kl_w * kl
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=getattr(self.config, "grad_clip_norm", 1.0))
        self.optimizer.step()
        self.global_step += 1
        log = logging.getLogger("AEON-Delta")
        log.info(f"Patched Phase A: kl_w={kl_w:.4f}, step={self.global_step}")
        return {
            "total_loss": float(loss.detach().cpu()),
            "recon_loss": float(recon.detach().cpu()),
            "info_nce": float(nce.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "kl_w": float(kl_w),
            "step": int(self.global_step),
        }
    SafeThoughtAETrainer.train_step = _patched_train_step
except Exception as e:
    logging.getLogger("AEON-Delta").error(f"[TrainHotfix] patch failed: {e}")

class FixedZDynamicsTrainer(AEONTrainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        self.optimizer = torch.optim.AdamW(self.model.rssm.parameters(), lr=config.learning_rate)
        self.config.use_amp = False
        self.logger = logging.getLogger("AEON-Delta")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(sh)

    def train_step(self, batch):
        self.model.rssm.train()
        z_t = batch[:, 0, :].to(self.device).float().contiguous()
        z_t1 = batch[:, 1, :].to(self.device).float().contiguous()
        z_t.requires_grad_(True)
        self.logger.debug(f"[FixedZDynamicsTrainer] z_t requires_grad: {z_t.requires_grad}, grad_fn: {z_t.grad_fn}")
        self.logger.debug(f"[FixedZDynamicsTrainer] z_t1 requires_grad: {z_t1.requires_grad}, grad_fn: {z_t1.grad_fn}")
        pred_z = self.model.rssm(z_t).contiguous()
        self.logger.debug(f"[FixedZDynamicsTrainer] pred_z requires_grad: {pred_z.requires_grad}, grad_fn: {pred_z.grad_fn}")
        mse = F.mse_loss(pred_z, z_t1)
        mask = torch.rand(z_t.shape, device=self.device) < 0.2
        if mask.any():
            idx = mask.nonzero(as_tuple=True)
            mask_loss = F.mse_loss(pred_z[idx], z_t1[idx])
        else:
            mask_loss = torch.zeros((), device=self.device, dtype=pred_z.dtype)
        kl = kl_diag_gaussians(pred_z, z_t1)
        ent_surrogate = cosine_spread_surrogate(pred_z)
        loss = mse + 0.2 * mask_loss + self.config.kl_weight * kl - 0.05 * ent_surrogate
        if not loss.requires_grad:
            self.logger.error(f"[FixedZDynamicsTrainer] CRITICAL ERROR: Loss tensor does not require grad.")
            self.logger.error(f"[FixedZDynamicsTrainer] loss.requires_grad: {loss.requires_grad}, loss.grad_fn: {loss.grad_fn}")
            self.logger.error(f"[FixedZDynamicsTrainer] pred_z.requires_grad: {pred_z.requires_grad}, pred_z.grad_fn: {pred_z.grad_fn}")
            self.logger.error(f"[FixedZDynamicsTrainer] z_t.requires_grad: {z_t.requires_grad}, z_t.grad_fn: {z_t.grad_fn}")
            raise RuntimeError("Loss tensor is not part of the computation graph.")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.rssm.parameters(), 1.0)
        self.optimizer.step()
        self.logger.info(f"Phase B step: mse={mse.item():.6f}, kl={kl.item():.6e}, ent={ent_surrogate.item():.6f}")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected")
        return {'total_loss': float(loss.detach().item()), 'mse': float(mse.detach().item()), 'kl': float(kl.detach().item()), 'ent': float(ent_surrogate.detach().item())}

    def fit(self, z_pairs_path: str, epochs: int = 6, batch_size: int = 256):
        self.logger.info(f"[FixedZDynamicsTrainer] Loading z_pairs from {z_pairs_path}")
        z_pairs = torch.load(z_pairs_path, map_location=self.device)
        if not isinstance(z_pairs, torch.Tensor) or z_pairs.dim() != 3 or z_pairs.shape[1] != 2:
            raise ValueError(f"Expected z_pairs tensor of shape [N,2,D], got {type(z_pairs)} shape={getattr(z_pairs, 'shape', None)}")
        loader = DataLoader(TensorDataset(z_pairs), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            progress_bar = tqdm(loader, desc=f"Phase B (Z-dyn) Epoch {epoch+1}/{epochs}")
            for (batch,) in progress_bar:
                try:
                    loss_dict = self.train_step(batch)
                    postfix = {
                        'loss': f"{loss_dict.get('total_loss', 0.0):.4f}",
                        'mse': f"{loss_dict.get('mse', 0.0):.4f}",
                        'kl': f"{loss_dict.get('kl', 0.0):.2e}",
                        'ent': f"{loss_dict.get('ent', 0.0):.4f}",
                    }
                    progress_bar.set_postfix(postfix)
                except Exception as e:
                    self.logger.error(f"Batch error: {e}")
                    continue

# Main Pipeline
def main(
    json_path: str = "/Users/vasapupkin/AEON/AEONSTART/combined.json",
    output_dir: str = "/Users/vasapupkin/AEON/AEONSTART/data/processed/",
    epochs_A: int = 30,
    epochs_B: int = 6,
    log_path: Optional[str] = "./aeon_training.log",
):
    logger = configure_logger(log_path)
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers unavailable; fallback to ord tokenize")
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AEONConfig(seq_length=64, z_dim=256, hidden_dim=256)
    config.kl_weight = 0.1
    if tokenizer:
        config.vocab_size = tokenizer.vocab_size
    else:
        config.vocab_size = 50000
    if not hasattr(config, "kl_warmup"):
        config.kl_warmup = 500
    if not hasattr(config, "grad_clip_norm"):
        config.grad_clip_norm = 1.0
    logger.info(f"Configuration: vocab_size={config.vocab_size}, kl_warmup={config.kl_warmup}")

    # Step 1: Load JSON
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
        raise ValueError("No texts in JSON")
    logger.info(f"Loaded {len(texts)} texts. Example: {texts[0][:50]}...")

    # Step 2: Tokenize
    logger.info("Tokenizing...")
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing"):
        t = tokenize_text(text, tokenizer=tokenizer, max_len=config.seq_length, device=device)
        if t is not None:
            tokenized.append(t)
    if not tokenized:
        raise ValueError("No valid tokens")
    full_tensor = torch.stack(tokenized)
    if torch.isnan(full_tensor).any() or torch.isinf(full_tensor).any():
        raise ValueError("NaN/Inf in full_tensor")

    # Step 3: Short vs full
    logger.info("Creating short/full datasets...")
    short_tokens = []
    pad_id = tokenizer.pad_token_id if tokenizer else 0
    half_len = config.seq_length // 2
    for t in full_tensor:
        non_pad_len = (t != pad_id).sum().item()
        if non_pad_len <= half_len:
            short_tokens.append(t)
        else:
            short_t = t[:half_len]
            short_tokens.append(torch.cat([short_t, torch.full((half_len,), pad_id, dtype=t.dtype, device=t.device)]))
    short_tensor = torch.stack(short_tokens)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(short_tensor, os.path.join(output_dir, "short.pt"))
    torch.save(full_tensor, os.path.join(output_dir, "full.pt"))
    logger.info("Saved short.pt and full.pt")

    # Step 4: Phase A
    logger.info("Phase A training starting...")
    model_A = AEONDelta(config).to(device)
    trainer_A = SafeThoughtAETrainer(model_A, config)
    trainer_A.fit(output_dir, epochs=epochs_A, curriculum=True)
    logger.info("Phase A done.")

    # Step 5: Build z_pairs
    logger.info("Building z_pairs...")
    model_A.eval()
    z_list = []
    with torch.no_grad():
        loader = DataLoader(full_tensor, batch_size=256, shuffle=False)
        for batch in tqdm(loader, desc="Encoding z"):
            z = model_A.encoder(batch.to(device))
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
            z_list.append(z.cpu())
    z_tensor = torch.cat(z_list)
    if z_tensor.size(0) < 2:
        raise ValueError("Need at least 2 texts to make z_pairs")
    z_pairs = torch.stack([z_tensor[:-1], z_tensor[1:]], dim=1)
    z_pairs_path = os.path.join(output_dir, "z_pairs.pt")
    torch.save(z_pairs, z_pairs_path)
    logger.info(f"Saved z_pairs at {z_pairs_path}")

    # Step 6: Phase B
    logger.info("Phase B training starting...")
    trainer_B = FixedZDynamicsTrainer(model_A, config)
    trainer_B.fit(z_pairs_path, epochs=epochs_B)
    logger.info("Phase B done.")

    # Verification and save
    dec_state = {k: v for k, v in model_A.state_dict().items() if k.startswith('decoder.')}
    if 'decoder.embed.weight' not in dec_state:
        logger.error("CRITICAL: 'decoder.embed.weight' is MISSING from the final model state_dict!")
        raise RuntimeError("Model state is corrupted. 'embed.weight' was not saved.")
    else:
        logger.info(f"VERIFICATION PASSED: 'decoder.embed.weight' found. Shape: {dec_state['decoder.embed.weight'].shape}")
    final_model_path = os.path.join(output_dir, "final_aeon_model.pt")
    torch.save(model_A.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AEON-Delta Training Pipeline")
    p.add_argument("--json_path", type=str, default="/Users/vasapupkin/AEON/AEONSTART/combined.json")
    p.add_argument("--output_dir", type=str, default="/Users/vasapupkin/AEON/AEONSTART/data/processed/")
    p.add_argument("--epochsA", type=int, default=30)
    p.add_argument("--epochsB", type=int, default=6)
    p.add_argument("--log", type=str, default="./aeon_training.log")
    args = p.parse_args()
    main(args.json_path, args.output_dir, args.epochsA, args.epochsB, args.log)
