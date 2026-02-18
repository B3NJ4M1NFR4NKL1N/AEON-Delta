"""
AEON-Δ RMT v3.1 — Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="aeon-delta",
    version="3.1.0",
    description="AEON-Δ RMT v3.1: A Cognitive Architecture for Emergent Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AEON Research Team",
    license="LicenseRef-AEON-Delta-NC-Research-Only-1.0",
    url="https://github.com/B3NJ4M1NFR4NKL1N/AEON-Delta",
    py_modules=["aeon_core", "aeon_server", "ae_train", "test_fixes"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.2.0",
    ],
    extras_require={
        "server": [
            "fastapi>=0.109.1",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
            "python-multipart>=0.0.22",
            "psutil>=5.9.0",
        ],
        "full": [
            "fastapi>=0.109.1",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
            "python-multipart>=0.0.22",
            "psutil>=5.9.0",
            "transformers>=4.48.0",
            "tqdm>=4.60.0",
        ],
        "dev": [
            "matplotlib>=3.5.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aeon-core=aeon_core:main",
            "aeon-train=ae_train:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    data_files=[("", ["AEON_Dashboard.html"])],
    include_package_data=True,
)
