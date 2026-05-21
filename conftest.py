"""
Shared pytest fixtures for the Under the Hood code companion.

These fixtures give every per-project test the same starting point:
- A deterministic seed (no flaky tests from random init).
- A "tiny" model config that runs on CPU in seconds, not minutes.
- A temporary outputs directory so tests don't pollute the real outputs/ folders.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass(frozen=True)
class TinyModelConfig:
    """
    A uniform proxy configuration used across every project's tier-2 tests.

    Chosen to:
      - Run on CPU in < 30s per project.
      - Exercise the full forward + backward path without trivializing it.
      - Stay consistent across projects so per-project deltas are comparable.
    """

    n_layer: int = 2
    n_head: int = 2
    d_model: int = 32
    d_ff: int = 64
    vocab_size: int = 256
    seq_len: int = 16
    batch_size: int = 2
    dropout: float = 0.0


@pytest.fixture
def tiny_model_config() -> TinyModelConfig:
    return TinyModelConfig()


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs before every test.

    autouse=True so every test gets reproducibility for free.
    """
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@pytest.fixture
def tmp_outputs(tmp_path: Path) -> Path:
    """
    Per-test isolated outputs directory.

    Use instead of writing to projects/NN_slug/outputs/ inside tests.
    """
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    return out


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Auto-skip @pytest.mark.gpu tests on CPU-only runners.

    The book's audience includes readers without a GPU; we surface the marker
    so they can see "this would have run if you had CUDA" rather than failing.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="requires CUDA device")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if os.environ.get("UTH_RUN_SLOW") != "1":
        skip_slow = pytest.mark.skip(reason="slow test; set UTH_RUN_SLOW=1 to enable")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
