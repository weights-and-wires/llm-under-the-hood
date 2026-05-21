# 2. Installing dependencies

## Core install (every project)

```bash
pip install -r requirements.txt
```

This gives you `torch`, `numpy`, `matplotlib`, `tiktoken`, `pytest` — enough to run roughly Projects 1–7 and to run every project's smoke tests.

## Dev install (if you'll modify code or run linters)

```bash
pip install -e ".[dev]"
```

Adds `ruff`, `pyright`, `pytest-xdist`, `jupytext`.

## Full install (heavier projects: datasets, transformers, FAISS, mixed-precision)

```bash
pip install -e ".[full]"
```

Adds `datasets`, `transformers`, `huggingface_hub`, `faiss-cpu`, `mmh3`, `pandas`, `tqdm`. Some later projects (8, 10, 14, 15, 28) need these.

## Per-project extras

A few projects need exotic dependencies (`flash-attn`, `bitsandbytes`, `llama-cpp-python`). Those live in `projects/NN_slug/requirements-extra.txt` and you install them only if you do that project:

```bash
pip install -r projects/16_quantization-and-deployment/requirements-extra.txt
```

## CUDA / CPU torch install

The default `requirements.txt` line `torch>=2.2` gets you whatever torch wheel pip picks for your platform — usually the **CUDA** build on Linux/Windows with a GPU, and the **CPU** build on macOS.

If you want to force the CPU build (smaller, faster install, no GPU needed):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

If you want a specific CUDA version (e.g. CUDA 12.1):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

The full matrix is at <https://pytorch.org/get-started/locally/>.

## Verifying the install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest --collect-only
```

If the second command reports collected tests without errors, you're done.
