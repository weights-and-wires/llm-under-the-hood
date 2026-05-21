# 5. Datasets and checkpoints

Most projects in this book fetch external data. None of it is bundled with the repo — the files are too large and would balloon the clone size. This page explains where each dataset comes from, how to fetch it, and how the `--tiny` flag lets you skip the fetch entirely while you're learning.

## The three tiers of data access

| Tier | What you get | Project supports |
|------|--------------|------------------|
| **Tiny (default)** | 1 MB synthetic substitute; runs on CPU in <60s | Every project, via `--tiny` flag |
| **Sample** | A few hundred MB; representative slice of the real dataset | Most projects that train (8, 10, 14, 24) |
| **Full** | The actual dataset (often GBs) | Set env var `UTH_FULL_DATA=1` to enable |

The `--tiny` mode is the default for a reason: it lets you learn the structure of the project without committing to a multi-GB download. Switch up only when you're ready to run the full lab.

## Datasets used in this book

### FineWeb-EDU (Project 8: Pretraining)

- **Source:** <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- **Full size:** ~5 TB (the book uses a 1B-token sample slice; ~5 GB)
- **Sample slice:** ~500 MB
- **Tiny substitute:** synthetic web-style text generated at runtime

```bash
# Sample slice (default for --full)
pip install datasets
huggingface-cli login   # if first time

# Tiny (default — no download)
python projects/08_pretraining-on-the-real-web/build.py
```

### TinyStories (Projects 1–5)

- **Source:** <https://huggingface.co/datasets/roneneldan/TinyStories>
- **Full size:** ~1 GB
- **Why we use it:** Small enough to actually fit in your laptop's RAM, varied enough that next-character prediction produces interesting results.

### WikiText-103 (Projects 6, 7, 13)

- **Source:** <https://huggingface.co/datasets/wikitext>
- **Full size:** ~500 MB

### GSM8K (Project 15: Reward Models + RLHF)

- **Source:** <https://huggingface.co/datasets/gsm8k>
- **Full size:** ~10 MB
- Small enough that the `--tiny` and `--full` runs use the same data.

### Custom preference datasets (Projects 14, 15)

Generated locally by the project's own scripts. No external download.

## Checkpoints used in this book

### nanochat reference checkpoint (Projects 6, 8, 12)

- **Source:** referenced in chapter prose; the book uses Karpathy's nanochat as a comparison point.
- **Where to get it:** `git clone https://github.com/karpathy/nanochat` — the model weights are not in this repo.
- The relevant projects gracefully fall back to a re-implementation if nanochat is absent.

### llama.cpp GGUF models (Project 16: Quantization)

- **Source:** <https://huggingface.co/models?library=gguf>
- **Full size:** 500 MB – 4 GB per model depending on quantization level.
- The project lets you point at any GGUF file via `--model PATH`; doesn't ship any.

### Embedding model checkpoints (Projects 28, 29: Retrieval)

- **Source:** <https://huggingface.co/sentence-transformers>
- The default `sentence-transformers/all-MiniLM-L6-v2` is ~80 MB and downloads on first use.

## Cache locations

By default, HuggingFace caches go to:

- **Linux / macOS:** `~/.cache/huggingface/`
- **Windows:** `C:\Users\<you>\.cache\huggingface\`

To point them elsewhere (e.g., to an external drive):

```bash
export HF_HOME=/mnt/external/huggingface
# Windows PowerShell:
$env:HF_HOME = "D:\huggingface"
```

The repo's `.gitignore` excludes `data/`, `checkpoints/`, `models/`, `.cache/`, `*.gguf`, `*.bin`, `*.safetensors` so any locally-fetched files stay out of git.

## What happens if a download fails

Every project that fetches data wraps the fetch with a clear error message — e.g. "Could not reach HuggingFace; rerun with `--tiny` to use the synthetic substitute." The synthetic substitute is intentionally limited (it won't produce a quality model) but it's enough to verify the pipeline runs and the shapes are right.
