# 3. GPU and hardware tiers

This is the most-asked question this book gets. The honest answer: **you can do most of the projects on a laptop** — the rest have proxy versions designed for limited hardware.

## Three tiers

| Tier | Hardware | What you can run | Projects |
|------|----------|------------------|----------|
| **Laptop / CPU** | Any laptop, no GPU required | Smoke + unit tests; tiny configs (`--tiny` flag); full code for Projects 1–7 | 1–7, plus tier-1 tests for every other project |
| **One consumer GPU** | 8–24 GB VRAM (RTX 3070 / 4070 / 4090, M-series Mac with `mps`) | Most full labs at modest scale; pretraining is short; quantization and serving are realistic | 8–16, 18, 22–25, 28–30 |
| **Cloud rental / data center** | Single A100 / H100, or short rental on Lambda / RunPod / Modal | The full versions of: scaled pretraining, RLHF, large MoE, long-context fine-tuning | 8, 10, 11, 14, 15, 17, 21, 27, 31 |

## What `--tiny` means

Every project's `build.py` accepts a `--tiny` flag. With it, the project runs on a **proxy configuration**:

- `n_layer=2`, `n_head=2`, `d_model=32`
- Synthetic 1 MB dataset substituted for FineWeb / HF datasets
- 50 steps instead of 10,000
- Should complete on CPU in under 60 seconds

This is what the CI tests run against. It's also what readers without a GPU should run for the full project — you learn the structure without the wait.

```bash
python projects/05_your-gpt-from-a-blank-file/build.py --tiny
```

For the full version (only if you have the hardware):

```bash
python projects/05_your-gpt-from-a-blank-file/build.py --full
```

## "But I want to do the real thing"

If you don't have a GPU and you really want to run the full version of a project, three options:

1. **Cloud rental — by the hour.** Lambda, RunPod, Vast.ai, Modal, Paperspace. An A100 runs about $1–2/hour. Most full labs in this book finish in under 3 hours.
2. **Colab Pro (low-end GPUs).** A T4 or L4 will handle Projects 8, 14, 16, 22 at modest scale. Slower than an A100 but ~10× cheaper.
3. **HuggingFace Spaces / Inference Endpoints.** For projects 14–16 where you're fine-tuning rather than pretraining, a hosted endpoint can be cheaper than renting GPUs by the hour.

## What you should NOT do

- **Don't buy a GPU specifically for this book.** The `--tiny` proxy versions are designed to give you the same structural insight without the hardware. Rent if you need scale.
- **Don't try to run the full FineWeb pretraining on a laptop.** It will not finish. `--tiny` substitutes a 1 MB synthetic corpus that's enough to show the pipeline working.
- **Don't run Project 16 (quantization to GGUF) without checking disk space.** A full INT4 export of a 1.5B parameter model is ~800 MB. Multiple variants × multiple projects = your `models/` folder fills up fast.
