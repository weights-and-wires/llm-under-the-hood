# 4. Running `.py` files (and converting to notebooks if you prefer)

This repo ships `.py` files, not `.ipynb` notebooks. The reasons:

- `.py` runs cleanly in CI; notebooks don't (output cells churn the diff).
- `.py` works in any editor; notebooks lock you into Jupyter / VSCode notebook UI.
- `.py` plays well with version control; notebook JSON does not.

But you can still get a notebook experience if that's how you prefer to learn.

## Running a project's main script

```bash
# From the repo root:
python projects/01_the-learning-machine/build.py
```

Common flags every project supports:

| Flag | What it does |
|------|--------------|
| (none) | Default config — runs on whatever hardware you have |
| `--tiny` | Proxy config: tiny model, synthetic data, ~60s on CPU |
| `--full` | Full lab — requires the hardware described in [03_gpu-and-hardware-tiers.md](03_gpu-and-hardware-tiers.md) |
| `--seed N` | Override the default seed (0) for reproducibility experiments |
| `--output-dir PATH` | Where to write outputs (defaults to `projects/NN_slug/outputs/`) |

## Running the BREAK IT experiment

Every project has a `break_it.py` next to `build.py`. The BREAK IT runs the same code with one mechanism deliberately disabled or sabotaged — you compare the output to `build.py` to learn *why* that mechanism is there:

```bash
python projects/01_the-learning-machine/break_it.py
```

## Running tests

```bash
# All projects
pytest

# One project
pytest projects/01_the-learning-machine/tests/

# Only fast tests (skip @pytest.mark.slow and @pytest.mark.gpu)
pytest -m "not slow and not gpu"

# Including slow tests
UTH_RUN_SLOW=1 pytest
```

## Converting `.py` to `.ipynb`

If you prefer to step through cells in Jupyter, use `jupytext`. Install once:

```bash
pip install jupytext
```

Then convert any project's `build.py`:

```bash
jupytext --to ipynb projects/01_the-learning-machine/build.py
# produces projects/01_the-learning-machine/build.ipynb
jupyter lab projects/01_the-learning-machine/build.ipynb
```

The conversion treats `#%%` lines as cell breaks. If you want cleaner cell boundaries, edit the `.py` to add `#%%` markers where you'd like a new cell — jupytext respects them on re-conversion.

To keep `.py` and `.ipynb` synced as you edit the notebook:

```bash
jupytext --set-formats py:percent,ipynb projects/01_the-learning-machine/build.py
```

After that, saving the `.ipynb` auto-updates the `.py` and vice versa.

## Why we don't ship notebooks directly

We considered shipping both formats. Three reasons we didn't:

1. **Output cell churn.** A notebook re-run changes random cell outputs and version-controlled metadata. The diffs become noise.
2. **Pedagogical drift.** Notebooks reward you for executing top-to-bottom. The book's BREAK IT philosophy needs you to comment lines out, re-run subsections, and compare. That's friction in a notebook UI.
3. **Honest deps.** A `.py` file declares its imports at the top. A notebook hides them across cells. The book is about understanding what's actually happening — that starts with imports.

If you disagree, the `jupytext` recipe above gets you a notebook in 10 seconds. We just don't make it the default.
