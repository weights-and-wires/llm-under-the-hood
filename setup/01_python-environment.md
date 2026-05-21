# 1. Python environment

This book targets **Python 3.11 or 3.12**. Earlier versions will hit type-syntax problems; newer versions may have torch wheel availability issues.

## The short version

```bash
# Check what you have
python --version

# Create an isolated env (any of these works; pick one)
python -m venv .venv && source .venv/bin/activate     # macOS / Linux
python -m venv .venv && .venv\Scripts\activate        # Windows PowerShell

# OR with uv (fast, recommended if you don't already have a venv tool)
uv venv && source .venv/bin/activate
```

Then proceed to [02_installing-dependencies.md](02_installing-dependencies.md).

## If you don't have Python at all

- **macOS:** `brew install python@3.12`
- **Linux (Debian/Ubuntu):** `sudo apt install python3.12 python3.12-venv`
- **Windows:** Download the installer from <https://www.python.org/downloads/> — check "Add Python to PATH" during install.

## If you're using conda / mamba

```bash
conda create -n uth python=3.12
conda activate uth
```

That's it for environments. You don't need notebook tooling, jupyter, or anything else yet — this repo runs `.py` files directly.
