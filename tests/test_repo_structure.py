"""
Smoke tests for the repo's structural invariants.

These are intentionally cheap. They catch the kind of breakage that a scaffold
PR can introduce silently — missing project folders, READMEs that vanished,
extractor that won't import. They run in <1s and need no external dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
TOOLS_DIR = REPO_ROOT / "tools"
SETUP_DIR = REPO_ROOT / "setup"

EXPECTED_PROJECT_COUNT = 35


def test_projects_dir_exists() -> None:
    assert PROJECTS_DIR.is_dir(), f"Missing projects/ directory at {PROJECTS_DIR}"


def test_projects_dir_has_expected_count() -> None:
    folders = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
    assert len(folders) == EXPECTED_PROJECT_COUNT, (
        f"Expected {EXPECTED_PROJECT_COUNT} project folders, found {len(folders)}: "
        f"{[p.name for p in folders]}"
    )


def test_project_folders_use_nn_slug_naming() -> None:
    """Every project folder must match the `NN_slug` convention (zero-padded number + kebab-case)."""
    import re

    pattern = re.compile(r"^\d{2}_[a-z0-9-]+$")
    folders = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
    bad = [p.name for p in folders if not pattern.match(p.name)]
    assert not bad, f"Project folders not matching NN_slug pattern: {bad}"


def test_every_project_has_readme() -> None:
    folders = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
    missing = [p.name for p in folders if not (p / "README.md").is_file()]
    assert not missing, f"Projects missing README.md: {missing}"


def test_project_numbers_are_contiguous_1_to_35() -> None:
    folders = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
    numbers = sorted(int(p.name.split("_", 1)[0]) for p in folders)
    assert numbers == list(range(1, EXPECTED_PROJECT_COUNT + 1)), (
        f"Project numbers should be 1..{EXPECTED_PROJECT_COUNT} contiguous; got {numbers}"
    )


def test_setup_docs_present() -> None:
    expected = {
        "README.md",
        "01_python-environment.md",
        "02_installing-dependencies.md",
        "03_gpu-and-hardware-tiers.md",
        "04_running-py-files.md",
        "05_datasets-and-checkpoints.md",
    }
    actual = {p.name for p in SETUP_DIR.iterdir() if p.is_file()}
    missing = expected - actual
    assert not missing, f"setup/ missing files: {sorted(missing)}"


def test_extract_code_module_imports() -> None:
    """The extractor module must at least import cleanly — guards against syntax regressions."""
    spec = importlib.util.spec_from_file_location("extract_code", TOOLS_DIR / "extract_code.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_code"] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "parse_chapter")
    assert hasattr(module, "slugify")


def test_slugify_examples() -> None:
    spec = importlib.util.spec_from_file_location("extract_code", TOOLS_DIR / "extract_code.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.slugify("The Learning Machine") == "the-learning-machine"
    assert module.slugify("From Prototype to nanoGPT") == "from-prototype-to-nanogpt"
    assert module.slugify("Long-Context Extension (RoPE, YaRN, NTK-Aware)") == (
        "long-context-extension-rope-yarn-ntk-aware"
    )
    assert module.slugify("DPO and Preference Optimization") == "dpo-and-preference-optimization"


def test_pyproject_toml_present() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "LICENSE").is_file()
    assert (REPO_ROOT / "conftest.py").is_file()


@pytest.mark.parametrize(
    "project_number,expected_slug_substring",
    [
        (1, "learning-machine"),
        (4, "attention-from-scratch"),
        (15, "grouped-query-attention"),
        (27, "quantization"),
        (35, "your-architecture"),
    ],
)
def test_specific_projects_present(project_number: int, expected_slug_substring: str) -> None:
    folders = [
        p
        for p in PROJECTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith(f"{project_number:02d}_")
    ]
    assert len(folders) == 1, f"Expected exactly one folder for project {project_number}"
    assert expected_slug_substring in folders[0].name, (
        f"Project {project_number} folder {folders[0].name!r} doesn't contain {expected_slug_substring!r}"
    )
