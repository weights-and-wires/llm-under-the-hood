"""
Extract runnable Python code from the book's chapter markdown sources.

Reads the chapters at --chapters-dir, parses each chapter's structured sections
(Hook / The Concept / Why It Matters / The Build / BREAK IT / ...), extracts the
fenced ```python blocks from each section, and writes per-project folders to
--out-dir (default: _generated/) containing:

    NN_slug/
      README.md          # prose: title, Hook, Concept, Why It Matters, how to run
      step_NN_<slug>.py  # one per ### Step N: subsection of The Build (if it has python)
      build.py           # concatenation of all The Build python blocks
      break_it.py        # concatenation of all BREAK IT python blocks

By default, output goes to _generated/ — promotion to projects/ is manual (or
--promote) so hand-edits in projects/ are never overwritten.

Usage:
    python tools/extract_code.py --dry-run             # parse + print stats only
    python tools/extract_code.py                       # extract to _generated/
    python tools/extract_code.py --promote             # copy _generated/* -> projects/*
    python tools/extract_code.py --report              # regenerate GAPS.md only

GAPS.md is gitignored — it's a private author-facing audit of where each
project stands (block counts, runs y/n, prose-only flags). Never committed.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAPTERS = Path("C:/autonovel-master/autonovel-master/chapters")
DEFAULT_OUT = REPO_ROOT / "_generated"
PROJECTS_DIR = REPO_ROOT / "projects"
GAPS_PATH = REPO_ROOT / "GAPS.md"

PYTHON_FENCE = re.compile(r"^```python\s*$", re.MULTILINE)
ANY_FENCE = re.compile(r"^```", re.MULTILINE)
H1 = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
H2 = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
H3_STEP = re.compile(r"^### Step (\d+):?\s*(.*)\s*$", re.MULTILINE)
H3_ANY = re.compile(r"^### (.+?)\s*$", re.MULTILINE)
PROJECT_TITLE = re.compile(r"^Project\s+(\d+):\s*(.+?)\s*$")


@dataclasses.dataclass
class Step:
    n: int
    title: str
    python_blocks: list[str]


@dataclasses.dataclass
class Chapter:
    path: Path
    h1: str
    project_n: int | None
    project_title: str | None
    sections: dict[str, str]
    build_steps: list[Step]
    break_it_blocks: list[str]
    total_python_blocks: int

    @property
    def is_project(self) -> bool:
        return self.project_n is not None

    @property
    def slug(self) -> str:
        if self.project_title is None:
            return ""
        return slugify(self.project_title)

    @property
    def folder_name(self) -> str:
        if not self.is_project:
            return ""
        assert self.project_n is not None
        return f"{self.project_n:02d}_{self.slug}"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text


def extract_python_blocks(markdown: str) -> list[str]:
    """
    Return all ```python fenced code blocks from `markdown`, in document order.

    Robust against fences that are immediately followed by content (no blank line)
    and against blocks that contain triple backticks inside comments (rare).
    """
    blocks: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"^```python\s*$", lines[i]):
            i += 1
            buf: list[str] = []
            while i < len(lines) and not re.match(r"^```\s*$", lines[i]):
                buf.append(lines[i])
                i += 1
            blocks.append("\n".join(buf))
        i += 1
    return blocks


def split_by_h2(markdown: str) -> dict[str, str]:
    """
    Split a chapter body into {section_name: section_body} on H2 boundaries.

    Section names are normalized (stripped, lowercased), e.g. "the build", "break it".
    """
    parts = re.split(r"^## (.+?)\s*$", markdown, flags=re.MULTILINE)
    sections: dict[str, str] = {}
    if len(parts) <= 1:
        return sections
    # parts[0] is the pre-first-H2 content; skip it.
    for i in range(1, len(parts), 2):
        name = parts[i].strip().lower()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections[name] = body
    return sections


def normalize_build_headings(build_body: str) -> str:
    """
    Chapters use two H3 conventions inside `## The Build`:
      - `### Step N: Title`   (most chapters)
      - `### N. Title`        (ch_05, ch_19, ch_35)
    Rewrite the second form to the first so split_build_into_steps handles both.
    """
    return re.sub(r"^### (\d+)\.\s+", r"### Step \1: ", build_body, flags=re.MULTILINE)


def split_build_into_steps(build_body: str) -> list[Step]:
    """
    The Build section is structured as `### Step N: <title>` subsections.

    Returns the steps in order; each step has its python blocks attached.
    Steps without any python blocks are still included (empty list) so the
    pedagogical numbering is preserved in the GAPS report.
    """
    build_body = normalize_build_headings(build_body)
    parts = re.split(r"^### Step (\d+):?\s*(.*?)\s*$", build_body, flags=re.MULTILINE)
    steps: list[Step] = []
    if len(parts) <= 1:
        return steps
    for i in range(1, len(parts), 3):
        n = int(parts[i])
        title = parts[i + 1].strip()
        body = parts[i + 2] if i + 2 < len(parts) else ""
        blocks = extract_python_blocks(body)
        steps.append(Step(n=n, title=title, python_blocks=blocks))
    return steps


def parse_chapter(path: Path) -> Chapter:
    text = path.read_text(encoding="utf-8")

    h1_match = H1.search(text)
    h1 = h1_match.group(1).strip() if h1_match else path.stem

    project_n: int | None = None
    project_title: str | None = None
    project_match = PROJECT_TITLE.match(h1)
    if project_match:
        project_n = int(project_match.group(1))
        project_title = project_match.group(2).strip()

    sections = split_by_h2(text)
    build_steps = split_build_into_steps(sections.get("the build", ""))
    break_it_blocks = extract_python_blocks(sections.get("break it", ""))

    total = sum(len(s.python_blocks) for s in build_steps) + len(break_it_blocks)
    # Also count python blocks outside Build / BREAK IT (e.g. The Concept) so
    # GAPS reflects the chapter's full extractable surface.
    all_blocks = extract_python_blocks(text)
    total = max(total, len(all_blocks))

    return Chapter(
        path=path,
        h1=h1,
        project_n=project_n,
        project_title=project_title,
        sections=sections,
        build_steps=build_steps,
        break_it_blocks=break_it_blocks,
        total_python_blocks=total,
    )


def iter_chapter_files(chapters_dir: Path) -> Iterator[Path]:
    paths = sorted(chapters_dir.glob("ch_*.md"))
    yield from paths


def render_readme(ch: Chapter) -> str:
    """
    Per-project README.md stub. Pulls prose from Hook + Concept + Why It Matters.

    The intent is for the user to edit this after extraction — embed outputs,
    sharpen the language, add personal commentary. The stub gives a coherent
    starting point so no project has an empty README.
    """
    hook = ch.sections.get("hook", "").strip()
    concept = ch.sections.get("the concept", "").strip()
    why = ch.sections.get("why it matters", "").strip()

    if not ch.is_project:
        return f"# {ch.h1}\n\n{hook}\n"

    title = f"Project {ch.project_n}: {ch.project_title}"
    body = [f"# {title}", ""]

    if hook:
        body += ["## Hook", "", hook, ""]
    if concept:
        body += ["## The Concept", "", concept, ""]
    if why:
        body += ["## Why It Matters", "", why, ""]

    body += [
        "## How to run this project",
        "",
        "```bash",
        "# Proxy run (tiny model, runs on CPU in <60s):",
        f"python projects/{ch.folder_name}/build.py --tiny",
        "",
        "# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):",
        f"python projects/{ch.folder_name}/build.py --full",
        "",
        "# The BREAK IT experiment:",
        f"python projects/{ch.folder_name}/break_it.py",
        "```",
        "",
        "## Outputs",
        "",
        "_To be captured in PR 3. Will include loss curves, sample generations, and any benchmark results._",
        "",
        "## Read in the book",
        "",
        f"This project is Chapter {ch.project_n} of *Under the Hood: Build Every Layer of a "
        "Large Language Model from Scratch*. Buy the book at "
        "<https://leanpub.com/under-the-hood>.",
        "",
    ]
    return "\n".join(body)


def render_build_py(ch: Chapter) -> str:
    """
    Concatenate every Build step's python blocks into a single runnable file,
    with comment headers preserving the step boundaries.
    """
    lines = [
        '"""',
        f"Project {ch.project_n}: {ch.project_title}",
        "",
        "Canonical runnable build for this project. Generated by tools/extract_code.py",
        "from the book's chapter markdown — re-run that tool to regenerate.",
        '"""',
        "",
    ]
    for step in ch.build_steps:
        if not step.python_blocks:
            continue
        lines.append(f"# === Step {step.n}: {step.title} ===")
        lines.append("")
        for block in step.python_blocks:
            lines.append(block.rstrip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_break_it_py(ch: Chapter) -> str:
    if not ch.break_it_blocks:
        return ""
    lines = [
        '"""',
        f"Project {ch.project_n}: BREAK IT experiment.",
        "",
        "Deliberately sabotages one mechanism from build.py to show what happens",
        "when it's removed. Compare outputs to the working version.",
        '"""',
        "",
    ]
    for block in ch.break_it_blocks:
        lines.append(block.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_step_py(ch: Chapter, step: Step) -> str:
    if not step.python_blocks:
        return ""
    lines = [
        '"""',
        f"Project {ch.project_n}: Step {step.n} — {step.title}",
        "",
        "Pedagogical reference: this file shows the code for this step in isolation.",
        "For the full assembled, runnable build, use build.py in this same folder.",
        '"""',
        "",
    ]
    for block in step.python_blocks:
        lines.append(block.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_chapter(ch: Chapter, out_dir: Path, placeholders_only: bool = False) -> dict:
    """Write a chapter's project folder. Returns a per-project status dict for GAPS.md."""
    status = {
        "project_n": ch.project_n,
        "title": ch.project_title or ch.h1,
        "folder": ch.folder_name or ch.path.stem,
        "python_blocks": ch.total_python_blocks,
        "step_count": len(ch.build_steps),
        "steps_with_code": sum(1 for s in ch.build_steps if s.python_blocks),
        "has_build_py": False,
        "has_break_it_py": False,
        "step_files_written": 0,
        "is_project": ch.is_project,
    }

    if not ch.is_project:
        return status

    proj_dir = out_dir / ch.folder_name
    proj_dir.mkdir(parents=True, exist_ok=True)

    (proj_dir / "README.md").write_text(render_readme(ch), encoding="utf-8")

    if placeholders_only:
        return status

    build_py = render_build_py(ch)
    if build_py.strip() and any(s.python_blocks for s in ch.build_steps):
        (proj_dir / "build.py").write_text(build_py, encoding="utf-8")
        status["has_build_py"] = True

    break_it_py = render_break_it_py(ch)
    if break_it_py.strip():
        (proj_dir / "break_it.py").write_text(break_it_py, encoding="utf-8")
        status["has_break_it_py"] = True

    for step in ch.build_steps:
        step_py = render_step_py(ch, step)
        if not step_py.strip():
            continue
        step_slug = slugify(step.title) if step.title else f"step-{step.n}"
        fname = f"step_{step.n:02d}_{step_slug}.py"
        (proj_dir / fname).write_text(step_py, encoding="utf-8")
        status["step_files_written"] += 1

    return status


def render_gaps_md(statuses: list[dict], chapters_dir: Path) -> str:
    total_blocks = sum(s["python_blocks"] for s in statuses)
    n_projects = sum(1 for s in statuses if s["is_project"])
    lines = [
        "# GAPS — author-facing extraction audit",
        "",
        "> **PRIVATE.** This file is `.gitignore`'d and must never be committed.",
        "> Regenerate with: `python tools/extract_code.py --report`",
        "",
        f"- Source: `{chapters_dir}`",
        f"- Total Python blocks extracted: **{total_blocks}**",
        f"- Projects detected: **{n_projects}** (expected 35)",
        "",
        "## Per-project status",
        "",
        "| #  | Project | Blocks | Steps | Steps w/code | build.py | break_it.py | step files | Status |",
        "|----|---------|--------|-------|--------------|----------|-------------|------------|--------|",
    ]
    for s in statuses:
        if not s["is_project"]:
            continue
        status_label = classify_status(s)
        lines.append(
            f"| {s['project_n']:2d} | {s['title']} | {s['python_blocks']} | "
            f"{s['step_count']} | {s['steps_with_code']} | "
            f"{'yes' if s['has_build_py'] else 'no'} | "
            f"{'yes' if s['has_break_it_py'] else 'no'} | "
            f"{s['step_files_written']} | {status_label} |"
        )

    non_project_chapters = [s for s in statuses if not s["is_project"]]
    if non_project_chapters:
        lines += [
            "",
            "## Non-project chapters (preface, preflight, etc.)",
            "",
        ]
        for s in non_project_chapters:
            lines.append(f"- `{s['folder']}` — {s['title']} ({s['python_blocks']} blocks)")

    lines += [
        "",
        "## Status legend",
        "",
        "- **complete** — ≥10 python blocks, build.py + break_it.py written, ≥3 steps with code",
        "- **partial** — build.py written but thin (<10 blocks or <3 steps with code)",
        "- **prose-only** — ≤4 python blocks; likely a discussion/motivation chapter that needs hand-crafted code",
        "- **hand-craft** — 0 python blocks in Build; needs original code from author",
        "",
    ]
    return "\n".join(lines)


def classify_status(s: dict) -> str:
    if not s["is_project"]:
        return "n/a"
    if s["python_blocks"] == 0:
        return "hand-craft"
    if s["python_blocks"] <= 4:
        return "prose-only"
    if s["python_blocks"] >= 10 and s["has_build_py"] and s["steps_with_code"] >= 3:
        return "complete"
    return "partial"


def promote(out_dir: Path, projects_dir: Path, force: bool) -> int:
    """Copy _generated/* into projects/*. Refuses to overwrite without --promote-force."""
    if not out_dir.exists():
        print(f"Nothing to promote: {out_dir} does not exist.", file=sys.stderr)
        return 1
    projects_dir.mkdir(exist_ok=True)
    promoted = 0
    for folder in sorted(out_dir.iterdir()):
        if not folder.is_dir():
            continue
        target = projects_dir / folder.name
        if target.exists() and not force:
            print(f"SKIP {target} (already exists; pass --promote-force to overwrite)")
            continue
        if target.exists() and force:
            shutil.rmtree(target)
        shutil.copytree(folder, target)
        print(f"PROMOTED {folder.name}")
        promoted += 1
    print(f"\nPromoted {promoted} projects to {projects_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=DEFAULT_CHAPTERS,
        help=f"Path to chapters/ markdown source (default: {DEFAULT_CHAPTERS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write extracted projects (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and report stats; do not write any files."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Regenerate GAPS.md from the current projects/ state without re-extracting.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After extraction, copy _generated/* into projects/*. Refuses to overwrite without --promote-force.",
    )
    parser.add_argument(
        "--promote-force",
        action="store_true",
        help="With --promote, overwrite existing projects/ folders. Destructive.",
    )
    parser.add_argument(
        "--placeholders-only",
        action="store_true",
        help="Write only per-project README.md files (no step_*.py, build.py, or break_it.py). For PR 1's scaffold-without-code phase.",
    )
    args = parser.parse_args()

    if not args.chapters_dir.exists():
        print(f"ERROR: chapters dir not found: {args.chapters_dir}", file=sys.stderr)
        return 2

    chapter_paths = list(iter_chapter_files(args.chapters_dir))
    if not chapter_paths:
        print(f"ERROR: no ch_*.md files in {args.chapters_dir}", file=sys.stderr)
        return 2

    print(f"Parsing {len(chapter_paths)} chapter files from {args.chapters_dir} ...")
    statuses: list[dict] = []
    chapters: list[Chapter] = []
    for p in chapter_paths:
        ch = parse_chapter(p)
        chapters.append(ch)
        status = {
            "project_n": ch.project_n,
            "title": ch.project_title or ch.h1,
            "folder": ch.folder_name or ch.path.stem,
            "python_blocks": ch.total_python_blocks,
            "step_count": len(ch.build_steps),
            "steps_with_code": sum(1 for s in ch.build_steps if s.python_blocks),
            "has_build_py": False,
            "has_break_it_py": False,
            "step_files_written": 0,
            "is_project": ch.is_project,
        }
        statuses.append(status)
        marker = f"Project {ch.project_n}" if ch.is_project else "(non-project)"
        print(
            f"  {p.name:14s} {marker:14s} {ch.total_python_blocks:3d} blocks, "
            f"{len(ch.build_steps):2d} steps"
        )

    total = sum(s["python_blocks"] for s in statuses)
    print(f"\nTotal Python blocks across all chapters: {total}")
    print(f"Projects detected: {sum(1 for s in statuses if s['is_project'])}")

    if args.dry_run:
        print("\n--dry-run: not writing any files.")
        return 0

    if args.report:
        GAPS_PATH.write_text(render_gaps_md(statuses, args.chapters_dir), encoding="utf-8")
        print(f"\nRegenerated {GAPS_PATH} (from current parse, no extraction performed).")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode = "(placeholders only)" if args.placeholders_only else ""
    print(f"\nExtracting to {args.out_dir}/ {mode}...")
    for ch, status in zip(chapters, statuses, strict=True):
        s = write_chapter(ch, args.out_dir, placeholders_only=args.placeholders_only)
        status.update(s)

    GAPS_PATH.write_text(render_gaps_md(statuses, args.chapters_dir), encoding="utf-8")
    print(f"Wrote {GAPS_PATH} (private, gitignored).")

    if args.promote:
        return promote(args.out_dir, PROJECTS_DIR, force=args.promote_force)

    print(
        f"\nExtraction complete. {sum(1 for s in statuses if s['is_project'])} projects written to {args.out_dir}/."
    )
    print("Promote to projects/ with: python tools/extract_code.py --promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
