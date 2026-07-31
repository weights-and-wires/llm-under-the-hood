/**
 * Build-time content pipeline.
 *
 * Reads the companion repo (../projects, ../setup, ../README.md) and emits
 * structured JSON the Astro site consumes at build time:
 *
 *   src/content/curriculum.json   — parts, ordering, project summaries (nav)
 *   src/content/projects/<n>.json — full per-project payload (readme sections,
 *                                    build/break code, step files, outputs)
 *   src/content/setup/<slug>.json — setup docs
 *   src/content/book.json         — title, author, leanpub url, method, parts meta
 *
 * Run via: node scripts/build-content.mjs
 * Astro's config also calls this automatically before build (astro.config hook
 * would be cleaner; for now run it as a prebuild / via the integration below).
 *
 * Design: the repo is the single source of truth. No hand-maintained manifest.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "../..");
const PROJECTS_DIR = path.join(REPO_ROOT, "projects");
const SETUP_DIR = path.join(REPO_ROOT, "setup");
const ROOT_README = path.join(REPO_ROOT, "README.md");

const OUT_DIR = path.resolve(__dirname, "../src/content");
const OUT_PROJECTS = path.join(OUT_DIR, "projects");
const OUT_SETUP = path.join(OUT_DIR, "setup");

// ---------------------------------------------------------------------------
// tiny markdown helpers (no deps) — we only need section splitting
// ---------------------------------------------------------------------------

function readFile(p) {
  try {
    return fs.readFileSync(p, "utf8");
  } catch {
    return null;
  }
}

/** Extract the first blockquote (the "Hook") from a project README. */
function extractHook(md) {
  const m = md.match(/^>\s*(.+)$/m);
  if (!m) return null;
  // grab contiguous blockquote lines
  const lines = md.split(/\r?\n/);
  const idx = lines.findIndex((l) => /^>\s/.test(l));
  if (idx === -1) return null;
  const out = [];
  for (let i = idx; i < lines.length; i++) {
    if (/^>\s/.test(lines[i])) {
      out.push(lines[i].replace(/^>\s?/, ""));
    } else if (lines[i].trim() === "") {
      // allow blank lines inside blockquote
      continue;
    } else {
      break;
    }
  }
  return out.join(" ").replace(/^\*|\*$/g, "").trim();
}

/** Extract the H1 title (strip leading "# "). */
function extractTitle(md) {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : null;
}

/**
 * Split markdown into named sections by ## headings.
 * Returns [{ heading, slug, body }] preserving order.
 */
function splitSections(md) {
  const lines = md.split(/\r?\n/);
  const sections = [];
  let current = { heading: "__intro__", slug: "", body: [] };
  for (const line of lines) {
    const m = line.match(/^##\s+(.+?)\s*$/);
    if (m) {
      if (current.body.length) sections.push({ ...current, body: current.body.join("\n") });
      current = { heading: m[1].trim(), slug: slugify(m[1]), body: [] };
    } else if (line.startsWith("# ")) {
      // skip the H1, it's handled separately
      continue;
    } else {
      current.body.push(line);
    }
  }
  if (current.body.length) sections.push({ ...current, body: current.body.join("\n") });
  return sections;
}

function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .trim();
}

/** Natural-sort step files by their numeric prefix: step_08_... < step_10_... */
function stepNumber(name) {
  const m = name.match(/step_(\d+)/);
  return m ? parseInt(m[1], 10) : 9999;
}

/** Humanize a step filename into a title: step_05_build-the-backward-pass.py
 *  -> "Build the backward pass" */
function stepTitle(name) {
  const m = name.match(/^step_\d+_(.+)\.py$/);
  if (!m) return name;
  return m[1]
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// parse root README -> parts + projects
// ---------------------------------------------------------------------------

function parseRootReadme() {
  const md = readFile(ROOT_README) ?? "";
  const lines = md.split(/\r?\n/);

  // locate the "## The 36 Projects" section
  const startIdx = lines.findIndex((l) => /^##\s+The 36 Projects/.test(l));
  if (startIdx === -1) throw new Error("Could not find '## The 36 Projects' in root README");

  const parts = [];
  let currentPart = null;

  for (let i = startIdx + 1; i < lines.length; i++) {
    const line = lines[i];
    // a new part heading (### Title) — but skip the subtitle line "### Build Every..."
    const partMatch = line.match(/^###\s+(.+?)\s*$/);
    if (partMatch) {
      const name = partMatch[1].trim();
      // The very first ### under the H1 is the subtitle; the real parts all
      // come AFTER "## The 36 Projects" and are followed by a table.
      // Heuristic: a real part is immediately followed (ignoring blanks) by a table row.
      let j = i + 1;
      while (j < lines.length && lines[j].trim() === "") j++;
      if (j < lines.length && /^\|/.test(lines[j])) {
        currentPart = { name, slug: slugify(name), projects: [] };
        parts.push(currentPart);
      }
      continue;
    }

    // a project table row. Folder column forms:
    //   | 1 | Title — desc. | [`projects/01_x`](projects/01_x) |
    //   | 1 | Title — desc. | `projects/01_x` |
    // Capture the first projects/<slug> token regardless of wrapping.
    const rowMatch = line.match(/^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*.*?projects\/([A-Za-z0-9_.\-]+).*\|\s*$/);
    if (rowMatch && currentPart) {
      const num = parseInt(rowMatch[1], 10);
      let titleField = rowMatch[2].trim();
      const folder = rowMatch[3].trim();
      // titleField looks like "The Learning Machine — scalar autograd, neurons..."
      // Split on em/en-dash for a clean title + one-line summary.
      let title = titleField;
      let summary = "";
      const dash = titleField.match(/\s+[—–-]\s+/); // em, en, or hyphen
      if (dash) {
        title = titleField.slice(0, dash.index).trim();
        summary = titleField.slice(dash.index + dash[0].length).trim();
      }
      currentPart.projects.push({ num, title, summary, folder });
    }

    // stop at the next H2
    if (/^##\s+/.test(line) && i > startIdx + 1) break;
  }

  return parts;
}

// ---------------------------------------------------------------------------
// per-project payload
// ---------------------------------------------------------------------------

function buildProjectPayload(part, proj) {
  const dir = path.join(PROJECTS_DIR, proj.folder);
  const readmePath = path.join(dir, "README.md");
  const readme = readFile(readmePath) ?? "";

  const readmeTitle = extractTitle(readme) ?? `Project ${proj.num}: ${proj.title}`;
  const hook = extractHook(readme) ?? proj.summary ?? "";
  const sections = splitSections(readme);

  // catalog files
  let entries = [];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    entries = [];
  }

  const buildPath = entries.find((e) => e.isFile() && e.name === "build.py");
  const breakPath = entries.find((e) => e.isFile() && e.name === "break_it.py");
  const stepFiles = entries
    .filter((e) => e.isFile() && /^step_\d+.*\.py$/.test(e.name))
    .map((e) => e.name)
    .sort((a, b) => stepNumber(a) - stepNumber(b) || a.localeCompare(b));

  // outputs dir
  const outputsDir = path.join(dir, "outputs");
  let outputs = [];
  if (fs.existsSync(outputsDir)) {
    outputs = fs
      .readdirSync(outputsDir)
      .filter((f) => !f.startsWith("."))
      .map((f) => {
        const full = path.join(outputsDir, f);
        const ext = path.extname(f).toLowerCase();
        const kind = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"].includes(ext)
          ? "image"
          : "text";
        return {
          name: f,
          kind,
          // text content inlined for display; images are copied & referenced
          content: kind === "text" ? (readFile(full) ?? "").trim() : null,
        };
      });
  }

  // tests dir
  const testsDir = path.join(dir, "tests");
  let tests = [];
  if (fs.existsSync(testsDir)) {
    tests = fs.readdirSync(testsDir).filter((f) => f.endsWith(".py")).map((f) => ({
      name: f,
      content: readFile(path.join(testsDir, f)) ?? "",
    }));
  }

  return {
    num: proj.num,
    folder: proj.folder,
    part: { name: part.name, slug: part.slug },
    title: proj.title,
    summary: proj.summary,
    readmeTitle,
    hook,
    sections,
    files: {
      build: buildPath
        ? { name: "build.py", content: readFile(path.join(dir, "build.py")) ?? "" }
        : null,
      breakIt: breakPath
        ? { name: "break_it.py", content: readFile(path.join(dir, "break_it.py")) ?? "" }
        : null,
      steps: stepFiles.map((name) => ({
        name,
        title: stepTitle(name),
        n: stepNumber(name),
        content: readFile(path.join(dir, name)) ?? "",
      })),
    },
    outputs,
    tests,
    hasBreakIt: !!breakPath,
    hasOutputs: outputs.length > 0,
    hasTests: tests.length > 0,
  };
}

// ---------------------------------------------------------------------------
// setup docs
// ---------------------------------------------------------------------------

function buildSetupDocs() {
  if (!fs.existsSync(SETUP_DIR)) return [];
  const files = fs
    .readdirSync(SETUP_DIR)
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .sort();

  return files.map((f) => {
    const md = readFile(path.join(SETUP_DIR, f));
    const title = extractTitle(md) ?? f.replace(/\.md$/, "");
    const sections = splitSections(md);
    return {
      slug: f.replace(/\.md$/, ""),
      filename: f,
      title,
      body: md,
      sections,
    };
  });
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeJson(p, obj) {
  fs.writeFileSync(p, JSON.stringify(obj, null, 2));
}

function main() {
  console.log("[content] reading repo at", REPO_ROOT);

  // fresh output
  fs.rmSync(OUT_DIR, { recursive: true, force: true });
  ensureDir(OUT_DIR);
  ensureDir(OUT_PROJECTS);
  ensureDir(OUT_SETUP);

  // copy project outputs (images) into public so they're servable
  const publicOutputs = path.resolve(__dirname, "../public/outputs");
  fs.rmSync(publicOutputs, { recursive: true, force: true });
  ensureDir(publicOutputs);

  const parts = parseRootReadme();
  console.log("[content] parsed %d parts", parts.length);

  const curriculum = []; // summary tree for nav
  let count = 0;

  for (const part of parts) {
    const partSummary = { name: part.name, slug: part.slug, projects: [] };
    for (const proj of part.projects) {
      const payload = buildProjectPayload(part, proj);

      // copy image outputs to public
      for (const o of payload.outputs) {
        if (o.kind === "image") {
          const src = path.join(PROJECTS_DIR, proj.folder, "outputs", o.name);
          // namespace by project number to avoid collisions
          const destDir = path.join(publicOutputs, String(proj.num));
          ensureDir(destDir);
          fs.copyFileSync(src, path.join(destDir, o.name));
        }
      }

      writeJson(path.join(OUT_PROJECTS, `${proj.num}.json`), payload);
      partSummary.projects.push({
        num: proj.num,
        folder: proj.folder,
        title: payload.title,
        hook: payload.hook,
        summary: payload.summary,
        slug: proj.folder,
        hasBreakIt: payload.hasBreakIt,
        hasOutputs: payload.hasOutputs,
      });
      count++;
    }
    curriculum.push(partSummary);
  }

  writeJson(path.join(OUT_DIR, "curriculum.json"), curriculum);

  const setup = buildSetupDocs();
  for (const doc of setup) {
    writeJson(path.join(OUT_SETUP, `${doc.slug}.json`), doc);
  }

  const book = {
    title: "Under the Hood",
    subtitle: "Build Every Layer of a Large Language Model from Scratch",
    author: "Ramchand Kumaresan",
    leanpubUrl: "https://leanpub.com/under-the-hood",
    githubUrl: "https://github.com/mechramc/Under-the-hood",
    method: ["Build it.", "Break it.", "Measure it."],
    projectCount: count,
    partCount: parts.length,
  };
  writeJson(path.join(OUT_DIR, "book.json"), book);

  console.log(
    "[content] wrote %d projects across %d parts, %d setup docs",
    count,
    parts.length,
    setup.length
  );
}

main();
