# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Flask web app entry; handles uploads, FFmpeg compression, and Qwen3‑VL API calls with SSE progress.
- `templates/index.html`: Main UI template.
- `uploads/`: Runtime folder for uploaded videos (created on demand).
- `requirements.txt`: Python dependencies.
- `run.bat`, `start.bat`: Windows helpers for local runs.
- `test_setup.py`: Environment sanity checks (Python, deps, CUDA, network).

## Build, Test, and Development Commands
- Setup (venv recommended): `python -m venv venv && venv\Scripts\pip install -r requirements.txt` (Windows) or `source venv/bin/activate && pip install -r requirements.txt`.
- Configure API key: `set DASHSCOPE_API_KEY=...` (Windows) or `export DASHSCOPE_API_KEY=...` (Linux/macOS).
- Run app: `python app.py` or `run.bat` / `start.bat` on Windows.
- Env check: `python test_setup.py` (verifies Python, deps, CUDA, folders, network).

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Templates: keep logic light; prefer view code in `app.py`. Use descriptive `id`/`class` names.
- Filenames: app modules as `*.py`, tests as `test_*.py`, Windows scripts as `*.bat`.
- Lint/format: no enforced tool; follow PEP 8 and keep imports sorted.
- Localization: user‑facing text is Simplified Chinese; keep additions consistent.

## Testing Guidelines
- Current checks live in `test_setup.py`. Run it after dependency changes.
- For new tests, use `pytest` with files named `test_*.py`. Co‑locate near code or add a `tests/` folder. Aim to cover API routes and FFmpeg helper functions.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (e.g., `feat: add SSE progress`, `fix(ffmpeg): handle NVENC check`). Keep messages in English; include context.
- PRs: add a clear description, linked issues, repro steps, and screenshots/GIFs for UI changes. Note any new env vars or external requirements (FFmpeg, CUDA).

## Security & Configuration Tips
- Do not commit real API keys. Use the `DASHSCOPE_API_KEY` environment variable locally and in deployment.
- FFmpeg is required on PATH; NVENC is optional but accelerates compression.
- Avoid logging secrets; redact keys in errors and console output.

<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **qwen235b** (41082 symbols, 45369 relationships, 300 execution flows).

GitNexus provides a knowledge graph over this codebase — call chains, blast radius, execution flows, and semantic search.

## Always Start Here

For any task involving code understanding, debugging, impact analysis, or refactoring, you must:

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/refactoring/SKILL.md` |

## Tools Reference

| Tool | What it gives you |
|------|-------------------|
| `query` | Process-grouped code intelligence — execution flows related to a concept |
| `context` | 360-degree symbol view — categorized refs, processes it participates in |
| `impact` | Symbol blast radius — what breaks at depth 1/2/3 with confidence |
| `detect_changes` | Git-diff impact — what do your current changes affect |
| `rename` | Multi-file coordinated rename with confidence-tagged edits |
| `cypher` | Raw graph queries (read `gitnexus://repo/{name}/schema` first) |
| `list_repos` | Discover indexed repos |

## Resources Reference

Lightweight reads (~100-500 tokens) for navigation:

| Resource | Content |
|----------|---------|
| `gitnexus://repo/{name}/context` | Stats, staleness check |
| `gitnexus://repo/{name}/clusters` | All functional areas with cohesion scores |
| `gitnexus://repo/{name}/cluster/{clusterName}` | Area members |
| `gitnexus://repo/{name}/processes` | All execution flows |
| `gitnexus://repo/{name}/process/{processName}` | Step-by-step trace |
| `gitnexus://repo/{name}/schema` | Graph schema for Cypher |

## Graph Schema

**Nodes:** File, Function, Class, Interface, Method, Community, Process
**Edges (via CodeRelation.type):** CALLS, IMPORTS, EXTENDS, IMPLEMENTS, DEFINES, MEMBER_OF, STEP_IN_PROCESS

```cypher
MATCH (caller)-[:CodeRelation {type: 'CALLS'}]->(f:Function {name: "myFunc"})
RETURN caller.name, caller.filePath
```

<!-- gitnexus:end -->
