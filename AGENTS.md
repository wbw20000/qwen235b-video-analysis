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

