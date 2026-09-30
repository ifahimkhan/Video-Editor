# Contributing to Video Editor

Thanks for your interest in helping! This project aims to make editing
spoken-word videos automatic: record, drop in the file, and get back a clean,
well-paced video. Contributions of every size are welcome, from fixing a typo
to building a whole feature.

## Ways to help

- **Build a planned feature.** Every item in [ROADMAP.md](ROADMAP.md) has an
  [issue](https://github.com/ifahimkhan/Video-Editor/issues) with a
  description, the files involved and a definition of done. Issues labelled
  [`good first issue`](https://github.com/ifahimkhan/Video-Editor/labels/good%20first%20issue)
  are small and self-contained.
- **Report a bug.** Open an issue with the steps to reproduce it, what you
  expected, what happened, your OS, Python version and `ffmpeg -version`
  output, and any error message.
- **Test it on your own videos** and tell us where the results are good or bad.
  Real recordings find problems that test clips don't.
- **Suggest a feature.** Open an issue describing the problem it solves. Tell
  us what part of editing takes you the most time.
- **Improve the docs.** If something in the README confused you, it will
  confuse others.

## Before you start

- **Comment on the issue** you want to work on, so nobody duplicates your work.
  A maintainer will assign it to you.
- **For big changes** (anything marked L or XL in the roadmap, or a feature
  that isn't on it), describe your plan in the issue first. It's much easier
  to agree on an approach before the code is written.

## Development setup

You need **Python 3.10+** and **FFmpeg 7 or newer** on your `PATH`
(`ffmpeg -version`). On Windows: `winget install Gyan.FFmpeg`.

```bash
git clone https://github.com/<your-username>/Video-Editor.git   # your fork
cd Video-Editor
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

python -m pytest -q             # all tests should pass
ruff check .                    # lint should be clean
python -m silence_remover       # run the editor
```

If `pip` fails with `CERTIFICATE_VERIFY_FAILED` on a work or school network,
see Troubleshooting in the [README](README.md#troubleshooting).

## How we add a feature

Every feature so far was built this way; please follow the same steps.

1. **Branch** from an up-to-date `main`:
   `git checkout main && git pull && git checkout -b feature/<short-name>`
   (use `fix/<short-name>` for bug fixes).
2. **Core logic first**, in its own module under `silence_remover/` with no Qt
   imports, so it can be unit tested. See `audio_export.py` and
   `swap_audio.py` for small examples.
3. **Tests** in `tests/`. Include at least one test that checks the **content**
   of the output (duration, loudness, codec, frames), not just that a file was
   written. A file-exists-only test once let a real bug through.
4. **Wire it up:** a CLI option in `cli.py`, and for GUI features, a control in
   `gui/main_window.py` with a background worker in `gui/workers.py`.
5. **Docs:** update the README (tools table, usage section, CLI options) and
   mark the item ☑ in `ROADMAP.md`.
6. **Check locally:** `ruff check .` and `python -m pytest -q`.
7. **Open a pull request** against `main`.

## Code conventions

- **Style:** `ruff check .` must pass (rules and the 100-character line length
  are set in `ruff.toml`). Use type hints on function signatures.
- **Keep Qt out of the core.** Apart from the `__main__.py` launcher, modules
  outside `gui/` must not import PyQt6. The GUI and CLI are thin layers over
  them.
- **Prefer immutable data:** frozen dataclasses and tuples, and functions that
  return new values instead of modifying their inputs. See `segments.py`.
- **Run FFmpeg through the existing helpers:**
  - `ffmpeg_tools.find_binary` and `probe` to locate FFmpeg and inspect files.
  - `render.run_with_progress` for long jobs, so progress, cancel and error
    messages work the same everywhere.
  - Pass arguments as a list (never a shell string).
- **Long FFmpeg expressions:** build sums with `render.sum_expr`, because
  FFmpeg rejects flat expressions of more than 100 terms.
- **Errors:** raise `FFmpegError` or `ValueError` with a message a user can act
  on; the GUI shows it as-is.
- **Small, focused files and functions.** If a module grows past a few hundred
  lines, split it.

## Tests

- `pytest` with plain `assert`s. Integration tests generate their own media
  with FFmpeg (see `make_clip` in `tests/conftest.py`), so there's no need to
  add large binary fixtures.
- Mark tests that need FFmpeg with `@requires_ffmpeg`.
- GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`). See `tests/test_gui.py`
  for driving the window and waiting for background workers.
- Coverage must stay at or above 80% (`python -m pytest --cov=silence_remover`).

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary in the imperative>

<optional body: what changed and why>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`.
For example: `feat: add drag and drop to open files`.

## Pull requests

- **One feature or fix per PR.** Small PRs are reviewed much faster.
- Put `Closes #<issue>` in the description, so merging closes the issue.
- Describe what changed, how you tested it, and include a screenshot for GUI
  changes.
- **CI must be green.** It runs lint and the tests on Ubuntu (Python 3.10,
  3.12, 3.13) and Windows (3.12).
- **If you merge `main` into your branch** to resolve conflicts, run
  `ruff check .` and the tests again before pushing. Conflict resolutions are
  where bugs sneak in.

## Licensing

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE). Only add code or assets you have the right to
share. If you add a third-party model, asset or bundled dependency, add its
license to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Be kind

Be respectful and constructive in issues, reviews and discussions. Everyone
here is learning, and questions are welcome. There are no silly ones.
