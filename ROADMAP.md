# Roadmap

Planned features for Silence Remover, to be built **one at a time**, each on
its own branch with tests, a pull request and green CI, the same way the audio
extractor and swap-audio features were added.

- **Effort:** S = a few hours, M = about a day, L = several days, XL = a week or more.
- **Status:** ☐ planned · ◐ in progress · ☑ done. Update the status in the same
  PR that ships the feature.
- **Issues:** every item has a GitHub issue with labels `enhancement`,
  `phase-N: …` and `effort: …`, grouped into one milestone per phase. Put
  `Closes #N` in the PR description so merging closes the issue.

---

## Contents

- [How we add a feature](#how-we-add-a-feature)
- [Overview](#overview)
- [Phase 1: Quick wins](#phase-1-quick-wins)
- [Phase 2: Preview and editing](#phase-2-preview-and-editing)
- [Phase 3: Audio quality](#phase-3-audio-quality)
- [Phase 4: Speed and scale](#phase-4-speed-and-scale)
- [Phase 5: AI features (Whisper)](#phase-5-ai-features-whisper)
- [Phase 6: Distribution](#phase-6-distribution)
- [Technical debt and known issues](#technical-debt-and-known-issues)
- [Already shipped](#already-shipped)

---

## How we add a feature

1. **Pick** the next item, mark it ◐ here, and create a branch:
   `git checkout main && git pull && git checkout -b feature/<short-name>`.
2. **Core first:** put the logic in its own module under `silence_remover/`,
   free of Qt, so it can be unit tested (as with `audio_export.py` and
   `swap_audio.py`).
3. **Tests** alongside it in `tests/`, including at least one that checks the
   *content* of the output, not just that a file exists. That check would have
   caught the swap-audio CLI bug.
4. **Wire it up:** a CLI flag in `cli.py`, then GUI controls in
   `gui/main_window.py` with a background worker in `gui/workers.py`.
5. **Docs:** update the README (features list, GUI section, CLI options table,
   examples) and mark the item ☑ here.
6. **Check locally:** `ruff check .` and `python -m pytest -q`.
7. **PR → CI green → merge into `main`.** If you merge `main` into your branch
   first, run lint and the tests again after resolving conflicts.

---

## Overview

| # | Feature | Phase | Effort | Status | Issue |
|---|---|---|---|---|---|
| 1.1 | Fix the empty timeline showing `0:00.0` everywhere | 1 | S | ☐ | [#3](https://github.com/ifahimkhan/Video-Editor/issues/3) |
| 1.2 | Audio offset control for Swap audio in the GUI | 1 | S | ☐ | [#4](https://github.com/ifahimkhan/Video-Editor/issues/4) |
| 1.3 | More audio export formats and a bitrate choice in the GUI | 1 | S | ☐ | [#5](https://github.com/ifahimkhan/Video-Editor/issues/5) |
| 1.4 | Remember settings and last folder | 1 | S | ☐ | [#6](https://github.com/ifahimkhan/Video-Editor/issues/6) |
| 1.5 | Presets (Podcast, Lecture, Vlog, Aggressive) | 1 | S | ☐ | [#7](https://github.com/ifahimkhan/Video-Editor/issues/7) |
| 1.6 | Drag and drop files onto the window | 1 | S | ☐ | [#8](https://github.com/ifahimkhan/Video-Editor/issues/8) |
| 1.7 | Keyboard shortcuts | 1 | S | ☐ | [#9](https://github.com/ifahimkhan/Video-Editor/issues/9) |
| 2.1 | Playback preview with cuts skipped | 2 | L | ☐ | [#10](https://github.com/ifahimkhan/Video-Editor/issues/10) |
| 2.2 | Manual cut editing on the timeline | 2 | L | ☐ | [#11](https://github.com/ifahimkhan/Video-Editor/issues/11) |
| 2.3 | Undo / redo for edits | 2 | M | ☐ | [#12](https://github.com/ifahimkhan/Video-Editor/issues/12) |
| 2.4 | Export cut list to other editors (EDL, FCPXML, Resolve) | 2 | M | ☐ | [#13](https://github.com/ifahimkhan/Video-Editor/issues/13) |
| 3.1 | Fades at cuts to prevent clicks | 3 | M | ☐ | [#14](https://github.com/ifahimkhan/Video-Editor/issues/14) |
| 3.2 | Built-in noise reduction | 3 | M | ☐ | [#15](https://github.com/ifahimkhan/Video-Editor/issues/15) |
| 3.3 | Loudness normalization (EBU R128) | 3 | M | ☐ | [#16](https://github.com/ifahimkhan/Video-Editor/issues/16) |
| 3.4 | Speed up pauses instead of cutting them | 3 | M | ☐ | [#17](https://github.com/ifahimkhan/Video-Editor/issues/17) |
| 3.5 | Choose the audio track for multi-track files | 3 | S | ☐ | [#18](https://github.com/ifahimkhan/Video-Editor/issues/18) |
| 4.1 | Export quality settings (CRF, preset, resolution) | 4 | S | ☐ | [#19](https://github.com/ifahimkhan/Video-Editor/issues/19) |
| 4.2 | Hardware-accelerated encoding (NVENC, QSV, AMF) | 4 | M | ☐ | [#20](https://github.com/ifahimkhan/Video-Editor/issues/20) |
| 4.3 | Batch queue in the GUI | 4 | M | ☐ | [#21](https://github.com/ifahimkhan/Video-Editor/issues/21) |
| 5.1 | Filler word removal ("um", "uh") | 5 | L | ☐ | [#22](https://github.com/ifahimkhan/Video-Editor/issues/22) |
| 5.2 | Automatic subtitles (SRT, optional burn-in) | 5 | L | ☐ | [#23](https://github.com/ifahimkhan/Video-Editor/issues/23) |
| 5.3 | Transcript-based editing | 5 | XL | ☐ | [#24](https://github.com/ifahimkhan/Video-Editor/issues/24) |
| 6.1 | Windows installer / standalone .exe with FFmpeg bundled | 6 | M | ☐ | [#25](https://github.com/ifahimkhan/Video-Editor/issues/25) |
| 6.2 | Log file and friendlier error reports | 6 | S | ☐ | [#26](https://github.com/ifahimkhan/Video-Editor/issues/26) |
| 6.3 | macOS CI and packaging | 6 | M | ☐ | [#27](https://github.com/ifahimkhan/Video-Editor/issues/27) |

The suggested order is top to bottom: Phase 1 items are small and independent,
which makes them good practice for the workflow. Phase 2.1 (playback preview)
is the biggest single improvement to day-to-day use.

---

## Phase 1: Quick wins

### 1.1 Fix the empty timeline showing `0:00.0` everywhere ☐ · S

**Why.** Before a file is opened, every time label reads `0:00.0` (visible in
the README screenshot), because the empty cut curve spans only 0.001 s.

**How.** In `WaveformTimeline.set_profile(None)`, set the view to a fixed
range such as 0–60 s and draw the placeholder text "Open a video to see its
waveform".

**Touches.** `gui/waveform_timeline.py`, `tests/test_gui.py`.

**Done when.** A new window shows sensible, distinct axis labels, and a test
checks the view range when no file is loaded.

### 1.2 Audio offset control for Swap audio in the GUI ☐ · S

**Why.** `--audio-offset` exists only in the CLI. Noise-removal tools
sometimes add a small delay, and GUI users need a way to correct it.

**How.** After the clean audio is picked, show a small dialog with an offset
spin box (−2000 to 2000 ms, step 10). Pass it to `SwapAudioWorker`, which
already reaches `swap_audio(offset_ms=…)`.

**Touches.** `gui/main_window.py`, `gui/workers.py`, `tests/test_gui.py`.

**Done when.** The offset entered in the GUI shifts the audio; a test checks
that the start of the output is silent for a positive offset.

### 1.3 More audio export formats and a bitrate choice in the GUI ☐ · S

**Why.** The extractor only writes 192 kbps MP3. Editors often want WAV
(lossless) and podcasters want AAC/M4A or FLAC.

**How.** Generalize `audio_export.extract_mp3` into `extract_audio(format,
quality)`:

| Format | Encoder |
|---|---|
| MP3 | `libmp3lame` |
| M4A | `aac` |
| WAV | `pcm_s16le` |
| FLAC | `flac` |
| Original | stream copy, when the container allows it |

Pick the format from the save dialog's file-type filter. Keep `--extract-audio`
backward compatible and add `--audio-format`.

**Touches.** `audio_export.py`, `cli.py`, `gui/main_window.py`, README, tests.

**Done when.** Each format round-trips through `ffprobe` with the right codec
and full duration.

### 1.4 Remember settings and last folder ☐ · S

**Why.** Every launch resets the sliders and starts file dialogs in the
default folder.

**How.** Use `QSettings("SilenceRemover", "SilenceRemover")`. Save the
detection mode, slider values, export mode and last-used folders on close, and
restore them on start. Ignore stored values that fall outside the current
ranges.

**Touches.** `gui/main_window.py`, a new `gui/settings.py`, tests (with an
isolated QSettings path).

**Done when.** Values survive a restart, and corrupted or out-of-range values
fall back to the defaults.

### 1.5 Presets ☐ · S

**Why.** New users don't know good values. One-click starting points help.

**How.** Keep a table of named `DetectionSettings` in `segments.py` or a
`presets.py`, for example:

| Preset | Settings |
|---|---|
| Podcast | voice, 0.5, 700 ms, 200 ms |
| Lecture | loudness −38 dB, 1000 ms, 300 ms |
| Vlog | voice, 0.4, 400 ms, 150 ms |
| Aggressive | loudness −30 dB, 300 ms, 80 ms |

Add a preset dropdown above the sliders and `--preset NAME` in the CLI;
explicit flags override the preset.

**Done when.** Selecting a preset moves all the sliders, and the CLI preset
plus an override produces the expected settings in a unit test.

### 1.6 Drag and drop files onto the window ☐ · S

**How.** `setAcceptDrops(True)`, plus `dragEnterEvent` / `dropEvent` on the
main window that accept a single local file with a known media extension and
reuse the `_choose_input` flow.

**Done when.** Dropping a file starts analysis; unsupported files are refused
with a message.

### 1.7 Keyboard shortcuts ☐ · S

**How.** `QShortcut` / `QAction` bindings:

| Shortcut | Action |
|---|---|
| Ctrl+O | Open |
| Ctrl+E | Export |
| Ctrl+Shift+E | Export audio |
| Esc | Cancel |
| Ctrl+Plus / Ctrl+Minus / Ctrl+0 | Zoom in / zoom out / zoom to fit |

Show the shortcuts in the tooltips.

**Done when.** Each shortcut triggers its action in an offscreen GUI test.

---

## Phase 2: Preview and editing

### 2.1 Playback preview with cuts skipped ☐ · L

**Why.** Users currently have to export to hear the result. Previewing is the
biggest missing piece compared with Filmora.

**How.**
- Use `QMediaPlayer` + `QVideoWidget` (PyQt6 multimedia) above the waveform.
- Add a playhead line on the timeline that follows playback. Clicking the
  timeline seeks.
- Add a "Skip cuts" toggle. While playing, when the position enters a cut,
  `setPosition` jumps to the next kept segment. Look this up by binary search
  on the kept starts; it's the same data as `DetectionResult.keep`.
- Add Space for play/pause and ←/→ for ±5 s.

**Risks.** Qt multimedia codec support differs by OS. On Windows it uses Media
Foundation, which covers H.264/AAC. Fall back gracefully, and show a message
if a file can't be played.

**Touches.** A new `gui/player.py`, `gui/waveform_timeline.py` (playhead,
click-to-seek), `gui/main_window.py`, tests (logic for the next kept position;
player behaviour mocked).

**Done when.** Playback skips every red region, and the playhead stays in step
within about 100 ms.

### 2.2 Manual cut editing on the timeline ☐ · L

**Why.** Automatic detection is never perfect. Users need to keep a pause for
effect or cut a cough.

**How.**
- Store a list of user overrides next to the settings: *force keep* ranges and
  *force cut* ranges.
- Add a pure function `apply_overrides(result, overrides) -> DetectionResult`
  in `segments.py`, applied after `detect()` and before keyframe snapping.
  Overrides survive slider changes.
- Timeline interaction: click a red region to toggle it to kept, and
  Shift+drag to mark a new cut. Draw overrides with a different outline
  colour.
- Add a side panel listing segments with start and end times, editable
  numerically.

**Touches.** `segments.py` (pure, heavily unit tested), `gui/waveform_timeline.py`,
`gui/main_window.py`.

**Done when.** Overrides show in the preview, stats and both export modes, and
the property tests pass: output segments are sorted and non-overlapping, and
forced ranges are respected.

### 2.3 Undo / redo for edits ☐ · M

**How.** Use `QUndoStack`, with each override change or slider change as a
command. The settings and overrides are immutable dataclasses, so a command
simply stores the old and new values. Bind Ctrl+Z / Ctrl+Y.

**Depends on.** 2.2.

### 2.4 Export cut list to other editors ☐ · M

**Why.** Many users finish their edit in Premiere, DaVinci Resolve or Final
Cut. A cut list lets them import the result without re-encoding anything.

**How.** Add a new `edl_export.py` that writes the kept segments as:

| Format | For |
|---|---|
| CMX3600 EDL | universal |
| FCPXML 1.9 | Final Cut, Resolve |
| CSV of in/out times | anything else |

Frame-accurate timecodes need the source frame rate from `ffprobe`
(`r_frame_rate`). The CLI gets `--export-edl FILE`.

**Done when.** Timecodes match the kept segments at 23.976, 25, 29.97
(drop-frame) and 30 fps. The generated files validate: parse the FCPXML with
an XML parser, and check EDL event counts.

---

## Phase 3: Audio quality

### 3.1 Fades at cuts to prevent clicks ☐ · M

**Why.** A hard cut in the middle of a waveform can click audibly.

**How.** In `render.build_filter_graph`, multiply the audio by a short ramp
(5–20 ms) at every join. Use a `volume` expression built with `sum_expr` so
large cut counts stay under FFmpeg's expression limit. This is precise mode
only; lossless can't change the audio. Add a "Fade length" setting (0 turns
it off).

**Done when.** A test measures that the sample-to-sample jump at every join
stays below a threshold with fades on, and exceeds it with fades off.

### 3.2 Built-in noise reduction ☐ · M

**Why.** A companion to swap audio: clean the audio without leaving the app.

**How.** Add an optional FFmpeg audio filter in both the precise export and
the audio extractor:
- `afftdn` (FFT denoiser, always available), or
- `arnndn` with a bundled RNNoise model, which gives better results for
  speech.

Offer a Strength setting (off / light / medium / strong). Analyze on the raw
audio, but apply the filter on export.

**Done when.** A tone-plus-noise test file has a measurably lower noise floor
after export, and the speech level is roughly unchanged.

### 3.3 Loudness normalization (EBU R128) ☐ · M

**Why.** Consistent volume across videos; YouTube and podcasts target
−14 / −16 LUFS.

**How.** Two-pass `loudnorm`:
1. Measure with `loudnorm=print_format=json` on the kept audio.
2. Apply with the measured values.

Offer a target setting (−14 / −16 / −23 LUFS). Store the measurement so it can
be shown in the stats line.

**Done when.** Integrated loudness of the output measures within ±1 LU of the
target.

### 3.4 Speed up pauses instead of cutting them ☐ · M

**Why.** Cutting every pause can sound rushed. Descript-style "shorten gaps"
keeps a natural rhythm.

**How.** A new mode where every silence longer than the minimum is shortened
to a fixed length (for example 300 ms) rather than removed. This reuses the
existing machinery: keep a fixed-length slice from the middle of each silence.
It's mostly segment arithmetic in `segments.py`.

**Done when.** Every silence in the output measures within ±20 ms of the
target gap.

### 3.5 Choose the audio track for multi-track files ☐ · S

**Why.** Screen recorders (OBS) often write the microphone and desktop audio
as separate tracks. Today the first track is always used.

**How.** List the audio streams with `ffprobe` (index, language, title,
channels) and add a track dropdown after opening a file. Pass `0:a:N` through
analysis, the renderers, the extractor and swap. The CLI gets
`--audio-track N`.

**Done when.** A two-track test file (tone on track 0, silence on track 1)
gives different detections depending on the selected track.

---

## Phase 4: Speed and scale

### 4.1 Export quality settings ☐ · S

**How.** Expose `EncoderSettings` (CRF, preset, audio bitrate) in the GUI as
an "Export settings…" dialog, plus an optional downscale (1080p / 720p). Add
matching CLI flags: `--crf`, `--preset`, `--max-height`.

### 4.2 Hardware-accelerated encoding ☐ · M

**Why.** Precise export re-encodes on the CPU; GPU encoders are 3–10× faster.

**How.** At startup, detect encoders that actually work by test-encoding one
frame with each of `h264_nvenc`, `h264_qsv` and `h264_amf`. Offer the working
ones as choices, and fall back to `libx264` if encoding fails.

**Done when.** The detection logic is unit tested with mocked FFmpeg output,
and a CI test confirms the fallback path.

### 4.3 Batch queue in the GUI ☐ · M

**Why.** The CLI can batch process, but GUI users can't.

**How.** Add a queue panel: add files, and apply the current settings to each
file. A worker processes them one at a time with per-file progress and error
status, and the queue continues past failures. Output names follow a pattern
(`{name}_cut.{ext}`).

**Depends on.** Ideally 1.4 (settings) and 1.5 (presets).

---

## Phase 5: AI features (Whisper)

These build on a transcript from `faster-whisper`, which runs on the CPU via
CTranslate2 and gives word-level timestamps. It adds a sizeable download (the
library plus a model of 75 MB or more), so make it an optional dependency, the
same approach as VAD: the features are greyed out when it isn't installed.

### 5.1 Filler word removal ☐ · L

**How.** Transcribe with word-level timestamps and mark words matching a
configurable list ("um", "uh", "erm", "you know"…) as forced cuts. Add a
toggle and a word list in the GUI.

**Depends on.** 2.2 (overrides), which is the natural way to merge
transcript cuts with silence cuts.

### 5.2 Automatic subtitles ☐ · L

**How.** Transcribe, then write SRT/VTT. When silence is removed, the
subtitle timings must be remapped through the same "time removed before T"
function the renderer uses. Optionally burn the subtitles in with the
`subtitles` filter in precise mode.

**Done when.** The remapped subtitle times stay in sync with the cut video
(checked in a test against known segment boundaries).

### 5.3 Transcript-based editing ☐ · XL

**Why.** A Descript-style workflow: delete text to cut the video.

**How.** A transcript panel next to the timeline. Selecting and deleting words
creates forced cuts, clicking a word seeks the preview, and the transcript and
timeline selections stay linked.

**Depends on.** 2.1, 2.2, 5.1.

---

## Phase 6: Distribution

### 6.1 Windows installer / standalone .exe ☐ · M

**Why.** Users currently need Python, a virtual environment and FFmpeg on
`PATH`.

**How.** PyInstaller one-folder build that bundles FFmpeg/ffprobe (LGPL/GPL
licence notes included) and the Silero model. Look for bundled binaries next
to the executable before `PATH` (`ffmpeg_tools.find_binary`). Wrap the build
in an installer (Inno Setup), and add a CI job that builds it and uploads it
as a release artifact when a version tag is pushed.

**Done when.** A clean Windows machine with no Python or FFmpeg can install
and run the app, and CI produces the installer on tag push.

### 6.2 Log file and friendlier error reports ☐ · S

**How.** Use the `logging` module, writing to
`%LOCALAPPDATA%/SilenceRemover/logs`, and record the FFmpeg command lines and
stderr tails there. Error dialogs show a short message plus "Open log folder".

### 6.3 macOS CI and packaging ☐ · M

**How.** Add `macos-latest` to the CI matrix, installing FFmpeg via Homebrew
or a static build. Later, add a `.app` bundle via PyInstaller.

---

## Technical debt and known issues

Worth fixing opportunistically, or before the feature that touches the same
code.

| Issue | Where | Suggested fix |
|---|---|---|
| The app is now a general video editor, but the package, command and window title still say "silence remover" | `silence_remover/`, `__main__.py`, `gui/main_window.py`, README, CI | Pick a product name. Rename the window title first (trivial), then the package (for example `video_editor`), keeping a `silence_remover` alias so `python -m silence_remover` still works. Rename `SILENCE_REMOVER_VAD_MODEL`, accepting the old name too. |
| The CLI switches between jobs with a growing chain of `if` flags; a missing `return` already caused one bug | `cli.py` | Move to argparse subcommands (`cut`, `extract-audio`, `swap-audio`) and keep the old flags as aliases |
| Video tail is dropped when the audio track is shorter than the video | `analysis.py`, `segments.py` | Use the container duration for the final segment |
| Lossless segments may end 0.1–0.2 s late (whole groups of frames) | `lossless.py` | Document it, or offer "snap ends back" as an option |
| README images live inside the Python package (`silence_remover/docs/`) | repo layout | Move them to a top-level `docs/` and update the links |
| Compiled `.pyc` files remain in the first commit's history | git history | Leave it (harmless); rewriting history isn't worth it |
| The tests only cover Windows and Linux | CI | See 6.3 |
| Only the first video and audio streams are used | all modules | See 3.5 |

---

## Already shipped

| Feature | Where |
|---|---|
| Loudness-based silence detection with threshold, minimum silence and padding | `analysis.py`, `segments.py` |
| Voice detection with Silero VAD | `vad.py` |
| Precise export (single-pass re-encode, A/V sync, supports thousands of cuts) | `render.py` |
| Fast lossless export (keyframe-snapped stream copy) | `lossless.py` |
| Interactive pyqtgraph waveform with draggable thresholds and zoom | `gui/waveform_timeline.py` |
| MP3 audio extractor | `audio_export.py` · PR #1 |
| Swap audio (replace the soundtrack, video untouched) | `swap_audio.py` · PR #2 |
| Remove echo (doubled voice from two microphones recording at once) | `echo_dsp.py`, `echo_removal.py` |
| CLI covering every feature | `cli.py` |
| CI: lint plus tests on Ubuntu and Windows, 80% coverage gate | `.github/workflows/ci.yml` |
