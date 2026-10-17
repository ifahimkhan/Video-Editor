"""Command-line options and runners for `--reel` (one range) and `--reels` (auto)."""

from __future__ import annotations

import argparse
from typing import Callable

from ..analysis import analyze
from ..ffmpeg_tools import probe
from ..segments import detect
from ..transcript import speech_captions
from .captions import clip_keep
from .export import render_reel
from .gemini_ranker import DEFAULT_GEMINI_MODEL, KEY_ENV_VAR
from .pipeline import RANKERS, REEL_SILENCE, create_reels, find_highlights
from .settings import Aspect, CaptionStyle, Layout, Rect, ReelSettings


def add_reel_arguments(parser: argparse.ArgumentParser, jobs) -> None:
    """Add the reel jobs to the `jobs` exclusive group and their options to `parser`."""
    r = ReelSettings()
    jobs.add_argument("--reel", action="store_true",
                      help="cut INPUT from --start to --end into a vertical reel "
                           "saved to OUTPUT (no silence removal)")
    jobs.add_argument("--reels", action="store_true",
                      help="find the best moments of INPUT from its transcript and save "
                           "each as a vertical reel in the OUTPUT folder")
    p = parser
    p.add_argument("--start", type=float, default=0.0, metavar="SECONDS",
                   help="with --reel: where the clip starts (default 0)")
    p.add_argument("--end", type=float, default=None, metavar="SECONDS",
                   help=f"with --reel: where the clip ends (default start + "
                        f"{r.max_length_s}s, or the end of the video)")
    p.add_argument("--clips", type=int, default=r.clip_count,
                   help=f"with --reels: how many clips to make (default {r.clip_count})")
    p.add_argument("--min-length", type=int, default=r.min_length_s, metavar="SECONDS",
                   help=f"with --reels: shortest clip (default {r.min_length_s})")
    p.add_argument("--max-length", type=int, default=r.max_length_s, metavar="SECONDS",
                   help=f"with --reels: longest clip (default {r.max_length_s})")
    p.add_argument("--ai", choices=RANKERS, default="local",
                   help="with --reels: who picks the clips; local scores the transcript "
                        f"offline, gemini sends it to Google Gemini (needs {KEY_ENV_VAR}; "
                        "default local)")
    p.add_argument("--gemini-model", default=None, metavar="NAME",
                   help=f"with --ai gemini: model name (default {DEFAULT_GEMINI_MODEL})")
    p.add_argument("--layout", choices=[m.value for m in Layout], default=r.layout.value,
                   help="with --reel/--reels: fit-blur/fit-bars show the whole video "
                        "centred, static-crop fills the canvas; speaker follows your face, "
                        "code follows your typing, split shows typing above the webcam, "
                        f"auto switches between code and speaker (default {r.layout.value})")
    p.add_argument("--aspect", choices=[a.value for a in Aspect], default=r.aspect.value,
                   help=f"with --reel/--reels: canvas shape (default {r.aspect.value})")
    p.add_argument("--webcam", metavar="X,Y,W,H", default=None,
                   help="with --reel/--reels: webcam overlay position in pixels "
                        "(default: found automatically in the bottom-right corner)")
    p.add_argument("--code-zoom", type=float, default=r.code_zoom,
                   help="with code/split/auto layouts: zoom into the typing area, "
                        f"1.0-3.0 (default {r.code_zoom})")
    p.add_argument("--split-ratio", type=float, default=r.split_ratio,
                   help=f"with --layout split: share of the height for the screen "
                        f"(default {r.split_ratio})")
    p.add_argument("--captions", action="store_true",
                   help="with --reel/--reels: burn word-by-word captions into the video "
                        "(needs faster-whisper)")
    p.add_argument("--caption-style", choices=[c.value for c in CaptionStyle],
                   default=r.caption_style.value,
                   help=f"with --captions: look of the captions (default "
                        f"{r.caption_style.value})")
    p.add_argument("--title-overlay", action="store_true",
                   help="with --reel/--reels: show the clip title at the top for the "
                        "first seconds")
    p.add_argument("--title", default=None,
                   help="with --reel --title-overlay: the title text")
    p.add_argument("--remove-silence", action="store_true",
                   help="with --reel/--reels: cut pauses inside each clip")


def reel_settings(args: argparse.Namespace) -> ReelSettings:
    return ReelSettings(
        layout=args.layout, aspect=args.aspect, split_ratio=args.split_ratio,
        code_zoom=args.code_zoom, clip_count=args.clips,
        min_length_s=args.min_length, max_length_s=args.max_length,
        webcam=Rect.parse(args.webcam) if args.webcam else None,
        captions=args.captions, caption_style=args.caption_style,
        title_overlay=args.title_overlay, remove_silence=args.remove_silence,
    )


def run_reel(args: argparse.Namespace, progress: Callable[[float], None]) -> int:
    settings = reel_settings(args)
    end_s = args.end
    if end_s is None:
        end_s = min(probe(args.input).duration_s, args.start + settings.max_length_s)
    words = ()
    if settings.captions:
        print("Transcribing the clip for captions…")
        words = speech_captions(args.input, model_size=args.whisper_model,
                                word_timestamps=True, start_s=args.start, end_s=end_s).words
    keep = None
    if settings.remove_silence:
        keep_all = detect(analyze(args.input, use_vad=False), REEL_SILENCE).keep
        keep = clip_keep(keep_all, args.start, end_s)
    render_reel(args.input, args.output, args.start, end_s, settings, on_progress=progress,
                words=words, title=args.title, keep=keep)
    print(f"\nSaved {args.output}")
    return 0


def run_reels(args: argparse.Namespace, progress: Callable[[float], None]) -> int:
    settings = reel_settings(args)
    if args.dry_run:
        picks = find_highlights(args.input, settings, args.ai, args.whisper_model,
                                cache_dir=args.output, gemini_model=args.gemini_model,
                                on_progress=progress)
        print()
        for i, pick in enumerate(picks, start=1):
            print(f"{i}. {_clock(pick.start_s)}-{_clock(pick.end_s)} "
                  f"({pick.length_s:.0f}s, score {pick.score:.1f}) {pick.title}")
        if not picks:
            print("No clip fits the length limits.")
        return 0
    run = create_reels(args.input, args.output, settings, args.ai, args.whisper_model,
                       gemini_model=args.gemini_model, on_progress=progress)
    print()
    for reel in run.reels:
        print(f"Saved {reel.output}  ({reel.highlight.length_s:.0f}s) {reel.highlight.title}")
    print(f"Summary: {run.summary_path}")
    return 0


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"
