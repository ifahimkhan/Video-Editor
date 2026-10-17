"""Command-line interface: `python -m silence_remover in.mp4 out.mp4 [options]`."""

from __future__ import annotations

import argparse
import sys

from .analysis import analyze
from .audio_export import DEFAULT_MP3_KBPS, MP3_BITRATES_KBPS, extract_mp3
from .echo_removal import remove_echo
from .ffmpeg_tools import FFmpegError, probe
from .lossless import read_keyframes, render_lossless, snap_result
from .reels.export import render_reel
from .reels.settings import Aspect, Layout, Rect, ReelSettings
from .render import render
from .segments import DetectionMode, DetectionSettings, detect
from .swap_audio import swap_audio
from .transcript import (
    DEFAULT_MODEL,
    MODEL_SIZES,
    TranscriptUnavailable,
    transcribe,
)


def build_parser() -> argparse.ArgumentParser:
    d = DetectionSettings()
    p = argparse.ArgumentParser(
        prog="silence_remover",
        description="Remove silent parts from a video. Run with no arguments for the GUI.",
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--mode", choices=[m.value for m in DetectionMode],
                   default=d.mode.value,
                   help="loudness: dB threshold (default); voice: Silero VAD, "
                        "ignores background noise")
    p.add_argument("-v", "--vad-threshold", type=float, default=d.vad_threshold,
                   help=f"voice mode: speech probability 0-1 (default {d.vad_threshold})")
    p.add_argument("-t", "--threshold", type=float, default=d.threshold_db,
                   help=f"silence threshold in dBFS (default {d.threshold_db})")
    p.add_argument("-m", "--min-silence", type=int, default=d.min_silence_ms,
                   help=f"minimum silence length to cut, ms (default {d.min_silence_ms})")
    p.add_argument("-p", "--padding", type=int, default=d.padding_ms,
                   help=f"softness kept around speech, ms (default {d.padding_ms})")
    p.add_argument("--lossless", action="store_true",
                   help="fast export without re-encoding; cuts start on keyframes, "
                        "so a little extra silence may remain")
    # Each of these replaces silence removal with a different job.
    audio_jobs = p.add_mutually_exclusive_group()
    audio_jobs.add_argument("--extract-audio", action="store_true",
                            help="save the full audio track of INPUT as MP3 to OUTPUT "
                                 "(no silence removal)")
    audio_jobs.add_argument("--swap-audio", metavar="AUDIO",
                            help="replace INPUT's soundtrack with AUDIO and save to OUTPUT "
                                 "(video copied untouched, no silence removal)")
    audio_jobs.add_argument("--remove-echo", action="store_true",
                            help="remove the echo of a voice recorded by two microphones "
                                 "at once; saves INPUT's audio to OUTPUT "
                                 "(.mp3, .wav, .flac or .m4a, no silence removal)")
    audio_jobs.add_argument("--transcript", action="store_true",
                            help="write what is said in INPUT to OUTPUT (.srt, .vtt or "
                                 ".txt) with faster-whisper (no silence removal)")
    audio_jobs.add_argument("--reel", action="store_true",
                            help="cut INPUT from --start to --end into a vertical reel "
                                 "saved to OUTPUT (no silence removal)")
    r = ReelSettings()
    p.add_argument("--start", type=float, default=0.0, metavar="SECONDS",
                   help="with --reel: where the clip starts (default 0)")
    p.add_argument("--end", type=float, default=None, metavar="SECONDS",
                   help=f"with --reel: where the clip ends (default start + "
                        f"{r.max_length_s}s, or the end of the video)")
    p.add_argument("--layout", choices=[m.value for m in Layout], default=r.layout.value,
                   help="with --reel: fit-blur/fit-bars show the whole video centred, "
                        "static-crop fills the canvas, split puts the screen above "
                        f"the webcam (default {r.layout.value})")
    p.add_argument("--aspect", choices=[a.value for a in Aspect], default=r.aspect.value,
                   help=f"with --reel: canvas shape (default {r.aspect.value})")
    p.add_argument("--webcam", metavar="X,Y,W,H", default=None,
                   help="with --reel --layout split: webcam overlay position in pixels "
                        "(default: bottom-right quarter)")
    p.add_argument("--split-ratio", type=float, default=r.split_ratio,
                   help=f"with --layout split: share of the height for the screen "
                        f"(default {r.split_ratio})")
    p.add_argument("--translate", action="store_true",
                   help="with --transcript: translate the speech to English")
    p.add_argument("--whisper-model", choices=MODEL_SIZES, default=DEFAULT_MODEL,
                   help=f"with --transcript: speech model size (default {DEFAULT_MODEL}; "
                        "larger is more accurate but slower)")
    p.add_argument("--mp3-bitrate", type=int, default=DEFAULT_MP3_KBPS,
                   choices=MP3_BITRATES_KBPS, metavar="KBPS",
                   help=f"MP3 bitrate for --extract-audio (default {DEFAULT_MP3_KBPS})")
    p.add_argument("--dry-run", action="store_true",
                   help="only print what would be cut")
    p.add_argument("--audio-offset", type=int, default=0, metavar="MS",
                   help="with --swap-audio: shift the new audio later (positive) "
                        "or earlier (negative), in ms (default 0)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.swap_audio:
            result = swap_audio(args.input, args.swap_audio, args.output,
                                offset_ms=args.audio_offset, on_progress=_print_progress)
            print(f"\nSaved {args.output}")
            if result.length_warning:
                print(f"Note: {result.length_warning}")
            return 0
        if args.reel:
            return _run_reel(args)
        if args.transcript:
            result = transcribe(args.input, args.output, translate=args.translate,
                                model_size=args.whisper_model,
                                on_progress=_print_progress)
            print(f"\n{result.summary}\nSaved {args.output}")
            return 0
        if args.remove_echo:
            result = remove_echo(args.input, args.output, on_progress=_print_progress)
            print(f"\n{result.summary}\nSaved {args.output}")
            return 0
        if args.extract_audio:
            extract_mp3(args.input, args.output, args.mp3_bitrate,
                        on_progress=_print_progress)
            print(f"\nSaved {args.output}")
            return 0
        settings = DetectionSettings(
            threshold_db=args.threshold, min_silence_ms=args.min_silence,
            padding_ms=args.padding, mode=args.mode, vad_threshold=args.vad_threshold,
        )
        use_vad = settings.mode is DetectionMode.VOICE
        profile = analyze(args.input, use_vad=use_vad)
        if use_vad and not profile.has_vad:
            print("Error: voice mode needs onnxruntime and the Silero model "
                  "(see README).", file=sys.stderr)
            return 1
        result = detect(profile, settings)
        if args.lossless:
            precise_ms = result.kept_ms
            result = snap_result(result, read_keyframes(args.input))
            print(f"Lossless: +{(result.kept_ms - precise_ms) / 1000:.1f}s kept "
                  "to start on keyframes")
        print(f"Keeping {result.kept_ms / 1000:.1f}s of {result.duration_ms / 1000:.1f}s "
              f"({len(result.cuts)} cuts)")
        if args.dry_run:
            return 0
        if not result.keep:
            print("Error: everything is below the threshold.", file=sys.stderr)
            return 1
        export = render_lossless if args.lossless else render
        export(args.input, args.output, result.keep, on_progress=_print_progress)
        print(f"\nSaved {args.output}")
        return 0
    except (FFmpegError, ValueError, OSError, TranscriptUnavailable) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


def _run_reel(args: argparse.Namespace) -> int:
    settings = ReelSettings(
        layout=args.layout, aspect=args.aspect, split_ratio=args.split_ratio,
        webcam=Rect.parse(args.webcam) if args.webcam else None,
    )
    end_s = args.end
    if end_s is None:
        end_s = min(probe(args.input).duration_s, args.start + settings.max_length_s)
    render_reel(args.input, args.output, args.start, end_s, settings,
                on_progress=_print_progress)
    print(f"\nSaved {args.output}")
    return 0


def _print_progress(fraction: float) -> None:
    print(f"\rExporting… {fraction * 100:5.1f}%", end="", flush=True)
