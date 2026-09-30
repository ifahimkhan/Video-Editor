import shutil
import subprocess

import numpy as np
import pytest

from silence_remover.analysis import FRAME_MS, LoudnessProfile

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def make_profile(levels_db: list[tuple[int, float]]) -> LoudnessProfile:
    """Build a profile from (duration_ms, level_db) runs."""
    parts = [np.full(ms // FRAME_MS, db, dtype=np.float32) for ms, db in levels_db]
    db = np.concatenate(parts)
    return LoudnessProfile(db=db, frame_ms=FRAME_MS, duration_ms=len(db) * FRAME_MS)


def make_clip(path, duration_s: int, silent_ranges: list[tuple[float, float]],
              video: bool = True, video_args: list[str] | None = None,
              fps: int = 25) -> None:
    """Test video: 440 Hz tone, muted inside `silent_ranges`."""
    mute = "+".join(f"between(t,{a},{b})" for a, b in silent_ranges) or "0"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if video:
        cmd += ["-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={fps}:duration={duration_s}"]
    cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}:sample_rate=48000"]
    audio_in = 1 if video else 0
    cmd += ["-filter_complex", f"[{audio_in}:a]volume='if({mute},0,1)':eval=frame[a]"]
    if video:
        cmd += ["-map", "0:v", "-c:v", "libx264", "-preset", "ultrafast", *(video_args or [])]
    cmd += ["-map", "[a]", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True)


@pytest.fixture(scope="session")
def speech_clip(tmp_path_factory):
    """10 s clip: tone, silent 2-4 s and 6-6.3 s (short pause), tone to end."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    make_clip(path, 10, [(2, 4), (6, 6.3)])
    return path
