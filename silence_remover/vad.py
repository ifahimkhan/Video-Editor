"""Silero voice activity detection (ONNX), fed incrementally while decoding.

Uses the Silero VAD v6 ONNX export distributed with faster-whisper
(MIT licensed): 16 kHz audio, 512-sample windows plus 64 samples of
context from the previous window, LSTM state (h, c) carried across calls.

onnxruntime is optional. When it or the model file is missing,
`load_vad()` returns None and the app falls back to loudness detection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512                      # 32 ms per probability
CONTEXT_SAMPLES = 64
VAD_FRAME_MS = WINDOW_SAMPLES * 1000 // VAD_SAMPLE_RATE
STATE_SHAPE = (1, 1, 128)
MODEL_ENV_VAR = "SILENCE_REMOVER_VAD_MODEL"
DEFAULT_MODEL = Path(__file__).parent / "models" / "silero_vad_v6.onnx"
EXPECTED_INPUTS = {"input", "h", "c"}


class VadUnavailable(RuntimeError):
    pass


def model_path() -> Path:
    override = os.environ.get(MODEL_ENV_VAR)
    return Path(override) if override else DEFAULT_MODEL


class SileroVad:
    """Streaming speech-probability estimator. One instance per file."""

    def __init__(self, path: Path | None = None) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise VadUnavailable("onnxruntime is not installed") from exc

        path = path or model_path()
        if not path.is_file():
            raise VadUnavailable(f"VAD model not found: {path}")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        try:
            self._session = onnxruntime.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises its own exception types
            raise VadUnavailable(f"Could not load VAD model: {exc}") from exc

        names = {i.name for i in self._session.get_inputs()}
        if names != EXPECTED_INPUTS:
            raise VadUnavailable(
                f"Unsupported VAD model (inputs {sorted(names)}); "
                "expected the Silero v6 export from faster-whisper."
            )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Speech probability per 512-sample window.

        `samples` is float32 in [-1, 1]; its length must be a multiple of 512.
        """
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        if samples.size % WINDOW_SAMPLES:
            raise ValueError(f"sample count must be a multiple of {WINDOW_SAMPLES}")

        full = np.concatenate([self._context, samples.astype(np.float32)])
        n_windows = samples.size // WINDOW_SAMPLES
        starts = np.arange(n_windows) * WINDOW_SAMPLES
        idx = starts[:, None] + np.arange(WINDOW_SAMPLES + CONTEXT_SAMPLES)
        batch = full[idx]

        probs, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = full[-CONTEXT_SAMPLES:].copy()
        return np.asarray(probs, dtype=np.float32).reshape(-1)


def load_vad() -> SileroVad | None:
    try:
        return SileroVad()
    except VadUnavailable as exc:
        log.info("Voice detection disabled: %s", exc)
        return None


def vad_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().is_file()
