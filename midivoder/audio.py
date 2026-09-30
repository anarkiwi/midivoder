"""Audio I/O and preprocessing."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load `path`, downmix to mono, resample to `target_sr`, peak-normalize."""
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    mono = data.mean(axis=1)
    mono = resample(mono, sr, target_sr)
    return normalize(mono)


def resample(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase-resample `signal` from `sr` to `target_sr`."""
    if sr == target_sr:
        return signal.astype(np.float64, copy=False)
    g = np.gcd(int(sr), int(target_sr))
    return resample_poly(signal, target_sr // g, sr // g).astype(np.float64)


def normalize(signal: np.ndarray, peak: float = 0.97) -> np.ndarray:
    """Scale `signal` so its absolute peak equals `peak` (silence is returned as-is)."""
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    if m <= 0.0:
        return signal.astype(np.float64, copy=False)
    return (signal * (peak / m)).astype(np.float64)


def write_wav(path: str, signal: np.ndarray, sr: int) -> None:
    """Write `signal` peak-normalized as 16-bit PCM."""
    sf.write(path, normalize(signal), sr, subtype="PCM_16")
