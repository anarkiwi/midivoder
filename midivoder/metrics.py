"""Fidelity metrics comparing reference speech against MIDI-resynthesized audio.

The two signals are DTW-aligned first (resynthesis introduces small timing drift), then
compared on spectral shape rather than sample-exact alignment. Because the MIDI is rendered
with a different timbre than natural speech, the primary gate is a bounded, level-invariant
**mel cosine distance** (how well the spectral envelope tracks the original); MCD and log-mel
L2 are reported as additional diagnostics.
"""

from __future__ import annotations

import math

import librosa
import numpy as np

from midivoder.audio import normalize, resample

_ANALYSIS_SR = 22050
_N_FFT = 1024
_HOP = 256
_N_MELS = 40
_N_MFCC = 14  # coefficient 0 (energy) is dropped, leaving 13 for MCD


def _prep(sig: np.ndarray, sr: int) -> np.ndarray:
    sig = normalize(resample(np.asarray(sig, dtype=np.float64), sr, _ANALYSIS_SR))
    trimmed, _ = librosa.effects.trim(
        sig, top_db=40
    )  # drop silence + decaying release tail
    return trimmed if trimmed.size else sig


def _dtw_path(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return wp[::-1]  # forward chronological order


def _mel_power(sig: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=sig, sr=_ANALYSIS_SR, n_fft=_N_FFT, hop_length=_HOP, n_mels=_N_MELS
    )


def _mfcc(sig: np.ndarray) -> np.ndarray:
    m = librosa.feature.mfcc(
        y=sig, sr=_ANALYSIS_SR, n_mfcc=_N_MFCC, n_fft=_N_FFT, hop_length=_HOP
    )
    return m[1:]  # drop c0 (overall energy)


def compare(
    ref: np.ndarray, ref_sr: int, deg: np.ndarray, deg_sr: int
) -> dict[str, float]:
    """DTW-align `ref` and `deg` and return mel cosine distance, mel L2 and MCD."""
    a, b = _prep(ref, ref_sr), _prep(deg, deg_sr)
    mel_a, mel_b = _mel_power(a), _mel_power(b)
    log_a, log_b = librosa.power_to_db(mel_a, ref=np.max), librosa.power_to_db(
        mel_b, ref=np.max
    )

    wp = _dtw_path(log_a, log_b)
    ia, ib = wp[:, 0], wp[:, 1]

    # Bounded, level-invariant spectral-shape distance (primary gate): 1 - mean cosine sim.
    va, vb = mel_a[:, ia], mel_b[:, ib]
    cos = np.sum(va * vb, axis=0) / (
        np.linalg.norm(va, axis=0) * np.linalg.norm(vb, axis=0) + 1e-12
    )
    mel_cos_dist = float(1.0 - cos.mean())

    # Log-mel L2 (dB) along the same alignment.
    mel_l2 = float(np.sqrt(np.mean((log_a[:, ia] - log_b[:, ib]) ** 2)))

    # Mel-cepstral distortion (diagnostic; large in absolute terms across timbres).
    mfa, mfb = _mfcc(a), _mfcc(b)
    wpm = _dtw_path(mfa, mfb)
    diffs = mfa[:, wpm[:, 0]] - mfb[:, wpm[:, 1]]
    mcd = float(
        (10.0 / math.log(10))
        * math.sqrt(2.0)
        * np.sqrt(np.sum(diffs**2, axis=0)).mean()
    )

    return {"mel_cos_dist": mel_cos_dist, "mel_l2": mel_l2, "mcd": mcd}
