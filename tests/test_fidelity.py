"""End-to-end fidelity gate: espeak -> WAV -> .mid -> render -> compare.

Requires fluidsynth, espeak-ng, a GM soundfont, librosa and matplotlib. Skips cleanly when any are
missing (so unit tests still run locally); the Docker image provides all of them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from midivoder import render
from midivoder.config import EncodeConfig
from midivoder.encoder import encode_wav_to_midi

librosa = pytest.importorskip("librosa")
librosa_display = pytest.importorskip("librosa.display")
metrics = pytest.importorskip("midivoder.metrics")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

# Calibrated on the first green Docker run; act as regression ceilings, not perfection targets.
# mel_cos_dist (primary) is a bounded [0,2] spectral-envelope distance; mel_l2 is a dB diagnostic.
COS_MAX = 0.50
MEL_MAX = 30.0

PHRASES = {
    "plosive": "Peter picked a pack of pickled peppers and tapped the kettle.",
    "fricative": "She sells sea shells; the fresh fish vanished.",
    "mixed": "The quick brown fox jumps over the lazy dog.",
}

_have_tools = (
    render.fluidsynth_available()
    and render.find_default_soundfont() is not None
    and (shutil.which("espeak-ng") or shutil.which("espeak"))
)
pytestmark = pytest.mark.skipif(
    not _have_tools, reason="fluidsynth/espeak/soundfont required"
)


def _artifacts_dir() -> str:
    d = os.environ.get("MIDIVODER_ARTIFACTS", os.path.join(os.getcwd(), "artifacts"))
    os.makedirs(d, exist_ok=True)
    return d


def _espeak(text: str, wav_path: str) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    subprocess.run(
        [exe, "-w", wav_path, "-s", "150", text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _save_spectrogram(path: str, sig: np.ndarray, sr: int, title: str) -> None:
    s = librosa.power_to_db(
        librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=64), ref=np.max
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa_display.specshow(s, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


@pytest.mark.fidelity
@pytest.mark.parametrize("name,text", list(PHRASES.items()))
def test_fidelity(tmp_path, name, text):
    """Rendered MIDI stays within spectral-distance ceilings of the espeak reference."""
    art = _artifacts_dir()
    ref_wav = str(tmp_path / "ref.wav")
    mid_path = str(tmp_path / "out.mid")

    _espeak(text, ref_wav)
    cfg = EncodeConfig()
    stats = encode_wav_to_midi(ref_wav, mid_path, cfg)

    deg, deg_sr = render.render_to_array(mid_path, sample_rate=44100)
    ref, ref_sr = sf.read(ref_wav, always_2d=True, dtype="float64")
    ref = ref.mean(axis=1)

    scores = metrics.compare(ref, ref_sr, deg, deg_sr)

    # Save artifacts for inspection regardless of pass/fail.
    sf.write(
        os.path.join(art, f"{name}_ref.wav"), ref / (np.max(np.abs(ref)) + 1e-9), ref_sr
    )
    sf.write(
        os.path.join(art, f"{name}_rendered.wav"),
        deg / (np.max(np.abs(deg)) + 1e-9),
        deg_sr,
    )
    _save_spectrogram(
        os.path.join(art, f"{name}_ref.png"), ref, ref_sr, f"{name} reference"
    )
    _save_spectrogram(
        os.path.join(art, f"{name}_rendered.png"), deg, deg_sr, f"{name} rendered"
    )
    with open(os.path.join(art, f"{name}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"phrase": text, "stats": stats, **scores}, fh, indent=2)

    assert (
        scores["mel_cos_dist"] < COS_MAX
    ), f"{name}: mel cos dist {scores['mel_cos_dist']:.3f} >= {COS_MAX} (mcd={scores['mcd']:.1f})"
    assert (
        scores["mel_l2"] < MEL_MAX
    ), f"{name}: mel L2 {scores['mel_l2']:.2f} >= {MEL_MAX}"
