"""Intelligibility evaluation: ideal-sine resynthesis of midivoder MIDI, Whisper WER, STOI.

`render` plays every tonal note as a pure sinusoid at the pitch and level the target synth
would produce (each synth's bend and amplitude law from midivoder.midi); `wer` transcribes
WAVs with faster-whisper against a reference text; `stoi` scores WAVs against the source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mido
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from midivoder.config import CR2_MAX_BEND_RANGE, PERCUSSION_CHANNEL, EncodeConfig
from midivoder.midi import (
    CR2_CC_BEND_RANGE,
    CR2_CC_RELEASE,
    CR2_CC_VOLUME,
    cr2_expiry_s,
    rendered_level,
    synth_bend_semitones,
)

_GM_DEFAULT_BEND_RANGE = 2.0


@dataclass
class _Channel:
    bend_range: float
    bend: int = 0
    volume: int = 127
    release: int = 0
    rpn: tuple[int, int] = (127, 127)
    notes: dict = field(default_factory=dict)  # note -> active _Tone
    released: dict = field(default_factory=dict)  # note -> (_Tone, off time)


@dataclass
class _Tone:
    channel: int
    note: int
    velocity: int
    points: list = field(default_factory=list)  # (time, midi pitch, amplitude)


def midi_tones(path: str, synth: str) -> tuple[list[_Tone], float]:
    """Sounding tonal notes of `path` with (time, pitch, level) breakpoints, and duration."""
    ceiling = EncodeConfig(synth=synth).note_ceiling
    cr2 = synth == "cr2"
    chans: dict[int, _Channel] = {}
    tones: list[_Tone] = []
    t = 0.0

    def point(ch: _Channel, tone: _Tone, level: bool = True) -> None:
        pitch = tone.note + synth_bend_semitones(
            ch.bend, ch.bend_range, synth, tone.note
        )
        amp = rendered_level(tone.velocity, ch.volume, tone.note, synth)
        tone.points.append((t, float(pitch), float(amp) if level else 0.0))

    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.is_meta or msg.channel == PERCUSSION_CHANNEL:
            continue
        ch = chans.setdefault(
            msg.channel,
            _Channel(float(CR2_MAX_BEND_RANGE if cr2 else _GM_DEFAULT_BEND_RANGE)),
        )
        if msg.type == "pitchwheel":
            ch.bend = msg.pitch
        elif msg.type == "control_change":
            if msg.control == CR2_CC_VOLUME:
                ch.volume = msg.value
            elif cr2 and msg.control == CR2_CC_BEND_RANGE:
                ch.bend_range = float(min(msg.value, CR2_MAX_BEND_RANGE))
            elif cr2 and msg.control == CR2_CC_RELEASE:
                ch.release = msg.value
            elif msg.control in (101, 100):
                ch.rpn = (
                    (msg.value, ch.rpn[1])
                    if msg.control == 101
                    else (ch.rpn[0], msg.value)
                )
            elif not cr2 and msg.control == 6 and ch.rpn == (0, 0):
                ch.bend_range = float(msg.value)
        elif msg.type == "note_on" and msg.velocity > 0:
            if msg.note > ceiling:
                continue
            old = ch.notes.get(msg.note)
            if old is not None and not cr2:
                point(ch, old, level=False)
                old = None
            if old is None and cr2 and msg.note in ch.released:
                old, off = ch.released.pop(msg.note)
                old = old if t < off + cr2_expiry_s(ch.release) else None
            if old is None:
                old = _Tone(msg.channel, msg.note, msg.velocity)
                tones.append(old)
            ch.notes[msg.note] = old
        elif msg.type in ("note_off", "note_on"):
            tone = ch.notes.pop(msg.note, None)
            if tone is not None:
                point(ch, tone, level=False)
                ch.released[msg.note] = (tone, t)
            continue
        else:
            continue
        for tone in ch.notes.values():
            point(ch, tone)
    for ch in chans.values():
        for tone in ch.notes.values():
            point(ch, tone, level=False)
    return tones, t


def render(path: str, synth: str, sr: int = 22050) -> np.ndarray:
    """Ideal-sine rendering of the tonal channels of midivoder MIDI `path`."""
    tones, duration = midi_tones(path, synth)
    out = np.zeros(int(np.ceil(duration * sr)) + 1)
    for tone in tqdm(tones, desc="sine render", unit="note", leave=False):
        times, pitch, amp = (np.array(v) for v in zip(*tone.points))
        start = np.rint(times * sr).astype(int)
        seg = np.diff(start)
        freq = np.repeat(440.0 * 2.0 ** ((pitch[:-1] - 69.0) / 12.0), seg)
        level = np.repeat(amp[:-1], seg)
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        out[start[0] : start[0] + freq.size] += level * np.sin(phase)
    return out


def word_error_rates(
    wavs: list[str], reference: str, model: str = "small.en", model_dir=None
) -> list[tuple[str, float, str]]:
    """(path, WER, transcript) for each WAV, transcribed by faster-whisper on CPU."""
    # pylint: disable=import-outside-toplevel,import-error
    import jiwer
    from faster_whisper import WhisperModel

    norm = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    asr = WhisperModel(
        model, device="cpu", compute_type="int8", download_root=model_dir
    )
    results = []
    for path in tqdm(wavs, desc="whisper", unit="file"):
        x, sr = sf.read(path, always_2d=True)
        audio = resample_poly(x.mean(axis=1), 16000, sr).astype(np.float32)
        segs, _ = asr.transcribe(
            audio, language="en", beam_size=5, condition_on_previous_text=False
        )
        hyp = " ".join(s.text for s in segs).strip()
        results.append((path, float(jiwer.wer(norm(reference), norm(hyp))), hyp))
    return results


def stoi_scores(wavs: list[str], reference_wav: str) -> list[tuple[str, float]]:
    """(path, STOI) of each WAV against `reference_wav`, both resampled to STOI's rate."""
    # pylint: disable-next=import-outside-toplevel,import-error
    from pystoi import stoi

    rate = 10000

    def load(path: str) -> np.ndarray:
        x, sr = sf.read(path, always_2d=True)
        return resample_poly(x.mean(axis=1), rate, sr)

    ref = load(reference_wav)
    results = []
    for path in tqdm(wavs, desc="stoi", unit="file"):
        x = load(path)
        n = min(x.size, ref.size)
        results.append((path, float(stoi(ref[:n], x[:n], rate))))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rnd = sub.add_parser("render", help="ideal-sine render a midivoder MIDI file")
    rnd.add_argument("input")
    rnd.add_argument("output")
    rnd.add_argument("--synth", choices=("gm", "cr2"), default="gm")
    rnd.add_argument("--rate", type=int, default=22050)
    wer = sub.add_parser("wer", help="Whisper word error rate of WAVs vs a text")
    wer.add_argument("reference", help="text file with the spoken words")
    wer.add_argument("wavs", nargs="+")
    wer.add_argument("--model", default="small.en")
    wer.add_argument("--model-dir", default=None)
    sti = sub.add_parser("stoi", help="STOI of WAVs vs the source voice WAV")
    sti.add_argument("reference", help="source voice WAV")
    sti.add_argument("wavs", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "render":
        y = render(args.input, args.synth, args.rate)
        peak = float(np.abs(y).max()) or 1.0
        sf.write(args.output, 0.9 * y / peak, args.rate, subtype="PCM_16")
        return 0
    if args.command == "stoi":
        for path, score in stoi_scores(args.wavs, args.reference):
            print(f"{path}\tSTOI {score:.3f}")
        return 0
    with open(args.reference, encoding="utf-8") as f:
        reference = f.read()
    for path, rate, hyp in word_error_rates(
        args.wavs, reference, args.model, args.model_dir
    ):
        print(f"{path}\tWER {rate:.3f}\t{hyp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
