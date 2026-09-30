"""Command-line interface for midivoder."""

from __future__ import annotations

import argparse
import sys

from midivoder import instruments, render
from midivoder.config import EncodeConfig
from midivoder.encoder import encode_wav_to_midi


def _add_encode_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="input voice WAV")
    p.add_argument("-o", "--output", required=True, help="output .mid path")
    p.add_argument(
        "--max-channels",
        type=int,
        default=EncodeConfig.max_channels,
        help="total MIDI channels incl. percussion (caps simultaneous partials)",
    )
    p.add_argument("--bpm", type=float, default=EncodeConfig.bpm)
    p.add_argument(
        "--ppq", type=int, default=EncodeConfig.ppq, help="ticks per quarter note"
    )
    p.add_argument(
        "--hop-ms",
        type=float,
        default=EncodeConfig.hop_ms,
        help="analysis hop (time resolution)",
    )
    p.add_argument("--frame-ms", type=float, default=EncodeConfig.frame_ms)
    p.add_argument(
        "--control-rate",
        type=float,
        default=EncodeConfig.control_rate_hz,
        help="CC11/pitch-bend events per second",
    )
    p.add_argument("--min-note-ms", type=float, default=EncodeConfig.min_note_ms)
    p.add_argument(
        "--bend-range",
        type=float,
        default=EncodeConfig.bend_range,
        help="pitch-bend range in semitones",
    )
    p.add_argument("--sample-rate", type=int, default=EncodeConfig.sample_rate)
    p.add_argument(
        "--instrument",
        default=EncodeConfig.instrument,
        help='"auto" or a GM program number',
    )
    p.add_argument(
        "--no-percussion", action="store_true", help="disable plosive percussion"
    )
    p.add_argument(
        "--synth",
        choices=("gm", "cr2"),
        default=EncodeConfig.synth,
        help="target synth: 'gm' (General MIDI) or 'cr2' (CHIME RED II)",
    )
    p.add_argument(
        "--soundfont", default=None, help="soundfont for auto instrument profiling"
    )


def _config_from_args(args: argparse.Namespace) -> EncodeConfig:
    return EncodeConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        max_channels=args.max_channels,
        bpm=args.bpm,
        ppq=args.ppq,
        control_rate_hz=args.control_rate,
        min_note_ms=args.min_note_ms,
        bend_range=args.bend_range,
        instrument=str(args.instrument),
        enable_percussion=not args.no_percussion,
        synth=args.synth,
    )


def _cmd_encode(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    stats = encode_wav_to_midi(args.input, args.output, cfg, soundfont=args.soundfont)
    print(f"encoded {args.input} -> {args.output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    if not render.fluidsynth_available():
        print("error: fluidsynth not installed", file=sys.stderr)
        return 2
    render.render_midi(
        args.input, args.output, soundfont=args.soundfont, sample_rate=args.sample_rate
    )
    print(f"rendered {args.input} -> {args.output}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    sf = args.soundfont or render.find_default_soundfont()
    if not render.fluidsynth_available() or sf is None:
        print(
            f"fluidsynth/soundfont unavailable; default program: "
            f"{instruments.DEFAULT_TONAL_PROGRAM} "
            f"({instruments.program_name(instruments.DEFAULT_TONAL_PROGRAM)})"
        )
        return 0
    prog = instruments.select_tonal_program(sf)
    print(f"selected tonal program: {prog} ({instruments.program_name(prog)})")
    print(f"soundfont: {sf}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` and dispatch to the selected subcommand; return the exit code."""
    parser = argparse.ArgumentParser(prog="midivoder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode a voice WAV into a .mid file")
    _add_encode_flags(enc)
    enc.set_defaults(func=_cmd_encode)

    rnd = sub.add_parser("render", help="render a .mid back to WAV via fluidsynth")
    rnd.add_argument("input", help="input .mid")
    rnd.add_argument("-o", "--output", required=True, help="output WAV path")
    rnd.add_argument("--sample-rate", type=int, default=44100)
    rnd.add_argument("--soundfont", default=None)
    rnd.set_defaults(func=_cmd_render)

    prof = sub.add_parser(
        "profile-instruments", help="auto-select the best tonal GM program"
    )
    prof.add_argument("--soundfont", default=None)
    prof.set_defaults(func=_cmd_profile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
