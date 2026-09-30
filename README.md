# midivoder

Vocode a voice WAV into a multi-channel MIDI file: tonal partials become pitch-bent
notes across channels, plosives become percussion. Targets General MIDI (`gm`, default)
or the CHIME RED II Tesla-coil controller (`cr2`, see [docs/cr2.md](docs/cr2.md)).

## Install

```
pip install .
```

Rendering and instrument profiling need `fluidsynth` and a GM soundfont
(set `MIDIVODER_SOUNDFONT` or pass `--soundfont`).

## Usage

```
midivoder encode voice.wav -o voice.mid [--synth gm|cr2] [--max-channels N] [--instrument ...]
midivoder render voice.mid -o voice_rendered.wav
midivoder profile-instruments
```

`midivoder encode --help` lists all analysis and MIDI options.

## Tests

Tests (including the end-to-end espeak → MIDI → fluidsynth fidelity check) run in Docker:

```
docker build --target test -t midivoder-test .
docker run --rm midivoder-test
```

## Docs

- [docs/cr2.md](docs/cr2.md) — CHIME RED II target
