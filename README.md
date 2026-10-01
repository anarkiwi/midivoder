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

## Intelligibility tools

Install with `pip install .[eval]` or build the Docker `eval` target
(`docker build --target eval -t midivoder-eval .`).

```
python3 tools/intelligibility.py render voice.mid voice_sine.wav --synth gm|cr2  # ideal-sine render under the synth's pitch and level laws
python3 tools/intelligibility.py wer words.txt a.wav [b.wav ...]                 # faster-whisper word error rate vs reference text
python3 tools/intelligibility.py stoi voice.wav a.wav [b.wav ...]                # STOI vs the source voice WAV
```

## Example

Speak a phrase with `espeak-ng`, encode it, and render the MIDI back to audio:

```
espeak-ng -w hello.wav "hello, I am a vocoder"
midivoder encode hello.wav -o hello.mid
midivoder render hello.mid -o hello_rendered.wav --sample-rate 22050
```

| Input voice | MIDI | Rendered (FluidR3 GM) |
| --- | --- | --- |
| [hello.wav](docs/examples/hello.wav) | [hello.mid](docs/examples/hello.mid) | [hello_rendered.wav](docs/examples/hello_rendered.wav) |

## Tests

Tests (including the end-to-end espeak → MIDI → fluidsynth fidelity check) run in Docker:

```
docker build --target test -t midivoder-test .
docker run --rm midivoder-test
```

## Docs

- [docs/cr2.md](docs/cr2.md) — CHIME RED II target
