# CHIME RED II (`--synth cr2`) target — findings & future work

midivoder can target two synths via `EncodeConfig.synth` / `--synth`:

- `gm` (default) — General MIDI, rendered with FluidSynth.
- `cr2` — [CHIME RED II](../../chime_red2), a Tesla-coil controller. Sound is a coil
  **gate pulse train**: each oscillator pulses the coil at the note frequency, and loudness
  is the gate **pulse width**. The sibling repo ships an off-target software simulation
  (`cr-render`) that runs the real synth code on a PC and renders a `.mid` to a WAV.

This document records how CR2 differs from GM, what `cr2` mode does today, the fidelity we
observed, and proposals for future work. It is about midivoder's CR2 *target*; firmware
proposals here are notes for the CR2 project, not changes made in that repo.

## Rendering through CR2

The sibling repo's `test/host/render.sh INPUT.mid OUTPUT.wav` builds a Docker image and
renders in one step. As of this writing that build **fails on a pre-existing cppcheck
false-positive** (`Unmatched suppression` in `Oscillator.h`), unrelated to our MIDI. The
already-built image can be driven directly:

```
docker run --rm -v "$PWD:/in:ro" -v "$PWD:/out" chime-red-host-tests \
  cr-render /in/IN.mid /out/OUT.wav --rate 44100 --tail 1
```

Useful `cr-render` flags: `--tail SECONDS`, `--gain G`, `--rate HZ`, `--seed N` (RNG for the
channel-10 pitched noise).

## How CR2 differs from General MIDI

| Aspect | General MIDI | CHIME RED II |
| --- | --- | --- |
| Tonal channels | 1–16 | **1–8 only** (others ignored) |
| Percussion channel | 10 (drum map) | 10 (**pitched noise**, not drums) |
| Polyphony | per soundfont | 16 oscillators total, shared; detune (CC94/95) halves it |
| Pitch-bend range | RPN 0,0 | **CC31** (RPN is ignored) |
| Per-note timbre | soundfont sample | coil pulse train (bright, harmonically rich) |
| Amplitude | velocity + CC7/CC11 | gate **pulse width** = `velocity × ADSR_level × CC7`, floored at `breakoutUs` |
| ADSR | per-patch | per-channel CCs: **73 attack, 75 decay, 24 sustain, 72 release** (0–127 ⇒ 0–4 s) |

Two facts drive everything below:

1. **Pitched noise on channel 10.** A note on channel 10 plays band-limited noise whose
   center is the note pitch and whose width is `±bendRange` semitones — a natural fit for
   speech fricatives/plosives, far better than GM drum hits.
2. **Amplitude is real-time pulse width.** `ModulateChain` (CR2 `CRMidi.cpp`) multiplies the
   pulse width by `pulseUsScale` (velocity), the ADSR `envelope->level`, and `midiChannel->volume`
   (CC7) **on every pulse**. CC7 is therefore a true real-time amplitude control. The whole
   chain sits above a hard `breakoutUs` floor (the coil's minimum spark on-time).

## What `--synth cr2` does today

Implemented in `config.py`, `midi.py`, `percussion.py`, `encoder.py`, `cli.py`:

- **Channel map.** Tonal partials are capped to channels 0–7 (`CR2_MAX_TONAL_INDEX`) plus the
  noise channel 9, matching exactly what CR2 listens on.
- **Bend range via CC31** instead of RPN. (Without this, midivoder's ±2-semitone pitchwheel
  values would be read against CR2's default ±12 range and barely move the pitch.)
- **ADSR setup.** Tonal channels use CR2's default instant-attack / full-sustain plus a short
  release so note-offs don't click and there is no long tail (the GM ocarina patch doubled
  render length to ~7 s; CR2 renders track the MIDI length, ~4 s). The noise channel gets a
  percussive envelope (instant attack, decay to zero sustain, short release).
- **Fricatives/plosives → pitched noise.** Percussion events carry the spectral `centroid`;
  in cr2 mode the noise note is `nearest_note(centroid)`, so bright sibilants land high and
  plosives low. (GM mode still uses the GM drum note.)
- **CC7 continuous amplitude track.** Because midivoder gives **one partial per channel for
  its lifetime**, CC7 *is* that partial's amplitude envelope. cr2 mode draws the amplitude
  contour as CC7 at the control rate and triggers notes at full velocity — no amplitude-based
  retriggering. This removed note churn and clicks and cut the message rate to ~600–720/s
  (from ~1000+). GM mode is unchanged (amplitude in note-on velocity, retrigger on big jumps).

## Fidelity observed

Reference phrases (espeak) → encode → render → mel cosine distance vs reference
(lower = better; `metrics.compare`):

| phrase | gm cos | cr2 cos |
| --- | --- | --- |
| mixed | ~0.34 | ~0.68 |
| plosive | ~0.36 | ~0.68 |
| fricative | ~0.43 | ~0.68 |

- CR2 scores worse, but this is a **timbre gap, not an encoding bug**: a coil pulse is bright
  and harmonically rich, whereas speech is defined by a few **formant** resonances with
  rolloff. A single pulse oscillator cannot make a formant-shaped (narrowband, peaked)
  spectrum, so vowels read as buzzy.
- The **CC7 amplitude track did not move this metric** (~0.68 before and after). That is
  expected: the metric is DTW-aligned and per-frame level-normalized, so it measures
  *spectral-envelope shape* and is largely blind to amplitude-envelope dynamics and click
  removal — exactly what CC7 improves. Judge that change by ear, not by this number.

## Future work

### On CR2 (firmware — notes for the chime_red2 project)

Ordered by fidelity gain per unit of effort.

1. **Formant / FOF oscillator mode (biggest win).** Add a voice mode where each glottal
   period (at the note pitch) triggers a short exponentially-decaying ring at a settable
   **formant frequency + bandwidth**. Summing 2–3 of these makes a vowel — how Klatt/FOF
   formant synths and the vocal tract work, and it maps cleanly onto a coil (a damped pulse
   burst at the formant rate, repeated at the pitch rate). This is the change that would
   actually close the speech-timbre gap and move the metric above.
2. **Expose more tonal channels (cheap).** `maxMidiChannel = 8` but there are 16 oscillators.
   Mapping ~15 tonal channels (keeping 10 for noise) lets midivoder place more partials
   simultaneously = more formant/harmonic detail. Small change to the channel-map array.
3. **Decouple noise center from bandwidth + HF tilt (cheap).** Channel-10 noise ties center
   and width together (width = `±bendRange`). Independent center/bandwidth plus a
   high-frequency tilt would let fricatives be distinguished — /s/ (~5–8 kHz, narrow-bright)
   vs /ʃ/ (~2–4 kHz) vs /f/ (flat).
4. **Companding the amplitude curve, not a new CC (medium).** The `breakoutUs` floor means
   soft passages clip to silence or pop regardless of how finely amplitude is drawn. Mapping
   velocity/volume → pulse width through a perceptual/log curve that packs speech's dynamic
   range into the usable band above breakout would improve perceived dynamics more than any
   controller change.
   - **CC7 vs a dedicated amplitude CC:** for midivoder's one-partial-per-channel model they
     are equivalent — a CC11-style expression would multiply in at the same point with the
     same 7-bit resolution, so it buys nothing here. The genuinely more capable option is
     **per-note** amplitude (poly-aftertouch / MPE-style per-voice gain), which lets notes in
     a chord on one channel have independent envelopes. midivoder does not need it because it
     already isolates partials onto channels.

### On the midivoder side

- **Formant-aware partial selection.** Group/select partials around formant peaks (the
  analysis already finds spectral peaks) to feed a future CR2 FOF mode, or to spend limited
  channels on the most perceptually important partials.
- **Per-fricative noise bandwidth.** Vary CC31 on the noise channel per hit to widen/narrow
  the noise band by consonant type (once CR2 decouples center/bandwidth, switch to that).
- **CR2-specific fidelity gate.** The current gate thresholds (`COS_MAX`) are GM-calibrated.
  A CR2 gate needs its own thresholds, and ideally a metric less dominated by absolute timbre
  (e.g. emphasising formant-track / pitch-track agreement over raw mel-envelope cosine).
- **Optional detune (CC94/95)** for a slightly less sterile tonal voice.
- **CC7 amplitude in GM mode.** GM also has a real-time channel volume; the same
  one-partial-per-channel trick could replace velocity retriggering there too.
