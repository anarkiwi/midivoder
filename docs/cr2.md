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
| Pitch-bend law | linear in semitones | **linear in period** toward note ± range, target clamped to ≤ 96 |
| Per-note timbre | soundfont sample | coil pulse train (bright, harmonically rich) |
| Note range | 0–127 | **≤ 96** (higher note-ons ignored) |
| Amplitude | SF2: `40·log10(v/127)` dB per velocity | pulse width above `breakoutUs` = `hzScale(note) × v/127 × ADSR × CC7/127`, all linear |
| ADSR | per-patch | per-channel CCs: **73 attack, 75 decay, 24 sustain, 72 release** (0–127 ⇒ 0–4 s) |

Two facts drive everything below:

1. **Pitched noise on channel 10.** A note on channel 10 plays band-limited noise whose
   center is the note pitch and whose width is `±bendRange` semitones — a natural fit for
   speech fricatives/plosives, far better than GM drum hits.
2. **Amplitude is real-time pulse width.** `ModulateChain` (CR2 `CRMidi.cpp`) multiplies the
   pulse width by `pulseUsScale` (velocity), the ADSR `envelope->level`, and `midiChannel->volume`
   (CC7) **on every pulse**. CC7 is therefore a true real-time amplitude control. The whole
   chain sits above a hard `breakoutUs` floor (the coil's minimum spark on-time), and is
   further scaled by `hzScale(note) = 1 − hz(note)/hz(96)` of the base note (0 at note 96).

## What `--synth cr2` does today

Implemented in `config.py`, `midi.py`, `percussion.py`, `encoder.py`, `cli.py`:

- **Channel map.** Tonal partials are capped to channels 0–7 (`CR2_MAX_TONAL_INDEX`) plus the
  noise channel 9, matching exactly what CR2 listens on.
- **Note ceiling.** Notes (tonal and noise) clamp to 95, one below `CR2_MAX_PITCH` = 96 where
  `hzScale` reaches 0, and spectral peaks at or above `hz(95.5)` are not tracked, so no channel
  is spent on a partial CR2 cannot sound.
- **Bend range via CC31** instead of RPN. (Without this, midivoder's ±2-semitone pitchwheel
  values would be read against CR2's default ±12 range and barely move the pitch.)
- **Period-linear bends.** `bend_value` inverts `PitchBender::BendHz`: the bend fraction is
  `(2^(−s/12) − 1) / (2^(−d/12) − 1)` for an offset of `s` semitones toward a target `d`
  semitones away (± range, clamped to note 96), so bends near the ceiling stay in tune.
- **ADSR setup.** Tonal channels use CR2's default instant-attack / full-sustain plus a short
  release so note-offs don't click and there is no long tail (the GM ocarina patch doubled
  render length to ~7 s; CR2 renders track the MIDI length, ~4 s). The noise channel gets a
  percussive envelope (instant attack, decay to zero sustain, short release).
- **Fricatives/plosives → pitched noise.** Percussion events carry the spectral `centroid`;
  in cr2 mode the noise note is `nearest_note(centroid)`, so bright sibilants land high and
  plosives low. (GM mode still uses the GM drum note.) Hit velocity is linear in the hit's
  amplitude relative to the loudest frame (GM: `127·sqrt(a)`, matching the SF2 curve).
- **CC7 continuous amplitude track.** Because midivoder gives **one partial per channel for
  its lifetime**, CC7 *is* that partial's amplitude envelope. cr2 mode draws the amplitude
  contour as CC7 at the control rate and triggers notes at full velocity — no amplitude-based
  retriggering, which removes note churn and clicks. For amplitude ratio `a` (vs. the loudest
  partial, floored at −48 dB) on base note `n`, `CC7 = 127·a / hzScale(n)`, clipped to 127,
  so the coil's linear pulse width reproduces `a`. GM carries `a` in note-on velocity
  `127·sqrt(a)` and retriggers on big level jumps.

## Fidelity observed

Reference phrases (espeak) → encode → render → mel cosine distance vs reference
(lower = better; `metrics.compare`):

| phrase | gm cos | cr2 cos |
| --- | --- | --- |
| mixed | ~0.27 | ~0.69 |
| plosive | ~0.31 | ~0.68 |
| fricative | ~0.37 | ~0.71 |

- CR2 scores worse, but this is a **timbre gap, not an encoding bug**: a coil pulse is bright
  and harmonically rich, whereas speech is defined by a few **formant** resonances with
  rolloff. A single pulse oscillator cannot make a formant-shaped (narrowband, peaked)
  spectrum, so vowels read as buzzy.
- The **CC7 amplitude track does not move this metric much**. That is expected: the metric is DTW-aligned and per-frame level-normalized, so it measures
  *spectral-envelope shape* and is largely blind to amplitude-envelope dynamics and click
  removal — exactly what CC7 improves. Judge that change by ear, not by this number.

## Future work

### On CR2 (firmware — notes for the chime_red2 project)

Ordered by fidelity gain per unit of effort.

1. **Formant / FOF oscillator mode (biggest win,
   [chime_red2#45](https://github.com/anarkiwi/chime_red2/issues/45)).** Add a voice mode where each glottal
   period (at the note pitch) triggers a short exponentially-decaying ring at a settable
   **formant frequency + bandwidth**. Summing 2–3 of these makes a vowel — how Klatt/FOF
   formant synths and the vocal tract work, and it maps cleanly onto a coil (a damped pulse
   burst at the formant rate, repeated at the pitch rate). This is the change that would
   actually close the speech-timbre gap and move the metric above.
2. **Expose more tonal channels (cheap,
   [chime_red2#46](https://github.com/anarkiwi/chime_red2/issues/46)).** `maxMidiChannel = 8` but there are 16 oscillators.
   Mapping ~15 tonal channels (keeping 10 for noise) lets midivoder place more partials
   simultaneously = more formant/harmonic detail. Small change to the channel-map array.
3. **Pitch-dependent pulse-width scaling
   ([chime_red2#47](https://github.com/anarkiwi/chime_red2/issues/47)).** `hzScale` drives
   output to zero at note 96 and caps what CC7 can compensate near it (CC7 ≤ 127 only reaches
   `a ≤ hzScale(n)`), so high partials are attenuated or lost. Making the curve configurable
   or optional would let the encoder use the top octave.
4. **Decouple noise center from bandwidth + HF tilt (cheap).** Channel-10 noise ties center
   and width together (width = `±bendRange`). Independent center/bandwidth plus a
   high-frequency tilt would let fricatives be distinguished — /s/ (~5–8 kHz, narrow-bright)
   vs /ʃ/ (~2–4 kHz) vs /f/ (flat).
5. **Companding the amplitude curve, not a new CC (medium).** The `breakoutUs` floor means
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
