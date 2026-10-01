"""Translate partials + percussion events into a Standard MIDI File."""

from __future__ import annotations

import numpy as np
import mido

from midivoder.config import CR2_MAX_PITCH, PERCUSSION_CHANNEL, EncodeConfig
from midivoder.partials import Partial
from midivoder.percussion import PercEvent

_DYNAMIC_RANGE_DB = 48.0  # amplitude window below ref_amp that stays audible
_AMP_FLOOR = 10.0 ** (-_DYNAMIC_RANGE_DB / 20.0)

# Event ordering at a shared tick: frees and setup must precede note-ons.
_PRIO_OFF = 0
_PRIO_SETUP = 1
_PRIO_CTRL = 2
_PRIO_ON = 3

# CHIME RED II control changes (see ../chime_red2/MIDI.md). CR2 ignores RPN, so the
# pitch-bend range is set with CC31 instead; ADSR is CC73/75/24/72 (each 0-127 -> 0-4s).
_CR2_CC_RESET = 121
_CR2_CC_VOLUME = 7
_CR2_CC_BEND_RANGE = 31
_CR2_CC_ATTACK = 73
_CR2_CC_DECAY = 75
_CR2_CC_SUSTAIN = 24
_CR2_CC_RELEASE = 72
# A short release (~0.1s) keeps tonal note-offs from clicking without leaving a tail.
_CR2_TONAL_RELEASE = 3
# Percussive noise burst: instant attack, decay to silence, short release.
_CR2_NOISE_DECAY = 10
_CR2_NOISE_RELEASE = 4


def midi_float(freq: float) -> float:
    """Fractional MIDI note number for `freq` Hz (A4 = 69)."""
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def nearest_note(freq: float, cfg: EncodeConfig) -> int:
    """Nearest MIDI note to `freq`, clamped to the synth's playable note range."""
    return int(np.clip(round(midi_float(freq)), cfg.note_min, cfg.note_ceiling))


def bend_value(midf: float, base_note: int, bend_range: float) -> int:
    """Signed 14-bit pitch-bend (mido range -8192..8191) for `midf` played as `base_note`."""
    semis = midf - base_note
    return int(np.clip(round(semis / bend_range * 8192.0), -8192, 8191))


def amp_ratio(amp: float, ref: float) -> float:
    """Amplitude relative to `ref`, clipped to the dynamic-range window."""
    return float(np.clip(amp / ref, _AMP_FLOOR, 1.0))


def amp_to_velocity(amp: float, ref: float, cfg: EncodeConfig) -> int:
    """Note-on velocity (1-127) whose rendered level is `amp / ref` on the target synth.

    SF2 (GM) attenuates by 40*log10(v/127) dB, so v = 127*sqrt(a); CR2 scales pulse
    width linearly by v/127.
    """
    a = amp_ratio(amp, ref)
    return max(1, int(round(127.0 * (a if cfg.synth == "cr2" else np.sqrt(a)))))


def cr2_hz_scale(note: int) -> float:
    """CR2's pitch-dependent pulse-width factor for base note `note` (0 at its top pitch)."""
    return max(0.0, 1.0 - 2.0 ** ((note - CR2_MAX_PITCH) / 12.0))


def amp_to_volume(amp: float, ref: float, note: int) -> int:
    """CR2 CC7 (0-127) giving level `amp / ref` on `note` at full velocity."""
    a = amp_ratio(amp, ref)
    return int(round(127.0 * a / max(cr2_hz_scale(note), a)))


def _assign_channels(partials: list[Partial], cfg: EncodeConfig) -> dict[int, int]:
    """Greedy interval scheduling of partials onto tonal channels (no temporal overlap)."""
    channels = cfg.tonal_channels
    chan_end = {ch: -1e18 for ch in channels}
    assignment: dict[int, int] = {}
    for idx, p in enumerate(partials):
        start, end = p.times[0], p.times[-1]
        free = [ch for ch in channels if chan_end[ch] <= start]
        if not free:
            continue  # would corrupt a channel's pitch-bend stream; drop it
        # Reuse the most-recently-freed channel so partials pack onto as few channels as possible.
        ch = max(free, key=lambda c: chan_end[c])
        chan_end[ch] = end
        assignment[idx] = ch
    return assignment


def build_midi(
    partials: list[Partial],
    perc_events: list[PercEvent],
    cfg: EncodeConfig,
    tonal_program: int,
) -> mido.MidiFile:
    """Assemble partials and percussion events into a multi-channel SMF."""
    events: dict[int, list[tuple[int, int, mido.Message]]] = {}

    def add(channel: int, tick: int, prio: int, msg: mido.Message) -> None:
        events.setdefault(channel, []).append((tick, prio, msg))

    assignment = _assign_channels(partials, cfg)
    ref_amp = max((p.peak_amp for p in partials), default=1.0) or 1.0
    used_channels = sorted(set(assignment.values()))

    for ch in used_channels:
        _setup_tonal_channel(add, ch, tonal_program, cfg)

    dt = 1.0 / cfg.control_rate_hz
    for idx, p in enumerate(partials):
        ch = assignment.get(idx)
        if ch is None:
            continue
        _emit_partial(add, ch, p, ref_amp, dt, cfg)

    if perc_events:
        _emit_percussion(add, perc_events, cfg)

    return _assemble(events, cfg)


def _cc(ch: int, control: int, value: int) -> mido.Message:
    return mido.Message("control_change", channel=ch, control=control, value=int(value))


def _setup_tonal_channel(add, ch: int, program: int, cfg: EncodeConfig) -> None:
    bend = int(np.clip(round(cfg.bend_range), 0, 127))
    if cfg.synth == "cr2":
        # CR2: reset first (clears CCs to defaults), then volume + CC31 bend range. The
        # default ADSR (instant attack, full sustain, zero release) already suits sustained
        # speech partials; add only a short release so note-offs don't click.
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_RESET, 0))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_VOLUME, 127))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_BEND_RANGE, bend))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_RELEASE, _CR2_TONAL_RELEASE))
        return
    add(ch, 0, _PRIO_SETUP, mido.Message("program_change", channel=ch, program=program))
    add(ch, 0, _PRIO_SETUP, _cc(ch, 7, 127))
    # RPN 0,0 = pitch-bend sensitivity, in semitones.
    add(ch, 0, _PRIO_SETUP, _cc(ch, 101, 0))
    add(ch, 0, _PRIO_SETUP, _cc(ch, 100, 0))
    add(ch, 0, _PRIO_SETUP, _cc(ch, 6, bend))
    add(ch, 0, _PRIO_SETUP, _cc(ch, 38, 0))


def _emit_partial(
    add, ch: int, p: Partial, ref_amp: float, dt: float, cfg: EncodeConfig
) -> None:
    t0, t1 = float(p.times[0]), float(p.times[-1])
    n = max(2, int((t1 - t0) / dt) + 1)
    grid_t = np.linspace(t0, t1, n)
    g_freq = np.interp(grid_t, p.times, p.freqs)
    g_amp = np.interp(grid_t, p.times, p.amps)

    # CR2 applies CC7 channel volume per coil pulse, so with one partial per channel we draw
    # the amplitude contour as CC7 and trigger at full velocity -- no amplitude retriggering,
    # which removes the note churn and clicks. GM has no clean per-note volume here, so it
    # still carries amplitude in the note-on velocity and retriggers on big level jumps.
    cc7_amp = cfg.synth == "cr2"
    base_note: int | None = None
    open_note: int | None = None
    onset_amp = 1e-12  # amplitude at the current note's onset (for retrigger decisions)
    last_vol: int | None = None
    for tt, ff, aa in zip(grid_t, g_freq, g_amp):
        tick = cfg.time_to_ticks(tt)
        midf = midi_float(ff)
        retrigger = base_note is None or abs(midf - base_note) > cfg.bend_range
        if not cc7_amp:
            retrigger = (
                retrigger
                or abs(20.0 * np.log10(aa / onset_amp + 1e-12)) > cfg.amp_retrigger_db
            )
        if retrigger:
            if open_note is not None:
                add(
                    ch,
                    tick,
                    _PRIO_OFF,
                    mido.Message("note_off", channel=ch, note=open_note, velocity=0),
                )
            base_note = nearest_note(ff, cfg)
            add(
                ch,
                tick,
                _PRIO_SETUP,
                mido.Message(
                    "pitchwheel",
                    channel=ch,
                    pitch=bend_value(midf, base_note, cfg.bend_range),
                ),
            )
            if cc7_amp:
                last_vol = amp_to_volume(aa, ref_amp, base_note)
                add(ch, tick, _PRIO_CTRL, _cc(ch, _CR2_CC_VOLUME, last_vol))
                velocity = 127
            else:
                velocity = amp_to_velocity(aa, ref_amp, cfg)
            add(
                ch,
                tick,
                _PRIO_ON,
                mido.Message("note_on", channel=ch, note=base_note, velocity=velocity),
            )
            open_note = base_note
            onset_amp = float(aa)
        else:
            add(
                ch,
                tick,
                _PRIO_CTRL,
                mido.Message(
                    "pitchwheel",
                    channel=ch,
                    pitch=bend_value(midf, base_note, cfg.bend_range),
                ),
            )
            if cc7_amp:
                vol = amp_to_volume(aa, ref_amp, base_note)
                if vol != last_vol:
                    add(ch, tick, _PRIO_CTRL, _cc(ch, _CR2_CC_VOLUME, vol))
                    last_vol = vol
    if open_note is not None:
        add(
            ch,
            cfg.time_to_ticks(t1) + 1,
            _PRIO_OFF,
            mido.Message("note_off", channel=ch, note=open_note, velocity=0),
        )


def _emit_percussion(add, perc_events: list[PercEvent], cfg: EncodeConfig) -> None:
    ch = PERCUSSION_CHANNEL
    hit_ticks = max(1, cfg.time_to_ticks(0.05))
    cr2 = cfg.synth == "cr2"
    if cr2:
        # CR2 channel 10 is a pitched-noise voice: the note pitch sets the noise band
        # center, so map the spectral centroid to a note (bright fricatives -> high noise,
        # plosives -> low thud). Shape each hit with a percussive ADSR.
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_RESET, 0))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_VOLUME, 127))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_ATTACK, 0))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_DECAY, _CR2_NOISE_DECAY))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_SUSTAIN, 0))
        add(ch, 0, _PRIO_SETUP, _cc(ch, _CR2_CC_RELEASE, _CR2_NOISE_RELEASE))
    for ev in perc_events:
        on = cfg.time_to_ticks(ev.time)
        note = nearest_note(ev.centroid, cfg) if cr2 else ev.note
        velocity = amp_to_velocity(ev.amp, 1.0, cfg)
        add(
            ch,
            on,
            _PRIO_ON,
            mido.Message("note_on", channel=ch, note=note, velocity=velocity),
        )
        add(
            ch,
            on + hit_ticks,
            _PRIO_OFF,
            mido.Message("note_off", channel=ch, note=note, velocity=0),
        )


def _assemble(
    events: dict[int, list[tuple[int, int, mido.Message]]], cfg: EncodeConfig
) -> mido.MidiFile:
    mid = mido.MidiFile(type=1, ticks_per_beat=cfg.ppq)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(cfg.bpm), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    for ch in sorted(events):
        msgs = sorted(events[ch], key=lambda e: (e[0], e[1]))
        track = mido.MidiTrack()
        prev = 0
        for tick, _prio, msg in msgs:
            track.append(msg.copy(time=tick - prev))
            prev = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)

    return mid
