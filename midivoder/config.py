"""Configuration for the WAV -> MIDI encoder.

Every field that trades resources against fidelity is surfaced here and on the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

# General MIDI percussion lives on channel 10 (zero-based index 9).
# CHIME RED II uses the same channel 10 for its pitched-noise voice.
PERCUSSION_CHANNEL = 9

# CHIME RED II only listens on MIDI channels 1-8 (tonal, zero-based 0-7) and 10.
CR2_MAX_TONAL_INDEX = 7
# CHIME RED II ignores note-ons above this note (C7, ~2093 Hz) on every channel.
CR2_MAX_PITCH = 96


@dataclass
class EncodeConfig:
    """Analysis, tracking and MIDI output parameters for the encoder."""

    # --- analysis ---
    sample_rate: int = 16000  # voice bandwidth; resampled on load
    frame_ms: float = 40.0  # STFT window length
    hop_ms: float = 10.0  # STFT hop (analysis time resolution)
    max_peaks_per_frame: int = 24  # spectral peaks considered before tracking
    peak_floor_db: float = -60.0  # peaks below (max - this) are ignored
    fmin: float = 60.0  # ignore spectral content below this (Hz)
    fmax: float = 6000.0  # ignore spectral content above this (Hz)

    # --- partial tracking ---
    match_cents: float = 80.0  # max pitch jump to continue a partial
    max_gap_frames: int = 2  # frames a partial may vanish before death
    min_partial_ms: float = 40.0  # discard partials shorter than this

    # --- MIDI / resource caps ---
    max_channels: int = 16  # total MIDI channels used (incl. percussion)
    bpm: float = 120.0  # tempo meta event
    ppq: int = 480  # ticks per quarter note (timing granularity)
    control_rate_hz: float = 50.0  # pitch-bend event rate
    min_note_ms: float = 30.0  # drop notes shorter than this
    bend_range: float = 2.0  # pitch-bend range in semitones (RPN 0,0)
    amp_retrigger_db: float = (
        6.0  # retrigger a note (new velocity) when level moves this much
    )
    note_min: int = 24  # clamp emitted note numbers
    note_max: int = 108

    # --- instruments ---
    instrument: str = "auto"  # "auto" or a GM program number as a string
    enable_percussion: bool = True
    synth: str = "gm"  # target synth dialect: "gm" or "cr2" (CHIME RED II)

    # --- percussion ---
    perc_onset_db: float = -45.0  # transient/noise floor for percussion hits
    perc_min_interval_ms: float = 60.0  # min spacing between percussion hits

    def __post_init__(self) -> None:
        if self.max_channels < 1:
            raise ValueError("max_channels must be >= 1")
        if self.ppq < 24:
            raise ValueError("ppq is unreasonably small")
        if self.synth not in ("gm", "cr2"):
            raise ValueError("synth must be 'gm' or 'cr2'")

    @property
    def hop_samples(self) -> int:
        """Analysis hop in samples."""
        return max(1, round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def frame_samples(self) -> int:
        """Analysis window length in samples (at least two hops)."""
        return max(
            self.hop_samples * 2, round(self.sample_rate * self.frame_ms / 1000.0)
        )

    @property
    def seconds_per_tick(self) -> float:
        """Duration of one MIDI tick at the configured tempo and PPQ."""
        return (60.0 / self.bpm) / self.ppq

    def time_to_ticks(self, seconds: float) -> int:
        """Convert `seconds` to a non-negative absolute tick count."""
        return max(0, round(seconds / self.seconds_per_tick))

    @property
    def tonal_channels(self) -> list[int]:
        """MIDI channels available for pitched partials (percussion reserved)."""
        budget = self.max_channels
        last_index = CR2_MAX_TONAL_INDEX if self.synth == "cr2" else 15
        channels: list[int] = []
        for ch in range(16):
            if ch > last_index:
                break
            if len(channels) >= (budget - 1 if self.enable_percussion else budget):
                break
            if self.enable_percussion and ch == PERCUSSION_CHANNEL:
                continue
            channels.append(ch)
        return channels

    @property
    def note_ceiling(self) -> int:
        """Highest note number the target synth will sound."""
        if self.synth != "cr2":
            return self.note_max
        # CR2 scales pulse width by 1 - hz(note)/hz(CR2_MAX_PITCH), silencing its top note.
        return min(self.note_max, CR2_MAX_PITCH - 1)

    @property
    def partial_fmax(self) -> float:
        """Upper frequency bound for spectral peaks fed to partial tracking."""
        if self.synth != "cr2":
            return self.fmax
        # Peaks at or above the half-semitone past the ceiling would round to a base note
        # CR2 cannot sound, so they are never tracked.
        return min(self.fmax, 440.0 * 2.0 ** ((self.note_ceiling + 0.5 - 69.0) / 12.0))

    @property
    def max_partials(self) -> int:
        """Maximum simultaneous tracked partials (one per tonal channel)."""
        return max(1, len(self.tonal_channels))
