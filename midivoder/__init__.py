"""midivoder: vocode a voice WAV into a multi-channel General MIDI file."""

from midivoder.config import EncodeConfig
from midivoder.encoder import encode_wav_to_midi

__all__ = ["EncodeConfig", "encode_wav_to_midi"]
__version__ = "0.1.0"
