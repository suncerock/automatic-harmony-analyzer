from collections import OrderedDict
from dataclasses import dataclass

from src.data.hchord import _shorthands


@dataclass(frozen=True)
class ChordLabelConfig:
    """Minimal label configuration used by the inference-time chord model."""

    require_harte: bool = False
    harte_vocab: str = "tetrads"

    require_rbp: bool = False
    rbp_enforce_root: bool = False
    rbp_enforce_bass: bool = False
    rbp_include_extended: bool = False

    @classmethod
    def rbp_only(cls) -> "ChordLabelConfig":
        return cls(require_rbp=True)

    @classmethod
    def mcfee_structured(cls, harte_vocab: str = "tetrads") -> "ChordLabelConfig":
        return cls(require_harte=True, harte_vocab=harte_vocab, require_rbp=True)

    @classmethod
    def poltronieri_structured(cls) -> "ChordLabelConfig":
        return cls.rbp_only()


ALL_TRIADS_INTERVALS = OrderedDict((triad, _shorthands[triad]) for triad in [
    "maj", "min", "dim", "aug", "sus4", "sus2"
])
ALL_TETRADS_INTERVALS = OrderedDict((tetrad, _shorthands[tetrad]) for tetrad in [
    "7", "maj7", "min7", "dim7", "hdim7", "minmaj7", "aug7", "7sus4", "maj6", "min6",
    "maj", "min", "dim", "aug", "sus4", "sus2"
])
