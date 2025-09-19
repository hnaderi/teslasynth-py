from dataclasses import dataclass


from .envelopes import ADSRConfig, CurveType
from .lfo import VibratoConfig


@dataclass(frozen=True)
class Instrument:
    envelope: ADSRConfig
    vibato: VibratoConfig | None = None


all: list[Instrument] = [
    Instrument(
        envelope=ADSRConfig(1000, 2000, 0.5, 1500, curve=CurveType.Exponential),
        vibato=None,
    )
]
