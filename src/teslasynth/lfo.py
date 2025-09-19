from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VibratoConfig:
    depth: float
    freq: float


class LFO:
    _phase: float = 0
    _config: VibratoConfig

    def __init__(self, config: VibratoConfig):
        self._config = config

    def update(self, dt: int) -> float:
        config = self._config
        self._phase += 2 * math.pi * config.freq * (dt / 1e6)
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        return config.depth * math.sin(self._phase)
