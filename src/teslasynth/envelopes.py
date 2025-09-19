from .curves import Curve, ExponentialCurve, LinearCurve, FlatCurve, EPSILON
from enum import Enum
from dataclasses import dataclass
import math


class CurveType(Enum):
    Linear = 0
    Exponential = 1


@dataclass(frozen=True)
class ADSRConfig:
    attack: float
    decay: float
    sustain_level: float
    release: float
    curve: CurveType = CurveType.Exponential


class Envelope:
    curves: list[Curve] = []
    _current_curve: int | None = None
    _is_off: bool = True
    _is_released: bool = False
    _time: int = 0
    _level: float = 0

    def __init__(self, curves: list[Curve]):
        self.curves = curves
        self._current_curve = 0
        self._is_off = False

    def update(self, time: int, is_note_on: bool) -> float:
        dt = time - self._time
        if dt <= 0:
            return self._level
        self._time = time

        if self._current_curve is None or len(self.curves) < 1:
            return self._level

        if not self._is_released and not is_note_on:
            self._is_released = True
            self._current_curve = len(self.curves) - 1

        curve = self.curves[self._current_curve]
        if curve.is_target_reached:
            if self._current_curve < len(self.curves) - 1:
                self._current_curve += 1
            else:
                self._current_curve = None
                self._is_off = True
        self._level = curve.update(dt)

        return self._level

    @property
    def is_on(self):
        return not self.is_off

    @property
    def is_off(self):
        return self._is_off


class ADSREnvelope(Envelope):
    def __init__(self, config: ADSRConfig):
        sustain = FlatCurve(config.sustain_level)
        if config.curve == CurveType.Exponential:
            log_factor = -math.log(EPSILON)  # ~6.907 for epsilon=0.001

            attack = ExponentialCurve(
                0, 1, tau=config.attack / log_factor if config.attack > 0 else 0
            )
            decay = ExponentialCurve(
                1,
                config.sustain_level,
                tau=config.decay / log_factor if config.decay > 0 else 0,
            )
            release = ExponentialCurve(
                config.sustain_level,
                0,
                tau=config.release / log_factor if config.release > 0 else 0,
            )
        else:
            attack = LinearCurve(0, 1, config.attack)
            decay = LinearCurve(1, config.sustain_level, config.decay)
            release = LinearCurve(config.sustain_level, 0, config.release)

        super().__init__([attack, decay, sustain, release])
