import math
from dataclasses import dataclass

EPSILON: float = 1e-3


class Curve:
    def update(self, dt: int) -> float:
        return 0

    def adjust_start(self, value: float):
        pass

    @property
    def is_target_reached(self) -> bool:
        return True


class FlatCurve(Curve):
    value: float

    def __init__(self, value: float):
        self.value = value

    def update(self, dt):
        return self.value

    @property
    def is_target_reached(self) -> bool:
        return False


class LinearCurve(Curve):
    slope: float
    target: float
    total_time: float
    _elapsed: int = 0
    _target_reached: bool = False
    _current: float

    def __init__(self, start: float, end: float, total_time: float):
        if total_time <= 0:
            self._target_reached = True
        else:
            self.slope = (end - start) / total_time
        self.target = end
        self._current = start
        self.total_time = total_time

    def adjust_start(self, value: float):
        if (value > self.target and self.slope < 0) or (
            value < self.target and self.slope > 0
        ):
            self._current = value

    def update(self, dt: int) -> float:
        if self.total_time <= 0 or self.is_target_reached:
            self._target_reached = True
            return self.target
        t = min(dt, self.total_time - self._elapsed)
        self._elapsed += t
        self._current += self.slope * t
        self._target_reached = math.fabs(self._current - self.target) < EPSILON
        return self.target if self.is_target_reached else self._current

    @property
    def is_target_reached(self) -> bool:
        return self._target_reached


@dataclass
class ExponentialCurve(Curve):
    current: float
    target: float
    tau: float

    def update(self, dt: int) -> float:
        """Exponential approach: current += (target - current) * (1 - exp(-dt/tau))"""
        if self.tau <= 0:
            return self.target
        coef = 1 - math.exp(-dt / self.tau)
        self.current += (self.target - self.current) * coef
        return self.current

    def adjust_start(self, value: float):
        if (value > self.target and self.current > self.target) or (
            value < self.target and self.current < self.target
        ):
            self.current = value

    @property
    def is_target_reached(self) -> bool:
        return math.fabs(self.current - self.target) <= EPSILON
