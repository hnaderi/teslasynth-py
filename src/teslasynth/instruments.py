from teslasynth.core import SynthConfig, Instrument, Note, Pulse
from typing import Iterator
from collections import namedtuple
from dataclasses import dataclass


class BasicInstrument(Instrument):
    def pulses(self, note: Note, config: SynthConfig) -> Iterator[Pulse]:
        period = config.note_period(note.number)
        volume = note.velocity / 127
        ticks = max(int(volume * config.max_ticks), config.min_ticks)

        number_of_pulses = (note.end - note.start) // period

        return [
            (start, end)
            for start, end in [
                (start, start + ticks)
                for start in [note.start + i * period for i in range(number_of_pulses)]
            ]
            if end > start
        ]


class Chiptune(Instrument):
    def pulses(self, note: Note, config: SynthConfig) -> Iterator[Pulse]:
        period = config.note_period(note.number)
        volume = note.velocity / 127
        ticks = max(int(volume * config.max_ticks), config.min_ticks)
        min_ticks = max(config.min_ticks, ticks // 4)

        number_of_pulses = (note.end - note.start) // period

        return [
            (start, end)
            for start, end in [
                (start, start + (ticks if i % 2 == 0 else min_ticks))
                for i, start in enumerate(
                    [note.start + i * period for i in range(number_of_pulses)]
                )
            ]
            if end > start
        ]


Envelope = namedtuple("Envelope", ["attack", "decay", "sustain", "release"])


@dataclass
class ADSRInstrument(Instrument):
    envelope: Envelope

    def pulses(self, note: Note, config: SynthConfig) -> Iterator[Pulse]:
        period = config.note_period(note.number)
        volume = note.velocity / 127
        ticks = max(int(volume * config.max_ticks), config.min_ticks)

        number_of_pulses = (note.end - note.start) // period

        return [
            (start, end)
            for start, end in [
                (start, start + ticks)
                for start in [note.start + i * period for i in range(number_of_pulses)]
            ]
            if end > start
        ]
