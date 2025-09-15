from teslasynth.core import SynthConfig, Instrument, Note, Pulse
from typing import Iterator


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
