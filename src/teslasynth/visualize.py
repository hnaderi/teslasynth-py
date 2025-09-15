#!/usr/bin/env python3

from matplotlib.ticker import FuncFormatter
import mido
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from collections import defaultdict
import sys
import traceback
from dataclasses import dataclass, field
from collections import namedtuple
import argparse
from abc import ABC, abstractmethod
from typing import Iterator

Pulse = namedtuple("Pulse", ["start", "end"])
Note = namedtuple("Note", ["number", "start", "end", "velocity"])


@dataclass
class SynthConfig:
    sample_rate: int = 100000
    max_on_time: int = 100
    min_on_time: int = 0
    ringdown_time: int = 100

    sampling_interval: float = field(init=False)
    note_numerator: float = field(init=False)
    max_ticks: int = field(init=False)
    min_ticks: int = field(init=False)

    def __post_init__(self):
        self.sampling_interval = 1e6 / self.sample_rate
        self.note_numerator = self.sample_rate / 440
        self.max_ticks = int(self.max_on_time / self.sampling_interval)
        self.min_ticks = int(self.min_on_time / self.sampling_interval)

    def note_period(self, number: int) -> int:
        return int(self.note_numerator / 2 ** ((number - 69) / 12))


class Instrument(ABC):
    @abstractmethod
    def pulses(self, note: Note, config: SynthConfig) -> Iterator[Pulse]:
        pass


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


@dataclass
class Voice:
    instrument: Instrument
    notes: Iterator[Note]

    @staticmethod
    def load_midi(
        instrument: Instrument,
        midi_file: mido.MidiFile,
        config: SynthConfig,
        channel: int = 0,
    ):
        ticks_per_beat = midi_file.ticks_per_beat
        tempo = Voice._get_tempo(midi_file)
        samples_per_beat = tempo / config.sampling_interval
        samples_per_tick = samples_per_beat / ticks_per_beat

        notes = Voice._extract_notes(
            midi_file,
            samples_per_tick=samples_per_tick,
            channel=channel,
        )

        if not notes:
            raise Exception(f"No notes found for channel {channel}")

        return Voice(instrument, notes)

    def _get_tempo(midi_file: mido.MidiFile):
        """Get the first tempo from the MIDI file or default to 500000 (120 BPM)."""
        tempo = 500000  # Default tempo in microseconds per quarter note
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    return tempo
        return tempo

    def _extract_notes(
        midi_file: mido.MidiFile,
        samples_per_tick: float,
        channel: int,
    ) -> list[Note]:
        """Extract note events from a MIDI file for a specific channel."""
        notes = []  # Store (note, start_time, end_time, velocity) tuples in ticks
        active_notes = defaultdict(list)  # Track active note_on events
        current_ticks = 0  # Cumulative ticks

        for track in midi_file.tracks:
            for msg in track:
                current_ticks += msg.time  # Add delta time in ticks
                if msg.type in ("note_on", "note_off") and msg.channel == channel:
                    if msg.type == "note_on" and msg.velocity > 0:
                        # Store note_on with start time
                        active_notes[msg.note].append((current_ticks, msg.velocity))
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        # Match note_off with the most recent note_on
                        if msg.note in active_notes and active_notes[msg.note]:
                            start_ticks, velocity = active_notes[msg.note].pop(0)
                            notes.append(
                                Note(
                                    number=msg.note,
                                    start=int(start_ticks * samples_per_tick),
                                    end=int(current_ticks * samples_per_tick),
                                    velocity=velocity,
                                )
                            )

        return sorted(notes, key=lambda x: x[1])


@dataclass
class SynthesizedTrack:
    pulses: list[Pulse]
    sample_rate: int
    signal: np.linspace = field(init=False)
    num_samples: int = field(init=False)

    def __post_init__(self):
        self.num_samples = max(end for start, end in self.pulses) + 1
        self.signal = np.zeros(self.num_samples, dtype=bool)
        for start, end in self.pulses:
            self.signal[start:end] = True

    def save_wav_file(self, output_path: str):
        audio_signal = self.signal.astype(np.uint8) * 255
        wavfile.write(output_path, self.sample_rate, audio_signal)

    def print_statistics(self):
        duration = 1e6 / self.sample_rate
        print(f"Tick duration: {duration}uS")
        total_on_time = sum(end - start for start, end in self.pulses)
        print(f"Total duty: {total_on_time * 100 / self.num_samples:.2f}%")
        total_pulses = len(self.pulses)
        print(f"Total samples: {self.num_samples}, pulses: {total_pulses}")

        min_pulse_length = min(end - start for start, end in self.pulses) * duration
        max_pulse_length = max(end - start for start, end in self.pulses) * duration
        avg_pulse_length = (total_on_time / total_pulses) * duration
        print(
            f"Min: {min_pulse_length}uS, Max: {max_pulse_length}uS, Avg: {avg_pulse_length:.2f}uS"
        )


@dataclass(frozen=True)
class Synth:
    config: SynthConfig

    def play(self, voice: Voice) -> SynthesizedTrack:
        pulses = Synth._process_pulses(
            pulses=(
                pulse
                for note in voice.notes
                for pulse in voice.instrument.pulses(note, self.config)
            ),
            ringdown_time=self.config.ringdown_time,
        )
        return SynthesizedTrack(list(pulses), sample_rate=self.config.sample_rate)

    def _process_pulses(
        pulses: Iterator[Pulse],
        ringdown_time: int,
    ):
        last_pulse_off = -ringdown_time
        current_pulse_start = None
        for start, end in pulses:
            if last_pulse_off < start:
                current_pulse_start = None
            if not current_pulse_start and start - last_pulse_off >= ringdown_time:
                current_pulse_start = start
                last_pulse_off = end
                yield (start, end)


def draw_piano_roll(
    voice: Voice,
    ax: plt.Axes,
    config: SynthConfig,
):
    cmap = plt.colormaps["tab20"]
    min_note = min([n[0] for n in voice.notes]) - 1 if voice.notes else 0
    max_note = max([n[0] for n in voice.notes]) + 1 if voice.notes else 127
    for note in voice.notes:
        start_sec = note.start * config.sampling_interval
        end_sec = note.end * config.sampling_interval
        ax.hlines(
            y=note,
            xmin=start_sec,
            xmax=end_sec,
            color=cmap((note.number % 12) / 12),
            alpha=0.6,
            linewidth=2,
        )
    ax.set_ylabel("MIDI Note Number")
    ax.set_ylim(min_note, max_note)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def time_formatter(x, pos):
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min

    if x_range > 1:
        unit = "s"
        factor = 1
    elif x_range > 1e-3:
        unit = "ms"
        factor = 1e3
    else:
        unit = "µs"
        factor = 1e6

    return f"{x * factor:.2f} {unit}"


def draw_pwm(
    track: SynthesizedTrack,
    ax: plt.Axes,
    config: SynthConfig,
):
    timeline = np.arange(track.num_samples)
    if track.num_samples > 0:
        ax.step(timeline, track.signal, color="red", where="post", linewidth=1)
        ax.set_ylabel("PWM Signal (On/Off)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (seconds)")


def visualize(
    voice: Voice,
    track: SynthesizedTrack,
    config: SynthConfig,
    channel,
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        num="MIDI interrupt visualization",
    )

    draw_piano_roll(voice, ax1, config)
    ax1.set_title(f"Piano Roll: Channel {channel}")
    draw_pwm(track, ax2, config)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(time_formatter))
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser("MIDI Interrupt visualizer")
parser.add_argument("file", help="MIDI file")
parser.add_argument("--sample-rate", "-S", type=int, default=100000)
parser.add_argument("--channel", "-c", type=int, default=0)
parser.add_argument("-o", "--output", help="write .wav output file", required=False)
parser.add_argument(
    "-q",
    "--no-visualization",
    help="Don't show visualization",
    action="store_true",
)


def main():
    try:
        args = parser.parse_args()

        config = SynthConfig(
            sample_rate=args.sample_rate,
        )

        midi_file_path = args.file
        channel = args.channel
        midi_file = mido.MidiFile(midi_file_path)

        instrument = BasicInstrument()
        voice = Voice.load_midi(instrument, midi_file, config)
        synth = Synth(config)
        track = synth.play(voice)

        track.print_statistics()

        if args.output:
            track.save_wav_file(args.output)

        if not args.no_visualization:
            visualize(
                voice=voice,
                track=track,
                config=config,
                channel=channel,
            )
    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
