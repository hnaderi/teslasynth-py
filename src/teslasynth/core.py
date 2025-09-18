import mido
from collections import defaultdict
import heapq
from enum import Enum, IntEnum
import numpy as np
from scipy.io import wavfile
from dataclasses import dataclass, field
from collections import namedtuple
import math

Pulse = namedtuple("Pulse", ["start", "end"])
NoteData = namedtuple("Note", ["number", "start", "end", "velocity"])


@dataclass
class Limits:
    max_on_time: int | None
    min_on_time: int | None
    min_deadtime: int | None
    max_duty: int | None
    max_notes: int = 4


@dataclass
class SamplingConfig:
    rate: int = 1e5
    interval: float = field(init=False)

    def __post_init__(self):
        self.interval = 1e6 / self.rate

    def freq_period(self, freq: float) -> int:
        return int(self.rate / freq)


@dataclass
class SynthConfig:
    limits: Limits
    a4: float = 440

    def note_freq(self, number: int) -> float:
        return self.a4 * (2 ** ((number - 69) / 12))


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


class EnvelopeState(IntEnum):
    Attack = 1
    Decay = 2
    Sustain = 3
    Release = 4
    Off = 5


@dataclass(frozen=True)
class VibratoConfig:
    depth: float
    freq: float


class EnvelopeRateType(Enum):
    Linear = 0
    Exponential = 1


@dataclass(frozen=True)
class Rate:
    value: float
    type: EnvelopeRateType = EnvelopeRateType.Linear

    def __call__(self, initial: float, ticks: int) -> float:
        match self.type:
            case EnvelopeRateType.Linear:
                return min(1.0, self.value * ticks + initial)
            case EnvelopeRateType.Exponential:
                return min(1.0, (self.value**ticks) * initial)

    def __str__(self):
        return f"{self.type}: {self.value}"


@dataclass(frozen=True)
class ADSRConfig:
    attack: Rate
    decay: Rate
    sustain_level: float
    release: Rate

    @staticmethod
    def linear(attack: float, decay: float, sustain: float, release: float):
        return ADSRConfig(Rate(attack), Rate(decay), sustain, Rate(release))


@dataclass(frozen=True)
class Instrument:
    envelope: ADSRConfig
    vibato: VibratoConfig | None = None


class Envelope:
    _level: float = 0
    _time: int = 0
    _state: EnvelopeState = EnvelopeState.Attack
    _config: ADSRConfig

    def __init__(self, config: ADSRConfig):
        self._config = config
        self._state = EnvelopeState.Attack

    def update(self, time: int, is_note_on: bool) -> float:
        if not is_note_on and self._state is not EnvelopeState.Off:
            self._state = EnvelopeState.Release

        dt = time - self._time
        if dt < 0:
            return self._level
        config = self._config
        match self._state:
            case EnvelopeState.Attack:
                self._level = config.attack(self._level, dt)
                if self._level >= 1:
                    self._state = EnvelopeState.Decay
            case EnvelopeState.Decay:
                self._level = max(
                    config.decay(self._level, dt),
                    config.sustain_level,
                )
                if self._level <= config.sustain_level:
                    self._state = EnvelopeState.Sustain
            case EnvelopeState.Release:
                self._level = config.release(self._level, dt)
                if self._level <= 0.01:
                    self._state = EnvelopeState.Off
                    self._level = 0

        self._time += dt
        return self._level

    @property
    def is_on(self):
        return not self.is_off

    @property
    def is_off(self):
        return self._state is EnvelopeState.Off


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


class Note:
    instrument: Instrument
    config: SynthConfig
    number: int
    velocity: int
    frequency: float

    env: Envelope
    lfo: LFO | None

    _current_time: int = -1
    _start_time: int = -1
    _release_time: int = -1
    _off_time: int = -1

    def __init__(
        self,
        instrument: Instrument,
        synth_config: SynthConfig,
        number: int,
        velocity: int,
    ):
        self.instrument = instrument
        self.config = synth_config
        self.number = number
        self.frequency = synth_config.note_freq(number)
        self.velocity = velocity
        self.env = Envelope(instrument.envelope)
        self.lfo = LFO(instrument.vibato) if instrument.vibato else None

    def update(self, micros: int) -> list[Pulse]:
        counter = 0
        pulses = []
        while counter <= micros and self._off_time < 0:
            current_time = self._current_time + counter
            is_note_on = (self._release_time < 0 and self._start_time >= 0) or (
                current_time < self._release_time
            )
            env_lvl = self.env.update(current_time, is_note_on=is_note_on)
            volume = self.config.limits.max_on_time * (self.velocity / 127) * env_lvl
            offset = self.lfo.update(current_time) if self.lfo else 0
            period = int(1e6 / (self.frequency + offset))
            if volume > 0:
                width = min(self.config.limits.max_on_time, volume, period // 2)
                start_time = self._current_time + counter
                pulses.append(Pulse(start_time, start_time + width))
            elif self._off_time < 0 and self.env.is_off:
                self._off_time = current_time

            counter += period

        self._current_time += counter

        return pulses

    def start(self, current_time: int):
        self._current_time = current_time
        self._start_time = current_time

    def release(self, time: int):
        self._release_time = time

    @property
    def release_time(self):
        return self._release_time

    @property
    def start_time(self):
        return self._start_time

    @property
    def off_time(self):
        return self._off_time


Instruments: list[Instrument] = [
    Instrument(
        envelope=ADSRConfig.linear(0.01, -0.02, 0.5, -0.05),
        vibato=VibratoConfig(5, 5),
    )
]


class SynthChannel:
    config: SynthConfig
    playing: dict[int, Note] = {}
    controls: dict[int, int] = {}
    instrument: Instrument

    active_notes = defaultdict(list)
    notes: list[NoteData] = []
    max_notes = 0

    def __init__(
        self,
        config: SynthConfig,
        instrument: Instrument | None = None,
    ):
        self.config = config
        self.instrument = instrument or Instruments[0]

    def __note_received(self, number: int, velocity: int, time: int):
        self.active_notes[number].append((time, velocity))

    def __note_processed(self, number: int, time: int):
        if number in self.active_notes and self.active_notes[number]:
            start, velocity = self.active_notes[number].pop(0)
            self.notes.append(
                NoteData(
                    number=number,
                    start=int(start),
                    end=int(time),
                    velocity=velocity,
                )
            )

    def on_note_on(self, number: int, velocity: int, time: int):
        note = self.playing.get(number)
        if velocity == 0:
            if note:
                note.release()
                self.__note_processed(number, time)
        else:
            self.__note_received(number, velocity, time)
            new_note = Note(
                instrument=self.instrument,
                synth_config=self.config,
                number=number,
                velocity=velocity,
            )
            if len(self.playing) < self.config.limits.max_notes or note:
                self.playing[number] = new_note
            else:
                note_to_steal = max(
                    self.playing.values(),
                    key=lambda n: (n.off_time, n.release_time),
                )
                if note_to_steal.off_time < 0 and note_to_steal.release_time < 0:
                    note_to_steal = min(
                        self.playing.values(),
                        key=lambda n: n.start_time,
                    )
                self.max_notes = max(len(self.playing), self.max_notes)
                self.playing.pop(note_to_steal.number)
                self.playing[number] = new_note
            new_note.start(time)

    def on_note_off(self, number: int, velocity: int, time: int):
        self.__note_processed(number, time)
        note = self.playing.get(number)
        if not note:
            return
        note.release(time)

    def on_pitchbend(self, value: int, time: int):
        pass

    def on_program_change(self, program: int):
        if len(Instruments) > program:
            self.instrument = Instruments[program]

    def on_control_change(self, control: int, value: int):
        self.controls[control] = value

    def sample(self, micros: int) -> list[Pulse]:
        return [
            pulse for note in self.playing.values() for pulse in note.update(micros)
        ]


Event = namedtuple("Event", ["time", "idx", "msg"])


@dataclass
class ProcessedTrack:
    notes: list[NoteData]
    messages: list[mido.Message]
    pulses: list[Pulse]
    sample_rate: int

    def make_track(self):
        return SynthesizedTrack(self.pulses, self.sample_rate)


class TeslaSynth:
    config: SynthConfig
    sampling: SamplingConfig

    def __init__(self, config: SynthConfig, sampling: SamplingConfig):
        self.config = config
        self.sampling = sampling

    def _msg_heap(
        self,
        midi_file: mido.MidiFile,
        channel: int,
    ) -> list[Event]:
        msgs = []
        now = 0
        for idx, msg in enumerate(midi_file.merged_track):
            if not hasattr(msg, "channel") or msg.channel is not channel:
                continue
            now += msg.time
            heapq.heappush(msgs, Event(now, idx, msg))

        return msgs

    def synthesize(
        self,
        path: str,
        channel: int = 0,
    ) -> ProcessedTrack:
        midi = mido.MidiFile(path)
        msgs = self._msg_heap(midi, channel)
        ctrl = SynthChannel(self.config)
        tempo = 500000  # half a second for 120bpm
        ticks_per_beat = midi.ticks_per_beat
        now = 0
        pulses = []
        process_messages = []

        while msgs:
            evt = heapq.heappop(msgs)
            msg = evt.msg
            if msg.time > 0:
                delta = msg.time * tempo // ticks_per_beat
                pulses.extend(ctrl.sample(delta))
                now += delta
            match msg.type:
                case "set_tempo":
                    tempo = msg.tempo
                case "note_on":
                    ctrl.on_note_on(number=msg.note, velocity=msg.velocity, time=now)
                case "note_off":
                    ctrl.on_note_off(number=msg.note, velocity=msg.velocity, time=now)
                case "pitchwheel":
                    ctrl.on_pitchbend(msg.pitch, time=now)
                case "control_change":
                    ctrl.on_control_change(msg.control, msg.value)
                case "program_change":
                    ctrl.on_program_change(msg.program)
            process_messages.append(msg)

        sampling_interval = self.sampling.interval
        sampled_pulses = [
            (int(start // sampling_interval), int(end // sampling_interval))
            for start, end in pulses
        ]
        return ProcessedTrack(
            messages=process_messages,
            notes=ctrl.notes,
            pulses=sampled_pulses,
            sample_rate=self.sampling.rate,
        )


def stats(f: str):
    mid = mido.MidiFile(f)
    print(f"Tracks: {len(mid.tracks)}")
    for tnum, track in enumerate(mid.tracks):
        types = set(msg.type for msg in track)
        print(f"Track #{tnum}")
        print(f"Types: {types}")
        print(f"Messages: {len(track)}")

    msgs = [msg for track in mid.tracks for msg in track]
    print("=" * 20)
    print(f"Total messages: {len(msgs)}")
    notes = [msg for msg in msgs if hasattr(msg, "channel")]
    channels = set(msg.channel for msg in notes)
    types = set(msg.type for msg in notes)
    print(f"Channels: {channels}, Types: {types}")

    return mid
