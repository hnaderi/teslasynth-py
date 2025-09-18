from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from teslasynth.core import (
    SynthesizedTrack,
    ProcessedTrack,
    ADSRConfig,
    VibratoConfig,
    Envelope,
    LFO,
    Instrument,
)

from collections import namedtuple

Note = namedtuple("Note", ["number", "start", "end", "velocity"])


def draw_piano_roll(proc: ProcessedTrack, ax: plt.Axes):
    notes = proc.notes
    cmap = plt.colormaps["tab20"]
    min_note = min([n[0] for n in notes]) - 1 if notes else 0
    max_note = max([n[0] for n in notes]) + 1 if notes else 127
    for note in notes:
        start_sec = note.start / 1e6
        end_sec = note.end / 1e6
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


def draw_pwm(track: SynthesizedTrack, ax: plt.Axes, sample_rate: int):
    timeline = np.arange(track.num_samples) / sample_rate
    if track.num_samples > 0:
        ax.step(timeline, track.signal, color="red", where="post", linewidth=1)
        ax.set_ylabel("PWM Signal (On/Off)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (seconds)")


def visualize(
    proc: ProcessedTrack,
    track: SynthesizedTrack,
    channel: int,
    sample_rate: int,
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        num="MIDI interrupt visualization",
    )

    draw_piano_roll(proc, ax1)
    ax1.set_title(f"Piano Roll: Channel {channel}")
    draw_pwm(track, ax2, proc.sample_rate)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(time_formatter))
    plt.tight_layout()
    plt.show()


def simulate_adsr(
    config: ADSRConfig,
    duration: int,
    note_on_duration: int,
    sample_rate: int,
):
    env = Envelope(config)
    times = np.arange(0, duration, 1e6 / sample_rate)
    amps = np.zeros_like(times)

    for i, t in enumerate(times):
        is_note_on = t < note_on_duration
        amps[i] = env.update(t, is_note_on)

    return times, amps


def simulate_lfo(
    config: VibratoConfig,
    duration: int,
    sample_rate: int,
):
    lfo = LFO(config)
    times = np.arange(0, duration, 1e6 / sample_rate)
    mods = np.zeros_like(times)

    for i, t in enumerate(times):
        mods[i] = lfo.update(1e6 / sample_rate)

    return times, mods


def visualize_components(
    instrument: Instrument,
    duration=1e6,
    note_on_duration=2e5,
    sample_rate=1e5,
):
    adsr = instrument.envelope
    lfo = instrument.vibato

    fig, axs = plt.subplots(2 if adsr and lfo else 1, 1, figsize=(10, 8))
    axs = np.atleast_1d(axs)

    idx = 0
    if adsr:
        times, amps = simulate_adsr(adsr, duration, note_on_duration, sample_rate)
        axs[idx].plot(times, amps, label="ADSR Amplitude")
        axs[idx].axvline(
            note_on_duration,
            color="r",
            linestyle="--",
            label="Note Off",
        )
        axs[idx].set_title(
            f"Envelope: A={adsr.attack}, D={adsr.decay}, S={adsr.sustain_level}, R={adsr.release}"
        )
        axs[idx].set_xlabel("Time (us)")
        axs[idx].set_ylabel("Amplitude")
        axs[idx].legend()
        axs[idx].grid(True)
        idx += 1

    if lfo:
        times, mods = simulate_lfo(lfo, duration, sample_rate)
        axs[idx].plot(times, mods, label="LFO Modulation")
        axs[idx].set_title(f"LFO: Freq={lfo.freq} Hz, Depth={lfo.depth}")
        axs[idx].set_xlabel("Time (us)")
        axs[idx].set_ylabel("Modulation Value")
        axs[idx].legend()
        axs[idx].grid(True)

    plt.tight_layout()
    plt.show()
