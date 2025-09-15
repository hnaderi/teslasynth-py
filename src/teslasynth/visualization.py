from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from teslasynth.core import Voice, SynthConfig, SynthesizedTrack


def draw_piano_roll(
    voice: Voice,
    ax: plt.Axes,
    config: SynthConfig,
):
    cmap = plt.colormaps["tab20"]
    min_note = min([n[0] for n in voice.notes]) - 1 if voice.notes else 0
    max_note = max([n[0] for n in voice.notes]) + 1 if voice.notes else 127
    for note in voice.notes:
        start_sec = note.start / config.sample_rate
        end_sec = note.end / config.sample_rate
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
    timeline = np.arange(track.num_samples) / config.sample_rate
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
