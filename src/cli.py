#!/usr/bin/env python3

import sys
import traceback
import argparse
from teslasynth.visualization import visualize
from teslasynth.core import (
    TeslaSynth,
    SynthConfig,
    SamplingConfig,
    Limits,
)


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
            limits=Limits(
                max_on_time=400,
                min_on_time=10,
                min_deadtime=200,
                max_duty=5,
                max_notes=4,
            )
        )
        sampling = SamplingConfig(rate=args.sample_rate)

        midi_file_path = args.file
        channel = args.channel

        synth = TeslaSynth(config, sampling)
        result = synth.synthesize(midi_file_path, channel)
        track = result.make_track()
        if args.output:
            track.save_wav_file(args.output)

        track.print_statistics()
        visualize(result, track, channel, sampling.rate)

    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
