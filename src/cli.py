#!/usr/bin/env python3

import mido
import sys
import traceback
import argparse
from teslasynth import visualization
from teslasynth.core import Voice, Synth, SynthConfig, BasicInstrument


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
            visualization.visualize(
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
