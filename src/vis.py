#!/usr/bin/env python3

import sys
import traceback
import argparse
from teslasynth.visualization import visualize_components
from teslasynth.core import Instruments


parser = argparse.ArgumentParser("Teslasynth instrument visualizer")
parser.add_argument("instrument", help="Instrument number", type=int)


def main():
    try:
        args = parser.parse_args()
        instrument_number = args.instrument
        if len(Instruments) > instrument_number:
            instrument = Instruments[instrument_number]
        else:
            print("No such instrument!")
            sys.exit(1)

        visualize_components(instrument)

    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
