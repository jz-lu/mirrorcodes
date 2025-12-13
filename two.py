import search
import time
import argparse

def twos(args):
    arg = args.index
    arr = [
            [64],
            [2, 32],
            [2, 2, 2, 2, 4],
            [2, 2, 2, 2, 2, 2]][arg]
    search._subgroup_codes_and_bins(3, 3, arr, time.monotonic())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select powers of two"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        choices=[0, 1, 2, 3]
    )
    args = parser.parse_args()
    twos(args)
