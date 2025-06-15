"""
`filter.py`

Main wrapper for iteratively filtering for good codes
that we search through in a clever version of brute force in `search.py`.
The filtering process will be dependent on a fixed n and is a 3-step procedure.

Stage 1) Find all inequivalent codes for a given n. (Equivalence is defined by a series of automorphisms).
         This is the "canonical set". This is extremely fast for each code, but there are a huge number of 
         codes to check.
Stage 2) Find all codes which passed stage 1 and whose rate is above a threshold. This is quite fast.
Stage 3) Find all codes which passed stage 2 and whose distance is good. Here, good depends on whether
         we can calculate the distance in a reasonable time. If we can, then the distance * rate
         should be comparable to n. If we cannot, then that's a good sign, and we should keep the code
         for further investigation.
Stage 4) TBD. For good enough codes, we want to be even more fine grained and see decoding, geometric
         properties, etc.

The multistage process is disconnected in the sense that you specify a stage k. 
At the end of each stage, a file is saved containing all the codes which passed
that stage. This code will then assume you have already went through all stages 
up to and including k-1. It will import the file from the previous stage and 
process those codes. Therefore, at each stage the next file saved is substantially
smaller than the previous file saved.
"""
import argparse
import numpy as np
import pickle
from constants import get_filename, \
                      RATE_THRESHOLD, DISTANCE_THRESHOLD, \
                      DISTANCE_RATE_THRESHOLD

from util import stimify_symplectic, binary_rank
from helix import find_stabilizers
from search import process_codes
from distance import distance

def stage1(n : int):
    """
    Stage 1 filtering. 

    Params:
        * n (int): number of qubits.

    Returns:
        * list of helix codes in (group, Z_0, X_0) form which pass stage 1.
    """
    return find_all_codes(n)


def stage2(n : int, codes : list):
    """
    Stage 2 filtering. 

    Params:
        * n (int): number of qubits.
        * codes (list): list of helix codes in (group, Z_0, X_0) form which passed stage 1.

    Returns:
        * list of helix codes in (group, Z_0, X_0) form which pass stage 2.
    """
    passing_codes = []
    for code in codes:
        tableau = find_stabilizers(code)
        r = binary_rank(tableau)
        k = n - r
        rate = k/n
        if rate >= RATE_THRESHOLD:
            passing_codes.append(code)
            print(f"Added [[{n}, {k}]] code of rate {rate} :{code}")
    return passing_codes


def stage3(n : int, codes : list):
    """
    Stage 3 filtering. 

    Params:
        * n (int): number of qubits.
        * codes (list): list of helix codes in (group, Z_0, X_0) form which passed stage 2.

    Returns:
        * list of helix codes in (group, Z_0, X_0) form which pass stage 3.
    """
    pass


def stage4(n : int, codes : list):
    # TODO
    raise Exception("Stage 4 has not been implemented yet.")


def main(args):
    in_directory = args.input
    out_directory = args.output
    if out_directory is None:
        out_directory = in_directory
    stage = args.stage
    n = args.size
    out_data = None

    if stage == 1:
        out_data = stage1(n)
    else:
        in_file = f"{in_directory}/{get_filename(stage, n)}"
        codes = None
        with open(in_file, "rb") as f:
            codes = pickle.load(f)
        if stage == 2:
            out_data = stage2(n, codes)
        elif stage == 3:
            out_data = stage3(n, codes)
        elif stage == 4:
            out_data = stage4(n, codes)
    
    out_file = f"{out_directory}/{get_filename(stage, n)}"
    with open(out_file, "wb") as f:
        pickle.dump(out_data, f)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter for good helix codes"
    )

    parser.add_argument(
        "--size", "-n",
        type=int,
        required=True,
        help="Number of qubits"
    )

    parser.add_argument(
        "--stage", "-s",
        type=int,
        required=True,
        choices=[1, 2, 3, 4]
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default='.',
        help="Location of input files (default ./)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Where to write output files (default the same as input directory)"
    )

    args = parser.parse_args()
    main(args)
