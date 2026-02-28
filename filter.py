"""
`filter.py`

Main wrapper for iteratively filtering for good codes
that we search through in a clever version of brute force in `search.py`.
The filtering process will be dependent on a fixed n and is a 3-step procedure.

Stage 1) Find all inequivalent codes for a given n. (Equivalence is defined by a series of automorphisms).
         This is the "canonical set". This is extremely fast for each code, but there are a huge number of 
         codes to check. Check the rate (n - rank of matrix) and discard if it zero.
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
import multiprocessing as mp
import numpy as np
import pickle

from constants import get_filename, \
                      RATE_THRESHOLD, DISTANCE_THRESHOLD, \
                      DISTANCE_RATE_THRESHOLD
from mirror import MirrorCode
from search import find_all_codes
from non_abelian import find_all_non_abelian_codes

def stage1(n:int, Z_wt:int, X_wt:int, min_k:int = 3, abelian:bool = True):
    """
    Stage 1 filtering. 

    Params:
        * n (int): number of qubits.
        * Z_wt (int): number of elements of Z_0.
        * X_wt (int): number of elements of X_0.
        * min_k (int, optional): Whether to filter stage 1 to not include codes with k < min_k.

    Returns:
        * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 1,
    """
    if abelian:
        return find_all_codes(n, Z_wt, X_wt, min_k = min_k)
    return find_all_non_abelian_codes(n, Z_wt, X_wt, min_k=min_k)


def stage2(n:int, codes:list, verbose:bool = False, abelian:bool = True):
    """
    Stage 2 filtering. 

    Params:
        * n (int): number of qubits.
        * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
          which passed stage 1.

    Returns:
        * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 2.
    """
    passing_codes = []
    for code_data in codes:
        group, z0, x0, _, k = code_data
        rate = k / n
        if rate >= RATE_THRESHOLD:
            passing_codes.append(code_data)
            if verbose:
                print(f"Added [[{n}, {k}]] code of rate {round(rate, 4)}")
    return passing_codes


def worker(queue, code, estimate):
    """
    Worker method for slow operation.

    Params:
        * queue: something for timeout thread
        * code (tuple): code whose distance we want to find
    """
    try:
        if estimate:
            result = code.get_d_est()
        else:
            result = code.get_d()
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", e))


def stage3(n:int, codes:list, t:int = 3, verbose:bool = False, estimate:bool = False, abelian:bool = True):
    """
    Stage 3 filtering. 

    Params:
        * n (int): number of qubits.
        * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
          which passed stage 2.
        * t (int): how many seconds you are willing to spend on the distance
          calculation.

    Returns:
        * list of helix codes in (group, Z_0, X_0, k, d, k*d/n) form which pass
          stage 3. (d -> -1, k*d/n -> -1 if distance failed to calculate in time t)
    """
    passing_codes = []
    seen = set()
    for code_data in codes:
        if abelian:
            group, z0, x0, is_css, k = code_data
            code = MirrorCode(group, z0, x0, n = n, k = k, is_css = is_css, abelian=True)
        else:
            group, z0, x0, symmetric, k = code_data
            code = MirrorCode(group, z0, x0, n = n, k = k, symmetric=symmetric, abelian=False)
        
        d = -1
        
        queue = mp.Queue()
        process = mp.Process(target = worker, args = (queue, code, estimate))
        process.start()
        process.join(t)
        if process.is_alive():
            process.terminate()
            process.join()
            if verbose:
                if abelian:
                    print(f"Distance calculation timed out at {t}s for {'' if is_css else 'non-'}CSS [[{n}, {k}]] code\ngroup =\n{group}\nz0 =\n{z0}\nx0 =\n{x0}")
                else:
                    print(f"Distance calculation timed out at {t}s for {'' if symmetric else 'a'}symmetric non-abelian [[{n}, {k}]] code\ngroup =\n{group}\nz0 =\n{z0}\nx0 =\n{x0}")
        else:
            status, payload = queue.get()
            if status != 'ok':
                raise payload
            d = payload
        
        goodness = k * d / n
        goodness_str = f" (GR = {round(goodness, 4)})" if d > 0 else ""
        if d == -1 or (d >= DISTANCE_THRESHOLD - 0.5
                       and (goodness >= DISTANCE_RATE_THRESHOLD
                            or (len(z0) == 2 and len(x0) == 2))):
            if abelian:
                if verbose and (k, d, is_css) not in seen:
                    # Only print codes with genuinely new parameters.
                    print(f"[[{n}, {k}, {d}]]{goodness_str} {'' if is_css else 'non-'}CSS code found")
                    if goodness >= 0.9:
                        print(f"*******  Someone has a bright future! *******")
            else:
                if verbose and (k, d, symmetric) not in seen:
                    # Only print codes with genuinely new parameters.
                    print(f"[[{n}, {k}, {d}]]{goodness_str} {'' if symmetric else 'a'}symmetric non-abelian code found")
                    if goodness >= 0.9:
                        print(f"*******  Someone has a bright future! *******")

            seen.add((k, d, is_css if abelian else symmetric))
            passing_codes.append((group, z0, x0, is_css if abelian else symmetric, k, d,
                                  -1 if d == -1 else round(k * d / n, 5)))
        # else:
        #     if verbose:
        #         print(f"[[{n}, {k}, {d}]]{goodness} code is BAD")
        
    return passing_codes

def stage4(n:int, codes:list, abelian = True):
    # TODO
    raise Exception("Stage 4 has not been implemented yet.")


def main(args):
    VERBOSE = args.verbose
    SAVE_DATA = not args.nosave
    in_directory = args.input
    out_directory = args.output
    est = args.estimate
    if out_directory is None:
        out_directory = in_directory
    Z_wt = args.Zweight
    if Z_wt is None:
        Z_wt = 3
    X_wt = args.Xweight
    if X_wt is None:
        X_wt = 3
    time = args.time
    if time is None:
        time = 3
    mink = args.mink
    if mink is None:
        mink = 3
    if est is None:
        est = False
    stages = args.stages
    n = args.size
    r = args.range
    width = args.width
    abelian = not args.groupsnonabelian
    if width is None:
        width = 100
    print(f"Running: n = {n}")
    out_data = None

    if args.fullsend:
        print("[Fullsend] Starting stage 1")
        out_data = stage1(n, Z_wt = Z_wt, X_wt = X_wt, min_k = mink, abelian = abelian)
        print(f"Filtered to {len(out_data)} codes")
        print("[Fullsend] Starting stage 2")
        out_data = stage2(n, out_data, abelian = abelian)
        print(f"Filtered to {len(out_data)} codes")
        print("[Fullsend] Starting stage 3")
        out_data = stage3(n, out_data, t = time, verbose = VERBOSE, abelian = abelian)
        print(f"Filtered to {len(out_data)} codes")

        if SAVE_DATA:
            out_file = f"{out_directory}/{get_filename(3, n, abelian = abelian)}"
            with open(out_file, "wb") as f:
                pickle.dump(out_data, f)

    else:
        for stage in str(stages):
            stage = int(stage)
            if stage == 1:
                out_data = stage1(n, Z_wt = Z_wt, X_wt = X_wt, min_k = mink, abelian = abelian)
            else:
                in_file = f"{in_directory}/{get_filename(stage - 1, n, abelian = abelian)}"
                in_data = None
                with open(in_file, "rb") as f:
                    in_data = pickle.load(f)
    
                if stage == 2:
                    out_data = stage2(n, in_data, verbose = VERBOSE, abelian = abelian)
                elif stage == 3:
                    if r is not None:
                        start = min(width * r, len(in_data))
                        end = min(width * (r + 1), len(in_data))
                        in_data = in_data[start : end]
                    out_data = stage3(n, in_data, t = time, verbose = VERBOSE, estimate=est, abelian = abelian)
                elif stage == 4:
                    out_data = stage4(n, in_data, abelian = abelian)
            
            if SAVE_DATA:
                out_file = f"{out_directory}/{get_filename(stage, n, r, abelian = abelian)}"
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
        "--mink", "-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--Zweight", "-z",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--Xweight", "-x",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--stages", "-s",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 12, 13, 14, 23, 24, 123, 124, 134, 234, 1234]
    )

    parser.add_argument(
        "--time", "-t",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--range", "-r",
        type=int,
    )

    parser.add_argument(
        "--width", "-w",
        type=int,
        default = 100,
    )

    parser.add_argument(
        "--fullsend", "-f",
        action="store_true",
        help="Run stages 1-3 all at once (don't do this for large n)"
    )

    parser.add_argument(
        "--groupsnonabelian", "-g"
        action="store_true",
        help="Search for non-abelian groups"
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

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include print statements"
    )

    parser.add_argument(
        "--estimate", "-e",
        action="store_true",
        help="Estimate distance instead of computing exactly. Will underreport by 0.5"
    )

    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Don't save files"
    )

    args = parser.parse_args()
    main(args)
