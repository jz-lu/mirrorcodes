"""
analysis.py

Code file for computing pseudothresholds of mirror codes in parallel,
using SLURM arrays.
"""
from mirror import MirrorCode
from benchmark import StabilizerCode, make_noise_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import multiprocessing
import os
import time


def make_sec_from_code(mirror_code, circuit_name, noise_model_name, p, num_rounds):
    """
    Make the syndrome extraction circuit from a specification of the code.
    Primarily used as a helper function for multiprocessing.
    """
    if circuit_name == 'superdense':
        return mirror_code.superdense(noise_model_name=noise_model_name,
                              p=p,
                              num_rounds=num_rounds
                              )
    elif circuit_name == 'css':
        return mirror_code.fully_ft_for_css(noise_model_name=noise_model_name,
                              p=p,
                              num_rounds=num_rounds
                              )
    elif circuit_name == 'pheno':
        return mirror_code.generic_sec(p_depol=p, 
                                       p_meas=p, 
                                       num_rounds=num_rounds)
    else:
        raise ValueError(f"Circuit name '{circuit_name}' not yet implemented.")



def main(args):
    """
    Reads in the mirror code input file and calls the benchmarker on the specified work thread.
    """
    file_path = args.filepath
    assert os.path.exists(file_path), f"{file_path} does not exist"
    code_list = None

    UPPER_EXPONENT = args.upper
    LOWER_EXPONENT = args.lower
    NUM_POINTS = args.npts
    SEC_CHOICE = args.circuit
    NOISE_MODEL = args.model
    NUM_SHOTS = args.shots
    RUN = args.run
    PHYS_IDENTIFIER = f"u{UPPER_EXPONENT}l{LOWER_EXPONENT}N{NUM_POINTS}"
    LOG_IDENTIFIER = f"{RUN}_{NOISE_MODEL}_{SEC_CHOICE}_u{UPPER_EXPONENT}l{LOWER_EXPONENT}N{NUM_POINTS}s{NUM_SHOTS}"

    with open(file_path, 'rb') as f:
        code_list = pickle.load(f)

    G, A, B, is_CSS, k, d_estimate, goodness = code_list[RUN]
    code = MirrorCode(
        group=G,
        z0=A,
        x0=B
    )
    n = np.prod(G)
    w = len(A) + len(B)
    ROUND_CHOICES = [d_estimate - 6, d_estimate - 3, d_estimate, d_estimate + 3]
    PS = np.logspace(LOWER_EXPONENT, UPPER_EXPONENT, NUM_POINTS)


    print("Finding stabilizers...")
    stabilizers = code.get_stim_tableau()
    print("Done.")

    benchmarker = StabilizerCode(stabilizers, verbose=False, name=f"[[{n}, {k}, {d_estimate}, {w}]]")
    params = [(code, SEC_CHOICE, NOISE_MODEL, p, nrd) for nrd in ROUND_CHOICES for p in PS]

    circuits = None
    print("Building syndrome extraction circuits...")
    start = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        circuits = pool.starmap(make_sec_from_code, params)
    end = time.time()
    print(f"Done. Took {(end-start):.4f} seconds, using {multiprocessing.cpu_count()} cores.")

    print(f"Benchmarking {NUM_SHOTS} shots...")
    start = time.time()
    phys_rate, log_rate = benchmarker.parallel_benchmark(ps=PS,
                               secs=circuits,
                               rounds_choices=ROUND_CHOICES,
                               num_shots=NUM_SHOTS,
                               plot=False
                               )
    end = time.time()
    print(f"Done. Took {(end-start):.4f} seconds, using {multiprocessing.cpu_count()} cores.")

    if RUN == 0:
        np.save(f"phys_rate_{PHYS_IDENTIFIER}.npy", phys_rate)
    np.save(f"log_rate_{LOG_IDENTIFIER}.npy", log_rate)
    print("Saved results to file.")
    
    return 0




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mirror code benchmarker')
    parser.add_argument('--run', '-r', type=int, help='Thread index', default=0)
    parser.add_argument('--filepath', '-p', type=str, help='Mirror code pickle file list', default='./codes.pkl')
    parser.add_argument("--circuit", '-c' type=str, help="Choice of syndrome extraction circuit", \
                        choices=['superdense', 'bare', 'barely_dressed', 'css', 'pheno'], default='superdense')
    parser.add_argument("--model", '-m' type=str, help="Noise model name", default='SI1000')
    parser.add_argument("--shots", '-s' type=int, help="Number of shots", default=100000)
    parser.add_argument("--upper", '-u' type=int, help="Noise rate exponent upper bound", default=-2)
    parser.add_argument("--lower", '-l' type=int, help="Noise rate exponent lower bound", default=-6)
    parser.add_argument("--npts", '-N' type=int, help="Number of points to do analysis on", default=20)
    args = parser.parse_args()
    main(args)



