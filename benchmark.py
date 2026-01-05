"""
`benchmark.py`
Code file with class to numerically study the performance of a general stabilizer code.
The code does not have to be CSS.
"""
from util import binary_rank, symp2Pauli, stimify_symplectic
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import stim
import multiprocessing
#import tesseract_decoder
#import tesseract_decoder.tesseract as tesseract
import time


#https://arxiv.org/pdf/2108.10457
def noise(p = 0.001, name = 'SD'):
    if name == 'SD':
        return {
            'p2': p,
            'p1': p,
            'p_init': p,
            'p_meas': p,
            'p_idle': p,
        }
    else:
        return {
            'p2': p,
            'p1': p / 10,
            'p_init': 2 * p,
            'p_meas': 5 * p,
            'p_idle': p / 10,
            'p_res_idle': 2 * p,
        }


def benchmark(sec, num_shots = 1000, verbose=False):
        """
        Benchmark the decoding performance of a code under a single set of parameters.

        Params:
            * sec (stim.Circuit): syndrome extraction circuit (SEC) for the stabilizer code.
              The noise model should be pre-incorporated into the SEC.
            * num_shots (int): number of trials to conduct.
        
        Returns:
            * results (dict): 'num_errors': number of shots with errors
                              'num_shots': number of shots total
                              'time_seconds': time it took to decode, in seconds
        """
        # Prepare tesseract decoder
        dem = sec.detector_error_model()
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=10000,
            no_revisit_dets=True,
            verbose=False,
            det_orders=tesseract_decoder.utils.build_det_orders(
                dem, num_det_orders=1,
                method=tesseract_decoder.utils.DetOrder.DetIndex,
                seed=137),
        )
        tesseract_dec = tesseract_config.compile_decoder()

        # Sample noise
        sampler = sec.compile_detector_sampler()
        dets, obs = sampler.sample(num_shots, separate_observables=True)
        
        # Run decoder on the noise
        num_errors = 0
        start_time = time.time()
        obs_predicted = tesseract_dec.decode_batch(dets)
        end_time = time.time()
        num_errors = np.sum(np.any(obs_predicted != obs, axis=1))

        results = {
            'num_errors': num_errors,
            'num_shots': len(dets),
            'time_seconds': end_time - start_time,
        }

        if verbose:
            print("Tesseract Decoder Stats:")
            print(f"   Number of Errors / num_shots: {results['num_errors']} / {results['num_shots']}")
            print(f"   Time: {results['time_seconds']:.4f} s")

        return results


class StabilizerCode():
    """
    Class structure for a generic stabilizer code, specified by its stabilizer tableau.
    Uses stim and tesseract_decoder to benchmark the code.
    """
    def __init__(self, stabilizers, verbose=False, name=None):
        """
        Initialization of StabilizerCode by specification of the stabilizers.
        Let the code have n physical qubits and a tableau of r stabilizers.

        Params:
            * stabilizers (list[stim.PauliString]): list of stabilizers in stim Pauli string form.
            * name (str, Optional): name of the code
            * verbose (bool, Optional): if True, prints log of benchmarking
        
        Returns:
            None
        """
        self.stabilizers = stabilizers
        completed_tableau = stim.Tableau.from_stabilizers(
            stabilizers,
            allow_redundant=True,
            allow_underconstrained=True,
        )
        obs_indices = [
            k
            for k in range(len(completed_tableau))
            if completed_tableau.z_output(k) not in stabilizers
        ]
        self.logical_zs = [
            completed_tableau.z_output(k)
            for k in obs_indices
        ]
        self.logical_xs = [
            completed_tableau.x_output(k)
            for k in obs_indices
        ]
        self.logicals = self.logical_zs + self.logical_xs
        self.verbose = verbose
        self.name = name
        self.num_logicals = len(self.logical_zs)

        if self.verbose:
            print(f"Initialized StabilizerCode. Stabilizer tableau:")
            for stab in stabilizers:
                print(stab)
        return
    

    def parallel_benchmark(self, ps, secs, 
                           rounds_choices = [3, 5, 7], num_shots = 1000, phenomenological=False,
                           plot=False, save_as="./result.jpeg"):
        """
        Benchmark a bunch of error probabilities in parallel, using a syndrome extraction circuit (SEC) with circuit-level noise.
        On input a list of data, 1-qubit, and 2-qubit error probabilities (must be the same size), runs benchmarking on each triple
        of error probabilities in parallel and returns their empirical logical error rates.
        This is done using the sinter package, which splits each benchmarking task into a bunch of individual worker processes and 
        runs these asynchronously in parallel.
        
        Params:
            * sec_maker_fn (function): function which on input (p_datas, p1s, p2s, num_rounds) returns the appropriate SEC. This function
                                       should also accept an optional argument `phenomenological` which if true uses phenomenological noise.
            * rounds_choices (list:int): list of choices for number of rounds of syndrome extraction.
            * num_shots (float): number of trials.
            * phenomenological (bool): if True, the SEC will only put data errors (i.e. p_data before each round) and measurement errors.
        
        Returns:
            * Logical error rates (list:float) = num trials with logical error / num_shots
        """
        num_round_choices = len(rounds_choices)
        num_noise = len(secs) // num_round_choices

        args = [(sec, num_shots) for sec in secs]
            
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(benchmark, args)
        
        # Currently, `results` is a flattened array. Unflatten before returning.
        # Also, compute the logical error rate per cycle
        logical_error_rates = np.zeros((num_round_choices, num_noise))
        for i, num_rounds in enumerate(rounds_choices):
            for j in range(num_noise):
                logical_error_rates[i, j] = 1 - (1 - results[i*num_noise + j]['num_errors'] / num_shots)**(1/num_rounds)

        if plot:
            # Compute the physical error rate
            physical_error_rates = 1 - (1 - ps)**self.num_logicals

            # Make a plot of the logical versus the physical error rate
            colors = plt.cm.tab10(np.linspace(0, 1, num_round_choices))
            for i, nrd in enumerate(rounds_choices):
                plt.loglog(ps, logical_error_rates[i], color=colors[i], label=nrd, 
                         marker='o', markersize=6, markeredgewidth=2,
                         linestyle='-')
            plt.loglog(ps, physical_error_rates, color='gray', linestyle='--')
            plt.grid()
            plt.grid(which="minor", color="0.9")
            plt.legend(title='Rounds')
            
            plt.xlabel("Physical error rate")
            plt.ylabel("Logical error rate")
            plt.title(f"Logical error rate scaling of the{'' if self.name is None else ' ' + self.name} code ({num_shots} shots)")
            plt.tight_layout()
            plt.savefig(save_as)
        
        return logical_error_rates

    def pseudothreshold(self, phenomenological=False, round_choices=[3, 5, 7]):
        """
        Compute the pseudothreshold of the code, using circuit-level noise, for each number of rounds in `round_choices`.
        
        Params:
            * sec_maker_fn (function): function which on input (p_datas, p1s, p2s, num_rounds) returns the appropriate SEC.
            * round_choices (list:int): list of choices for number of rounds of syndrome extraction.
            * num_shots (float): number of trials.
            * phenomenological (bool): if True, the SEC will only put data errors (i.e. p_data before each round) and measurement errors.
        
        Returns:
            * pseudothresholds: (listfloat): computed pseudothreshold of the code, one for each number of rounds in `round_choices`.
        """
        pass


