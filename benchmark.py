"""
`benchmark.py`
Code file with class to numerically study the performance of a general stabilizer code.
The code does not have to be CSS.
"""
from util import binary_rank, symp2Pauli, stimify_symplectic
import numpy as np
import stim
import tesseract_decoder
import tesseract_decoder.tesseract as tesseract
import time


class StabilizerCode():
    """
    Class structure for a generic stabilizer code, specified by its stabilizer tableau.
    Uses stim and tesseract_decoder to benchmark the code.
    """
    def __init__(self, stabilizers):
        """
        Initialization of StabilizerCode by specification of the stabilizers.
        Let the code have n physical qubits and a tableau of r stabilizers.

        Params:
            * stabilizers (numpy.ndarray): r x 2n matrix whose rows are stabilizers in [Z | X] symplectic form.
        
        Returns:
            None
        """
        self.stabilizers = stimify_symplectic(stabilizers)
        completed_tableau = stim.Tableau.from_stabilizers(
            self.stabilizers,
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
        return
    
    def benchmark(self, sec, num_rounds = 3, num_shots = 1000):
        """
        Benchmark the decoding performance of a code under a single set of parameters.

        Params:
            * sec (stim.Circuit): syndrome extraction circuit (SEC) for the stabilizer code.
              The noise model should be pre-incorporated into the SEC.
            * num_rounds (int): number of rounds of syndrome extraction.
            * num_shots (int): number of trials to conduct.
        
        Returns:
            * Logical error rate (float) = num trials with logical error / num_shots
        """
        pass

    def parallel_benchmark(self, p_datas, p1s, p2s, sec_maker_fn, num_rounds = 3, num_shpts = 1000):
        """
        Benchmark a bunch of error probabilities in parallel.
        
        Params:
            * 
        
        Returns:
            * 
        """
        pass

    def pseudothreshold(self):
        pass



