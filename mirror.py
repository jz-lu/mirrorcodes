"""
`mirror.py`
Code file to construct a mirror code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} x ... x Z_{aN}
    * Z0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to Z0
    * X0 [{u1, u2, ..., uQ}]: a set of N-tuples specifying elements of the group which belong to X0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the mirror code.
The check weight is exactly |Z0| + |X0| - |Z0 ^ X0| <= |Z0| + |X0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form---there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
"""
import itertools as it
import numpy as np
from util import find_isos, find_strides, shift_X
from util import binary_rank, symp2Pauli, stimify_symplectic
from test_cases import get_stabilizers
from distance import distance, distance_estimate, make_code
import stim
# import tesseract_decoder
# import tesseract_decoder.tesseract as tesseract
import time


"""
DESCRIPTION OUT OF DATE. CANONICALIZATION AND SEARCH IS NOW ENTIRELY IN SEARCH.PY
Part I: Mapping the mirror code parameterization to a canonical stabilizer tableau.

We specify a mirror code by giving an abelian group description (a tuple of prime powers describing a 
direct product of cyclic groups) and two subsets, the Z subset and the X subset. This fully describes a mirror code.
The functions in Part I (a) compute the stabilizer tableau of a mirror code description (`find_stabilizers`), 
(b) check if it is CSS (`css_flips`), and (c) check if the code is "canonical". 
By canonical, we mean that it is the unique representative under a set of operations which preserve the code.
These include swapping the Z and X subsets, permuting the elements of the subsets, automorphisms of the group, etc.
By having a canonical representation, we save a great deal of space during numerical search of the code.
"""


def build_set(group, a, b):
    """
    Utility function. Given two sets of arrays, finds all possible differences
    between an element of one array and an element of the other.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * a (np.ndarray): 2D Array. The first array of the two whose differences
          we will be finding. It is a list of all the elements of the group in the
          first array. Number of columns should match length of group.
        * b (np.ndarray): 2D Array. The second array of the two whose differences
          we will be finding. It is a list of all the elements of the group in the
          first array. Number of columns should match length of group.
    
    Returns:
        * 2D numpy array containing all the elements of the group that are the
          difference between something in a and something in b.
    """
    s = []
    group = np.array(group)
    a = np.array(a)
    b = np.array(b)
    for i in a:
        for j in b:
            s.append(np.mod(j - i, group))
    return np.unique(s, axis=0)


def css_flips(group, z0, x0):
    """
    Compute the stabilizer tableau of a code. Tableau is returned in symplectic
    form with the convention [Z | X]. Automatically checks if the code is CSS and
    converts the tableau to CSS form if so.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the Z weight.
          Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the X weight.
          Also accepts tuples at any level.

    Returns:
        * Tuple containing two terms. The first is a bool which is True iff the
          code passed was CSS. The second is a numpy array of qubits (in tuple /
          array form) which need to be hadamarded in order to make the code CSS.
          The second argument is [] if the first is False.
    """
    n = np.prod(group)
    same_diffs = np.unique(np.vstack([build_set(group, z0, z0),
                                      build_set(group, x0, x0)]), axis=0)
    zx = build_set(group, z0, x0)

    flips = np.zeros((1, len(group)), dtype=int)
    for g in same_diffs:
        gen_g = [g]
        cur = g
        while np.max(cur) > 0:
            cur = np.mod(cur + g, group)
            gen_g.append(cur)
        cur_flips = flips.copy()
        for i in cur_flips:
            flips = np.append(flips, np.mod(i + gen_g, group), axis=0)
        flips = np.unique(flips, axis=0)
        if len(flips) == n:
            return False, []

    bad = set()
    strides = find_strides(group)
    for i in it.product(*[range(0, a, 2 - a % 2) for a in group]):
        bad.update(np.mod(i + zx, group) @ strides)
    for i in flips @ strides:
        if i in bad:
            return False, []
    return True, flips


def find_stabilizers(group, z0, x0):
    """
    Compute the stabilizer tableau of a code. Tableau is returned in symplectic
    form with the convention [Z | X]. Automatically checks if the code is CSS and
    converts the tableau to CSS form if so.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the Z weight.
          Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the X weight.
          Also accepts tuples at any level.
    
    Returns:
        * 2D numpy array in symplectic form (Z|X) with the stabilizers of the code.
          Returns n stabilizers, so contains linearly dependent checks. Should
          automatically be in CSS form if code is CSS.
        * boolean on whether or not the code is CSS
    """
    group = np.array(group, np.int64)
    z0 = np.array(z0, np.int64)
    x0 = np.array(x0, np.int64)
    n = int(np.prod(group))
    strides = find_strides(group)
    stabilizers = np.zeros((n, 2 * n), dtype=np.uint8)

    can_flip, flips = css_flips(group, z0, x0)

    for i, g in enumerate(it.product(*[range(a) for a in group])):
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1

    # if can_flip:
    #     for g in flips:
    #         index = g @ strides
    #         stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]]
    return stabilizers, can_flip


"""
Part II: stim circuit manipulation

Here we have some custom functions which modify a stim circuit in place for numerical analysis purposes.
"""

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

def pauli_to_observable_include_target(pauli: stim.PauliString) -> list[stim.GateTarget]:
    obs_pauli_targets = []
    for i in range(len(pauli)):
        if pauli[i] != 0:
            obs_pauli_targets.append(stim.target_pauli(i, pauli[i]))
    return obs_pauli_targets

def append_observable_includes_for_paulis(circuit: stim.Circuit, paulis: list[stim.PauliString]) -> None:
    for i, obs in enumerate(paulis):
        circuit.append(
            "OBSERVABLE_INCLUDE",
            targets=pauli_to_observable_include_target(pauli=obs),
            arg=i
        )
    
def append_noisy_gate(circuit : stim.Circuit, gate : str, locality : int, qubits : list, p : float, *p2):
    """
    Inserts a noisy gate into the circuit by adding the desired gate, preceded by a depolarizing error.
    Only supports 1- and 2-qubit gates. 
    For a t-qubit gate, t-qubit depolarizing noise will be added after the gate.

    Params:
        * circuit (stim.Circuit): stim circuit to which the gate is to be added.
        * gate (str): string description of the gate, e.g. 'RX' or 'CNOT'.
        * locality (str): number of qubits the gate acts on. Should be in {1, 2}.
        * qubits (list): the qubits on the circuit on which the gate is applied. 
            For 1-qubit gate, will apply on every qubit in list. For 2-qubit gate, must be only 2 qubits.
        * p (float): noise rate parameter, should be in [0, 1].
        * after (bool): put noise after instead of before the gate (only use when operation is a RESET).
    
    Returns:
        * None (modifies the circuit in place).
    """
    assert locality in [1, 2], f"Only 1- and 2-qubit gates supported, but got {locality}-qubit gate request"
    measure = gate in ['MZ', 'MX', 'MRX', 'MRZ']
    if measure:
        circuit.append(f"DEPOLARIZE{locality}", qubits, p)
    circuit.append(gate, qubits)
    if not measure:
        circuit.append(f"DEPOLARIZE{locality}", qubits, p)
    if gate in ['MRX', 'MRZ']:
        circuit.append(f"DEPOLARIZE{locality}", qubits, p2)
    return

"""
Part III: Tesseract functions
"""


def print_decoder_results(results):
  print("Tesseract Decoder Stats:")
  print(f"   Number of Errors / num_shots: {results['num_errors']} / {results['num_shots']}")
  print(f"   Time: {results['time_seconds']:.4f} s")


def run_tesseract_decoder(decoder, dets, obs):
  # Run and time the Tesseract decoder
  num_errors = 0
  start_time = time.time()
  obs_predicted = decoder.decode_batch(dets)
  num_errors = np.sum(np.any(obs_predicted != obs, axis=1))
  end_time = time.time()

  return {
      'num_errors': num_errors,
      'num_shots': len(dets),
      'time_seconds': end_time - start_time,
  }



"""
Main class: MirrorCode
This is the class which holds all the basic functionality of a mirror code, specified by the (group, X_set, Z_set) description.
"""
class MirrorCode():
    """
    Class structure for a mirror code, specified by an abelian group, 
    a Z-type generator, and a X-type generator.

    Params:
        * group: abelian group in structure form, i.e. list of powers of primes.
        * z0: list of coordinates in structure form for Z-type generator.
        * x0: list of coordinates in structure form for X-type generator.
        * n (Optional): number of qubits = product of structure form orders.
        * k (Optional): number of logical qubits.
        * d (Optional): distance of the code.
        * is_css (Optional): whether or not code is CSS.
    
    The optional variables can be specified if they are precomputed. If they are
    not specified, they are computed by the class functions the first time they are queried.
    """
    def __init__(self, group, z0, x0, n=None, k=None, d=None, is_css=None, d_est=None):
        self.group = group
        self.z0 = np.array(z0, dtype = int)
        self.x0 = np.array(x0, dtype = int)
        self.wx = len(x0)
        self.wz = len(z0)
        self.stabilizers = None
        self.stim_tableau = None
        self.CSS = is_css

        self.n = int(n) if n is not None else None
        self.k = k
        self.d = d
        self.d_est = d_est

    def get_stabilizers(self):
        if self.stabilizers is None:
            self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
        return self.stabilizers
    
    def get_stim_tableau(self):
        if self.stim_tableau is None:
            self.stim_tableau = stimify_symplectic(self.get_stabilizers())
        return self.stim_tableau
    
    def get_stim_logical_paulis(self):
        """
        Get the logical Paulis in stim Pauli form, concatenated together in one list.
        """
        z, x = make_code(self.get_stim_tableau())[1:]
        self.stim_logical_paulis = [z, x]
        return z + x
    
    def get_n(self):
        if self.n is None:
            self.n = int(np.prod(self.group))
        return self.n
    
    def get_k(self):
        if self.k is None:
            self.k = self.get_n() - binary_rank(self.get_stabilizers())
        return self.k
    
    def get_d(self, verbose=False):
        if self.d is None:
            tableau = self.get_stim_tableau()
            assert self.CSS is not None, f"You screwed up somewhere?"
            self.d = distance(tableau, self.CSS, verbose=verbose)
        return self.d
    
    def get_d_est(self):
        if self.d_est is None:
            if self.d is None:
                tableau = self.get_stim_tableau()
                self.d_est = distance_estimate(tableau)
            else:
                self.d_est = self.d
        return self.d_est
    
    def is_CSS(self):
        if self.CSS is None:
            self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
        return self.CSS

    def get_rate(self):
        return self.get_k() / self.get_n()
    
    def get_rel_dist(self):
        return self.get_d() / self.get_n()
    
    def syndrome_extraction_circuit(self, p_data, p1, p2, num_rounds=3, option=0) -> stim.Circuit:
        """
        Make a syndrome extraction circuit corresponding to the mirror code
        instantiated in this class.
        There is an `option` command which lets you choose from a list of 
        circuits to return (with different fault tolerance capabilities).
        Currently we have an optimized implementation "made by hand" which only 
        supports weight 6 mirror codes.

        Options:
            0. Not fault-equivalent to the cat-state syndrome extraction circuit (SEC), but fewer 2-qubit gates
            1. Fault-equivalent to the cat-state SEC.

        Params:
            * p_data (float): 1-qubit data error probability parameter in [0, 3/4] (error accrued pre-extraction).
            * p1 (float): 1-qubit error probability parameter in [0, 3/4].
            * p2 (float): 2-qubit error probability parameter in [0, 15/16].
            * num_rounds (int): number of rounds of syndrome extraction. 
            * option (int): menu of options for which circuit to output
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # assert self.wz == 3 and self.wx == 3, f"Idk how to make short circuits otherwise"
        assert 0 <= p1 <= 3/4, f"1-qubit error probability {p1} must be within [0, 3/4]"
        assert 0 <= p1 <= 15/16, f"2-qubit error probability {p2} must be within [0, 15/16]"

        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        ANCILLA_PER_STAB = 5
        if option == 1:
            ANCILLA_PER_STAB = 6
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()

        # Do some perfect measurements before we get into actual extraction for detection purposes.
        # append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)

        # Implement depolarizing noise modeling errors on data qubits accrued while idling in memory (pre-syndrome extraction)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)
        
        if option == 0:
            # Initialize ancillary system
            for ancilla_block_qubit in range(n, (ANCILLA_PER_STAB+1)*n, ANCILLA_PER_STAB):
                # Initialize first 2 qubits to |+>
                append_noisy_gate(sec, "RX", 1, [ancilla_block_qubit, ancilla_block_qubit + 1], p1)

                # Initialize last 3 qubits to |0>
                append_noisy_gate(sec, "RZ", 1, [ancilla_block_qubit + 2, ancilla_block_qubit + 3, ancilla_block_qubit + 4], p1)

                # Add a CNOT to make a Bell pair
                append_noisy_gate(sec, "CNOT", 2, [ancilla_block_qubit, ancilla_block_qubit + 2], p2)
            
            for round_idx in range(num_rounds):
                # Do the syndrome extraction "in parallel" for each stabilizer
                # in the obvious way (not depth-optimized, but is OK for zero idle noise)
                for j, stab in enumerate(stabilizers):
                    Z_part, X_part = stab[:n], stab[n:]

                    # X part first
                    X_supp = [i for i in range(n) if X_part[i] != 0]
                    Z_supp = [i for i in range(n) if Z_part[i] != 0]
                    assert len(X_supp) == len(Z_supp) == 3

                    for op in range(6):
                        if op <= 2:
                            # CNOT between X check and ancilla
                            append_noisy_gate(sec, "CNOT", 2, [X_supp[op], (j+1)*n + op], p2)
                            if op == 1:
                                # Do a CNOT between ancillas 0 and 3
                                append_noisy_gate(sec, "CNOT", 2, [(j+1)*n, (j+1)*n + 3], p2)
                            elif op == 2:
                                # Do a CNOT between ancillas 1 and 4
                                append_noisy_gate(sec, "CNOT", 2, [(j+1)*n + 1, (j+1)*n + 4], p2)
                        elif op <= 5:
                            # CZ between Z check and anncilla
                            if op == 3:
                                # Also add 2 CNOT gates between the ancillas
                                append_noisy_gate(sec, "CZ", 2, [Z_supp[0], (j+1)*n], p2)
                                append_noisy_gate(sec, "CNOT", 2, [(j+1)*n + 1, (j+1)*n + 3], p2)
                                append_noisy_gate(sec, "CNOT", 2, [(j+1)*n + 2, (j+1)*n + 4], p2)
                            elif op == 4:
                                # Also add measurements of last 2 ancillas and detections for some ancillas, then reset
                                append_noisy_gate(sec, "CZ", 2, [Z_supp[1], (j+1)*n + 2], p2)
                                append_noisy_gate(sec, "MZ", 1, [(j+1)*n + 3, (j+1)*n + 4], p1)
                                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2)])
                            elif op == 5:
                                append_noisy_gate(sec, "CZ", 2, [Z_supp[2], (j+1)*n + 1], p2)
                                append_noisy_gate(sec, "CNOT", 2, [(j+1)*n, (j+1)*n + 2], p2)
                                # Reset the last 2 ancillas
                                if round_idx < num_rounds - 1:
                                    append_noisy_gate(sec, "RZ", 1, [(j+1)*n + 3], p1)
                                    append_noisy_gate(sec, "RX", 1, [(j+1)*n + 4], p1)
                
                # If this is the last round, measure the first 3 ancillas and call it a day.
                # If this is not the last round, also initialize a new Bell pair in parallel.
                for j in range(n):
                    append_noisy_gate(sec, "MRX", 1, [(j+1)*n, (j+1)*n + 1], p1)
                    if round_idx == 0:
                        sec.append("DETECTOR", targets=[stim.target_rec(-(3*n+j+2)), stim.target_rec(-1), stim.target_rec(-2)])
                    else:
                        sec.append("DETECTOR", targets=[stim.target_rec(-(4*n+2)), stim.target_rec(-(4*n+1)), stim.target_rec(-1), stim.target_rec(-2)])

                for j in range(n):
                    append_noisy_gate(sec, "MRZ", 1, [(j+1)*n + 2], p1)
                    sec.append("DETECTOR", targets=[stim.target_rec(-1)])

                if round_idx < num_rounds - 1: # more rounds to go
                    for j in range(n):
                        append_noisy_gate(sec, "CNOT", 2, [(j+1)*n + 3, (j+1)*n + 4], p2)

                        sec.append("SWAP", [(j+1)*n + 3, (j+1)*n]) #! this action should be noiseless
                        sec.append("SWAP", [(j+1)*n + 4, (j+1)*n + 2]) #! this action should be noiseless

        elif option == 1:
            #! TODO: this option is not ready! e.g. no noise implemented, ordering of stab/op is wrong, etc.
            for ancilla_block_qubit in range(n, (ANCILLA_PER_STAB+1)*n, ANCILLA_PER_STAB):
                # Initialize first 3 qubits to |+>
                sec.append("RX", [ancilla_block_qubit, ancilla_block_qubit + 1, ancilla_block_qubit + 2])

                # Initialize last 3 qubits to |0>
                sec.append("RZ", [ancilla_block_qubit + 3, ancilla_block_qubit + 4, ancilla_block_qubit + 5])

            for round_idx in range(num_rounds):
                # Do the syndrome extraction in parallel for each stabilizer
                for j, stab in enumerate(stabilizers):
                    Z_part, X_part = stab[:n], stab[n:]

                    # X part first
                    X_supp = [i for i in range(n) if X_part[i] != 0]
                    Z_supp = [i for i in range(n) if Z_part[i] != 0]
                    for op in range(9):
                        assert len(X_supp) == len(Z_supp) == 3
                        base = n + j * ANCILLA_PER_STAB
                        if op <= 2:
                            # CNOT between X check and ancilla
                            sec.append("CNOT", [base + op, X_supp[op]])
                            if op == 1:
                                # Do a CNOT between ancillas 0 and 3
                                sec.append("CNOT", [base, base + 3])
                            elif op == 2:
                                # Do a CNOT between ancillas 1 and 4
                                sec.append("CNOT", [base + 1, base + 3])
                                sec.append("CNOT", [base, base + 4])
                        elif op <= 5:
                            # CZ between Z check and ancilla
                            sec.append("CZ", [base + op - 3, Z_supp[op - 3]])
                            if Z_supp[op - 3] in X_supp:
                                sec.append("S", [base + op - 3])
                            if op == 3:
                                # Also add 2 CNOT gates between the ancillas
                                sec.append("CNOT", [base + 2, base + 4])
                                sec.append("CNOT", [base + 1, base + 5])
                                # Add measurement of ancilla 3
                                sec.append("MRZ", [base + 3]) # A
                            elif op == 4:
                                # Add some CNOTs and measurements
                                sec.append("CNOT", [base + 2, base + 5])
                                sec.append("MRZ", [base + 4]) # B
                            elif op == 5:
                                sec.append("CNOT", [base, base + 1])
                                sec.append("MRZ", [base + 5]) # C

                                # Detector for A + B + C == 0
                                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3)])

                        elif op == 6:
                            # Remaining measurements and detectors
                            sec.append("MZ", [base + 1]) # E
                            sec.append("RX", [base + 1]) # E reset

                            # Detector for A + E == 0
                            sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-4)]) 
                        elif op == 7:
                            sec.append("MRX", [base + 0]) # D
                        elif op == 8:
                            sec.append("MRX", [base + 2]) # F

                            if round_idx == 0:
                                # Detect D + F + (corresponding stabilizer measurement) == 0
                                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-(5*j+n+6))]) 
                            else:
                                # Detect D + F + (same but previous round) == 0
                                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-(6*n+1)), stim.target_rec(-(6*n+2))]) 
        
        else:
            raise ValueError(f"Option {option} is not a valid choice!")

        # append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec

    def new_sec(self, p_data, p1, p2, num_rounds=3) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        ANCILLA_PER_STAB = 3
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()

        # Do some perfect measurements before we get into actual extraction for detection purposes.
        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)

        # Implement depolarizing noise modeling errors on data qubits accrued while idling in memory (pre-syndrome extraction)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)

        for ancilla_block_qubit in range(n, (ANCILLA_PER_STAB+1)*n, ANCILLA_PER_STAB):
            # Initialize first 3 qubits to |+>
            sec.append("RX", [ancilla_block_qubit, ancilla_block_qubit + 1, ancilla_block_qubit + 2])


        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                Z_part, X_part = stab[:n], stab[n:]

                # X part first
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]
                for op in range(9):
                    assert len(X_supp) == len(Z_supp) == 3
                    base = n + j * ANCILLA_PER_STAB
                    if op <= 2:
                        # CNOT between X check and ancilla
                        sec.append("CNOT", [base + op, X_supp[op]])
                    elif op <= 5:
                        if op == 3:
                            sec.append("MPP", stim.PauliString(f"Z{base}*Z{base+1}")) # A
                            sec.append("MPP", stim.PauliString(f"Z{base}*Z{base+2}")) # B
                            sec.append("MPP", stim.PauliString(f"Z{base+1}*Z{base+2}")) # C
                        # CZ between Z check and ancilla
                        sec.append("CZ", [base + op - 3, Z_supp[op - 3]])
                        if Z_supp[op - 3] in X_supp:
                            sec.append("S", [base + op - 3])
                        if op == 5:
                            sec.append("CNOT", [base, base + 1])

                            # Detector for A + B + C == 0
                            sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3)])
                    elif op == 6:
                        # Remaining measurements and detectors
                        sec.append("MZ", [base + 1]) # E
                        sec.append("RX", [base + 1]) # E reset

                        # Detector for A + E == 0
                        sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-4)]) 
                    elif op == 7:
                        sec.append("MRX", [base + 0]) # D
                    elif op == 8:
                        sec.append("MRX", [base + 2]) # F

                        if round_idx == 0:
                            # Detect D + F + (corresponding stabilizer measurement) == 0
                            sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-(5*j+n+6))]) 
                        else:
                            # Detect D + F + (same but previous round) == 0
                            sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-(6*n+1)), stim.target_rec(-(6*n+2))]) 
        
        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec

    #!! ONLY THE NEXT THREE SECS ACTUALLY WORK RIGHT NOW
    def generic_sec(self, p_data, p1, p2, num_rounds):
        """
        A generic "fake"/unphysical syndrome extraction circuit which should work for any stabilizer tableau.
        """
        num_qubits = num_stabilizers = self.get_n()
        stabilizer_paulis = self.get_stim_tableau()
        all_logicals_paulis = self.get_stim_logical_paulis()

        circuit = stim.Circuit()

        append_observable_includes_for_paulis(circuit=circuit, paulis=all_logicals_paulis)
        circuit.append("MPP", stabilizer_paulis)
        circuit.append("DEPOLARIZE1", targets=list(range(num_qubits)), arg=p_data)

        for _ in range(num_rounds):
            circuit.append("MPP", stabilizer_paulis)

            for i in range(num_stabilizers):
                circuit.append(
                    "DETECTOR",
                    targets=[
                        stim.target_rec(i - 2 * num_stabilizers),
                        stim.target_rec(i - num_stabilizers)
                    ]
                )

        append_observable_includes_for_paulis(circuit=circuit, paulis=all_logicals_paulis)
        return circuit

    def bare_ancilla_sec(self, p_data, p1, p2, num_rounds=3) -> stim.Circuit:
        """
        Make a syndrome extraction circuit corresponding to the mirror code
        instantiated in this class.
        There is an `option` command which lets you choose from a list of 
        circuits to return (with different fault tolerance capabilities).
        Currently we have an optimized implementation "made by hand" which only 
        supports weight 6 mirror codes.

        Options:
            0. Not fault-equivalent to the cat-state syndrome extraction circuit (SEC), but fewer 2-qubit gates
            1. Fault-equivalent to the cat-state SEC.

        Params:
            * p_data (float): 1-qubit data error probability parameter in [0, 3/4] (error accrued pre-extraction).
            * p1 (float): 1-qubit error probability parameter in [0, 3/4].
            * p2 (float): 2-qubit error probability parameter in [0, 15/16].
            * num_rounds (int): number of rounds of syndrome extraction. 
            * option (int): menu of options for which circuit to output
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # assert self.wz == 3 and self.wx == 3, f"Idk how to make short circuits otherwise"
        assert 0 <= p1 <= 3/4, f"1-qubit error probability {p1} must be within [0, 3/4]"
        assert 0 <= p2 <= 15/16, f"2-qubit error probability {p2} must be within [0, 15/16]"

        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)
        
        for _ in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                Z_part, X_part = stab[:n], stab[n:]
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]
                # Y_supp = [i for i in range(n) if Z_part[i] != 0 and X_part[i] != 0]
                append_noisy_gate(sec, "RX", 1, [n+j], p1)
                for q in X_supp:
                    append_noisy_gate(sec, "CNOT", 2, [n + j, q], p2)
                for q in Z_supp:
                    append_noisy_gate(sec, "CZ", 2, [n + j, q], p2)
                    if q in X_supp:
                        append_noisy_gate(sec, "S", 1, [n+j], p1)
                # for q in Y_supp:
                #     append_noisy_gate(sec, "CY", 2, [n + j, q], p2)
                append_noisy_gate(sec, "MX", 1, [n + j], p1)
                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-(n + 1))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec
    
    def barely_dressed_ancilla_sec(self, p_data, p_meas, p1, p2, num_rounds=3, phenomenological=False) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = self.get_stim_tableau()
        all_logical_paulis = self.get_stim_logical_paulis()
        if phenomenological:
            p2 = 0
            p1 = 0

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        
        for round_idx in range(num_rounds):
            sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)

            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                base = n + 2*j
                Z_part, X_part = stab[:n], stab[n:]
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]

                sec.append("RX", [base])
                sec.append("RZ", [base+1])
                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)

                for q in X_supp:
                    append_noisy_gate(sec, "CNOT", 2, [base, q], p2)
                for q in Z_supp:
                    append_noisy_gate(sec, "CZ", 2, [base, q], p2)
                    if q in X_supp:
                        # raise LookupError("You found me :)")
                        append_noisy_gate(sec, "S", 1, [base+1], p1)

                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                sec.append("MRX", [base], p_meas)
                sec.append("MRZ", [base+1], p_meas)
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])
                if round_idx == 0:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(n + j + 2))])
                else:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(2*n + 2))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec

    def casually_dressed_ancilla_sec(self, p_data, p1, p2, num_rounds=3) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)
        
        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                base = n + 2*j
                Z_part, X_part = stab[:n], stab[n:]
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]

                append_noisy_gate(sec, "RX", 1, [base], p1)
                append_noisy_gate(sec, "RZ", 1, [base + 1], p1)
                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)

                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[0]], p2)
                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[1]], p2)
                sec.append("MPP", stim.PauliString(f"Z{base}*Z{base+1}"))
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])

                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[2]], p2)
                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[0]], p2)

                sec.append("MPP", stim.PauliString(f"Z{base}*Z{base+1}"))
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])

                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[1]], p2)
                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[2]], p2)
                for q in Z_supp:
                    if q in X_supp:
                        append_noisy_gate(sec, "S", 1, [base+1], p1)

                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                append_noisy_gate(sec, "MRX", 1, [base], p1)
                append_noisy_gate(sec, "MRZ", 1, [base + 1], p1)
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])
                if round_idx == 0:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(n + 3*j + 4))])
                else:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(4*n + 2))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec
    
    def smart_casual_ancilla_sec(self, p_data, p1, p2, num_rounds=3) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_data)
        
        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                base = n + 4*j
                Z_part, X_part = stab[:n], stab[n:]
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]

                append_noisy_gate(sec, "RX", 1, [base], p1)
                append_noisy_gate(sec, "RZ", 1, [base + 1, base + 2, base + 3], p1)
                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)

                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[0]], p2)
                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[1]], p2)

                append_noisy_gate(sec, "CNOT", 2, [base, base+2], p2)
                append_noisy_gate(sec, "CNOT", 2, [base+1, base+2], p2)
                append_noisy_gate(sec, "MRZ", 1, [base + 2], p1)
                
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])

                append_noisy_gate(sec, "CNOT", 2, [base, X_supp[2]], p2)
                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[0]], p2)

                append_noisy_gate(sec, "CNOT", 2, [base, base+3], p2)
                append_noisy_gate(sec, "CNOT", 2, [base+1, base+3], p2)
                append_noisy_gate(sec, "MRZ", 1, [base + 3], p1)
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])

                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[1]], p2)
                append_noisy_gate(sec, "CZ", 2, [base, Z_supp[2]], p2)
                for q in Z_supp:
                    if q in X_supp:
                        append_noisy_gate(sec, "S", 1, [base+1], p1)

                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                append_noisy_gate(sec, "MRX", 1, [base], p1)
                append_noisy_gate(sec, "MRZ", 1, [base + 1], p1)
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])
                if round_idx == 0:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(n + 3*j + 4))])
                else:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(4*n + 2))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec
    
    def fully_ft_for_css(self, model, p, num_rounds=3) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()
        ANCILLA_PER_STAB = 3

        noise_dict = noise(p)
        p1 = noise_dict['p1']
        p2 = noise_dict['p2']
        p_init = noise_dict['p_init']
        p_meas = noise_dict['p_meas']
        p_idle = noise_dict['p_idle']

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_init)
        
        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stabilizers):
                base = n + ANCILLA_PER_STAB * j
                Z_part, X_part = stab[:n], stab[n:]
                X_supp = [i for i in range(n) if X_part[i] != 0]
                Z_supp = [i for i in range(n) if Z_part[i] != 0]

                if round_idx == 0:
                    append_noisy_gate(sec, "RX", 1, [base], p_init)
                    append_noisy_gate(sec, "RZ", 1, [base + 1, base + 2], p_init)
                
                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                append_noisy_gate(sec, "CNOT", 2, [base, base + 2], p2)

                for i in range(3):
                    append_noisy_gate(sec, "CNOT", 2, [base + i, X_supp[i]], p2)
                    append_noisy_gate(sec, "CZ", 2, [base + i, Z_supp[i]], p2)
                    if Z_supp[i] in X_supp:
                        append_noisy_gate(sec, "S", 1, [base + i], p1)

                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                append_noisy_gate(sec, "CNOT", 2, [base, base + 2], p2)

                if round_idx == num_rounds - 1:
                    append_noisy_gate(sec, "MX", 1, [base], p_meas)
                    append_noisy_gate(sec, "MZ", 1, [base + 1, base + 2], p_meas)
                else:
                    append_noisy_gate(sec, "MRX", 1, [base], p_meas, p_init)
                    append_noisy_gate(sec, "MRZ", 1, [base + 1, base + 2], p_meas, p_init)

                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2)])
                if round_idx == 0:
                    sec.append("DETECTOR", targets=[stim.target_rec(-3), stim.target_rec(-(n + 2 * j + 3))])
                else:
                    sec.append("DETECTOR", targets=[stim.target_rec(-3), stim.target_rec(-(3*n + 3))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec
    
    def superdense(self, model, p, num_rounds=3) -> stim.Circuit:
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)
        all_logical_paulis = self.get_stim_logical_paulis()
        ANCILLA_PER_STAB = 2
        assert n % 2 == 0

        noise_dict = noise(p)
        p1 = noise_dict['p1']
        p2 = noise_dict['p2']
        p_init = noise_dict['p_init']
        p_meas = noise_dict['p_meas']
        p_idle = noise_dict['p_idle']

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.append("MPP", stabilizer_stim)
        sec.append("DEPOLARIZE1", [i for i in range(n)], p_init)
        
        stab_pairs = []
        for i in range(0, n, 2):
            stab_pairs += [[stabilizers[i], stabilizers[i + 1]]]

        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for j, stab in enumerate(stab_pairs):
                stab1, stab2 = stab[0], stab[1]
                base = n + ANCILLA_PER_STAB * j
                Z_part1, X_part1 = stab1[:n], stab1[n:]
                X_supp1 = [i for i in range(n) if X_part1[i] != 0]
                Z_supp1 = [i for i in range(n) if Z_part1[i] != 0]
                Z_part2, X_part2 = stab2[:n], stab2[n:]
                X_supp2 = [i for i in range(n) if X_part2[i] != 0]
                Z_supp2 = [i for i in range(n) if Z_part2[i] != 0]

                if round_idx == 0:
                    append_noisy_gate(sec, "RX", 1, [base, base + 1], p_init)
                
                append_noisy_gate(sec, "CZ", 2, [base, base + 1], p2)

                for i in X_supp1:
                    append_noisy_gate(sec, "CNOT", 2, [base, i], p2)
                for i in Z_supp1:
                    append_noisy_gate(sec, "CZ", 2, [base, i], p2)
                    if i in X_supp1:
                        append_noisy_gate(sec, "S", 1, [base], p1)
                for i in X_supp2:
                    append_noisy_gate(sec, "CNOT", 2, [base + 1, i], p2)
                for i in Z_supp2:
                    append_noisy_gate(sec, "CZ", 2, [base + 1, i], p2)
                    if i in X_supp2:
                        append_noisy_gate(sec, "S", 1, [base + 1], p1)

                append_noisy_gate(sec, "CZ", 2, [base, base + 1], p2)

                if round_idx == num_rounds - 1:
                    append_noisy_gate(sec, "MX", 1, [base, base + 1], p_meas)
                else:
                    append_noisy_gate(sec, "MRX", 1, [base, base + 1], p_meas, p_init)

                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-n - 1)])
                sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-n - 2)])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec
    

    def benchmark(self, model : str, p : float, num_rounds : int = 3, num_shots : int = 1000, phenomenological=False):
        """
        Use a decoder to numerically compute the logical error rate of the code.
        We use Tesseract, a heuristic-enhanced BP+OSD-type decoder which natively interacts with stim.
        Oversimplified version of circuit-level noise with no idle noise currently implemented.

        Params:
            * p_data (float): 1-qubit data error probability.
            * p_meas (float): measurement output error probability.
            * p1 (float): 1-qubit error probability.
            * p2 (float): 2-qubit error probability.
            * num_rounds (int): number of syndrome extraction rounds.
            * num_shots (int): number of trials to make during benchmarking.
            * phenomenological (bool): if True, the SEC will only put data errors (i.e. p_data before each round) and measurement errors.

        """
        assert 0 <= p <= 1/2, f"Error probability {p} must be within [0, 1/2]"

        print("Making the syndrome extraction circuit...")
        # sec = self.syndrome_extraction_circuit(p_data, p1, p2, num_rounds, option=1)
        # sec = self.new_sec(p_data, p1, p2, num_rounds)
        # sec = self.smart_casual_ancilla_sec(p_data, p1, p2, num_rounds)
        # sec = self.superdense(model, p, num_rounds)
        sec = self.fully_ft_for_css(model, p, num_rounds)
        # sec = self.barely_dressed_ancilla_sec(p_data, p_meas, p1, p2, num_rounds)
        # sec = self.bare_ancilla_sec(p_data, p1, p2, num_rounds)
        print("Done.")
        
        print("Creating detector error model...")
        dem = sec.detector_error_model()
        print("Done.")

        print("Sampling errors from the model...")
        sampler = sec.compile_detector_sampler()
        dets, obs = sampler.sample(num_shots, separate_observables=True)
        print("Done.")

        print("Setting up Tesseract config...")
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=10000,
            no_revisit_dets=True,
            # verbose=True,
            det_orders=tesseract_decoder.utils.build_det_orders(
                dem, num_det_orders=1,
                method=tesseract_decoder.utils.DetOrder.DetIndex,
                seed=2384753),
        )
        print("Done.")
        # print(f'Tesseract decoder configurations --> {tesseract_config}\n')
        
        print("Running Tesseract decoder...")
        sampler = sec.compile_detector_sampler()
        dets, obs = sampler.sample(num_shots, separate_observables=True)
        tesseract_dec = tesseract_config.compile_decoder()
        results = run_tesseract_decoder(tesseract_dec, dets, obs)
        print("Done.")
        print_decoder_results(results)

        return


if __name__ == "__main__":
    """
    Run unit tests.
    """

    # Make some CSS codes and check if they are CSS
    # CSS_group = (4, 7, 3)
    # n = int(np.prod(CSS_group))
    # X0 = ((0, 4, 1), (2, 3, 2))
    # Z0 = ((1, 6, 2), (3, 1, 0), (1, 1, 1))
    # print(canonicalize(CSS_group, X0, Z0))
    # CSS_stabs = find_stabilizers(CSS_group, Z0, X0)
    # print(f"Your CSS stabs are:")
    # for stab in CSS_stabs[0]:
    #     print(symp2Pauli(stab, n))

    code = MirrorCode(
        group = [2, 2, 3, 3],
        z0 = [[0, 0, 0, 0],
       [0, 1, 0, 1],
       [1, 0, 0, 2]],
        x0 = [[0, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 1, 2, 0]]
    )
    # code = MirrorCode(
    #     group = [2, 3, 5],
    #     z0 = [[0, 0, 0],
    #    [0, 0, 1],
    #    [0, 1, 3]],
    #     x0 = [[1, 0, 0],
    #    [1, 0, 2],
    #    [1, 1, 1]],
    #    is_css = False
    # )

    # code = MirrorCode([3, 3], [[0, 0], [0, 1]], [[1, 0], [1, 1]])
    # code = MirrorCode([2, 2], [[0, 0], [0, 1]], [[1, 0], [1, 1]])


    code.benchmark(
        model = 'SD',
        p = 0.001,
        num_rounds = 3,
        num_shots = 1000
    )


    # Make some non-CSS codes and check if they are CSS
