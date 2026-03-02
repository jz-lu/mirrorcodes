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
from circuit import cached_schedule
from util import find_strides
from util import binary_rank, stimify_symplectic
from benchmark import make_noise_model
from distance import distance, distance_estimate, make_code
import stim
from non_abelian import build_indexed_group_ops
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

def non_abelian_stabilizers(code):
    g = code.actualgroup
    n = code.get_n()
    stabs = np.zeros((n, 2 * n), dtype=np.uint8)
    for i in range(n):
        for j in code.z0:
            stabs[i, g.mul(j, i)] = 1
        for j in code.x0:
            if code.symmetric:
                stabs[i, g.mul(j, g.inv(i)) + n] = 1
            else:
                stabs[i, g.mul(g.inv(i), j) + n] = 1
    comm = np.mod(stabs[:, :n] @ stabs[:, n:].T, 2)
    return stabs if (comm == comm.T).all() else None

def valid_non_abelian(code):
    return code.get_stabilizers() is not None

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

    if can_flip:
        for g in flips:
            index = g @ strides
            stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]]
    return stabilizers, can_flip


"""
Part II: stim circuit manipulation

Here we have some custom functions which modify a stim circuit in place for numerical analysis purposes.
"""

class PushCircuit:
    """
    A class which holds a stim circuit and has some custom functions to modify the circuit in place.
    """
    def __init__(self, noise, base_n, qps):
        """
        Initialize the circuit.

        Params:
            * noise (dict): noise model object which contains the error probabilities to be applied. 
                            Should have attributes p_init, p_idle, p_meas, p1, and p2.
            * base_n (int): number of physical qubits in the code
            * qps (int): "qubits per stabilizer". This is the number of ancilla qubits required to measure each stabilizer.
        """
        self.circuit = stim.Circuit()
        self.n = base_n * (qps + 1)
        self.base_n = base_n
        self.qps = qps
        self.idling = [True] * self.n
        self.noise = noise
        self.resonate = False

    def push_gate(self, gate : str, targets, noiseless = False):
        """
        Inserts a noisy gate into the circuit by adding the desired gate, preceded by a depolarizing error.
        Only supports 1- and 2-qubit gates. 
        For a t-qubit gate, t-qubit depolarizing noise will be added after the gate.

        Params:
            * gate (str): string description of the gate, e.g. 'RX' or 'CNOT'.
            * targets (list): the qubits on the circuit on which the gate is applied. 
                For 1-qubit gate, will apply on every qubit in list. For 2-qubit gate, must be only 2 qubits.
            * noiseless (bool): if True, no noise will be applied to the gate.
        
        Returns:
            * None (modifies the circuit in place).
        """
        if len(gate) > 1 and gate[:2] == "MR":
            assert False, "Measurements and resets should be separate"
        if ((gate[0] == 'M' and gate != 'MPP') or gate[0] == 'R') and len(targets) > 1:
            for qubit in targets:
                self.push_gate(gate, [qubit], noiseless=noiseless)
            return
        if gate not in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
            for qubit in targets:
                self.idling[qubit] = False
        if noiseless or gate in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
            self.circuit.append(gate, targets)
            return
        if gate[0] == 'M':
            self.circuit.append(gate, targets, self.noise['p_meas'])
            self.resonate = True
            return
        self.circuit.append(gate, targets)

        # Reset gates: set to |+>, |0>, |+i>, respectively for X, Y, and Z
        if gate[0] == 'R':
            # self.resonate = True
            self.circuit.append("Z_ERROR" if gate == 'RX' else 'X_ERROR', targets, self.noise['p_init'])

        # 1- and 2-qubit unitary gates. Locality t = t-qubit gate. We only do 1 and 2. All 2-qubit gates are controlled-U gates.
        else:
            if gate[0] == 'C':
                locality = 2
            else:
                locality = 1
            self.circuit.append(f"DEPOLARIZE{locality}", targets, self.noise[f'p{locality}'])
        return
    
    def tick(self):
        """
        Increment time forward by 1 unit. This is important for the introduction of idle noise.
        At each time step, if nothing happens to a qubit, then we say it is "idling" at that time, and 
        we apply idle noise to it.
        """
        for qubit in range(self.n):
            if self.idling[qubit]:
                self.circuit.append("DEPOLARIZE1", qubit, self.noise['p_res_idle'] if self.resonate else self.noise['p_idle'])
            self.idling[qubit] = True
        self.resonate = False
        self.circuit.append("TICK")

    def gate_round(self, gate, targets_list, noiseless=False, tick_after=True):
        """
        Call this function if you want to apply a gate to every qubit in `targets_list` in each block of ancilla
        qubits used to extract the syndrome associated with a given stabilizer.

        Params:
            * gate (str): description of gate, e.g. 'X'.
            * targets_list (list): list of qubits the gate acts on. Note that a 2-qubit gate's targets must be only the 2 qubits that
                                   it acts on. Each 2-qubit gate should be pushed separately.
            * noiseless (bool): set to `True` to forego all noise.
            * tick_after (bool): increment time forward if `True`.
        
        Returns: None
        """
        for j in range(self.base_n):
            self.push_gate(gate, [self.base_n + j * self.qps + target for target in targets_list], noiseless=noiseless)
        if tick_after:
            self.tick()


"""
Some stim functions (adapted from the Tesseract Decoder tutorial)
whose goal is to take a Pauli observable and include it into a stim circuit.
This is useful for computing the distance of a code via stim.
"""
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
    def __init__(self, group, z0, x0, n=None, k=None, d=None, is_css=None, d_est=None,
                 abelian=True, symmetric=True, actualgroup=None):
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
        self.abelian = abelian
        self.symmetric = symmetric
        self.actualgroup = actualgroup
        if actualgroup is None and not self.abelian:
            self.actualgroup = build_indexed_group_ops(group)

    def get_stabilizers(self):
        if self.stabilizers is None:
            if self.abelian:
                self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
            else:
                self.stabilizers = non_abelian_stabilizers(self)
                self.CSS = False
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
            self.n = int(np.prod(self.group)) if self.abelian else self.actualgroup.n
        return self.n
    
    def get_k(self):
        if self.k is None:
            self.k = self.get_n() - binary_rank(self.get_stabilizers())
        return self.k
    
    def get_d(self, verbose=False):
        if self.d is None:
            tableau = self.get_stim_tableau()
            assert self.CSS is not None, f"The guy who programmed this screwed up somewhere?"
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
            self.get_stabilizers()
            return self.CSS
        return self.CSS

    def get_rate(self):
        return self.get_k() / self.get_n()
    
    def get_rel_dist(self):
        return self.get_d() / self.get_n()
    
    def phenomenological_sec(self, noise, num_rounds=3):
        """
        Generic unphysical SEC which works for any stabilizer code. Useful for phenomenological noise analysis,
        in which we only consider measurement errors and some depolarizing error at the beginning of every round
        of syndrome extraction.

        Params:
            * noise (dict): noise object with keys 'p_depol' and 'p_meas'
            * num_rounds (int): number of rounds of syndrome extraction.
        """
        num_qubits = num_stabilizers = self.get_n()
        stabilizer_paulis = self.get_stim_tableau()
        all_logicals_paulis = self.get_stim_logical_paulis()

        circuit = stim.Circuit()

        append_observable_includes_for_paulis(circuit=circuit, paulis=all_logicals_paulis)
        circuit.append("MPP", stabilizer_paulis)

        for _ in range(num_rounds):
            circuit.append("DEPOLARIZE1", targets=list(range(num_qubits)), arg=noise['p_depol'])
            circuit.append("MPP", stabilizer_paulis, arg=noise['p_meas'])

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

    def bare_ancilla_sec(self, noise, num_rounds=3) -> stim.Circuit:
        """
        Initialize one qubit per stabilizer, densely use controlled-NOT gates to 
        extract the syndrome from every single stabilizer, with essentially no optimization being done
        to reduce the depth other than using a SAT solver to schedule the gates efficiently.
        (We do this in all SECs, however.)
        Key features: no attempt at FT, but minimal overhead of 1 qubit per stabilizer.

        Params:
            * noise (dict): noise object (see PushCircuit documentation)
            * num_rounds (int): number of rounds of syndrome extraction.
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        QUBITS_PER_STAB = 1
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)

        sec = PushCircuit(noise, n, QUBITS_PER_STAB)

        circuit_solution = cached_schedule(self)[4]
        max_tick = 0
        for i in circuit_solution:
            for j in i:
                if j is not None:
                    max_tick = max(max_tick, j)

        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        
        for _ in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            sec.gate_round("RX", [0])
            for tick in range(1, max_tick + 1):
                for k in range(n):
                    for j, stab in enumerate(stabilizers):
                        if circuit_solution[j][k] == tick:
                            if stab[k] == 1 and stab[n + k] == 0:
                                sec.push_gate("CZ", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 0 and stab[n + k] == 1:
                                sec.push_gate("CX", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 1 and stab[n + k] == 1:
                                sec.push_gate("CY", [n + j * QUBITS_PER_STAB, k])
                            else:
                                assert False, f"Invalid stabilizer entry {stab[k]}{stab[n + k]} at stabilizer {j} and qubit {k}"
                sec.tick()
            sec.gate_round("MX", [0])
            for j, stab in enumerate(stabilizers):
                sec.push_gate("DETECTOR", [stim.target_rec(-n + j), stim.target_rec(-2 * n + j)])
        
        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        for j, stab in enumerate(stabilizers):
            sec.push_gate("DETECTOR", [stim.target_rec(-n + j), stim.target_rec(-2 * n + j)])

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)
        return sec.circuit
    
    def loop_flag_sec(self, noise, num_rounds=3) -> stim.Circuit:
        """
        Adds on a second qubit for each stabilizer used as a flag to detect if an error happens during
        the naive bare syndrome extraction onto the single qubit in the bare ancilla SEC.
        Key features: baby's first attempt at FT (provably FT for weight <= 4), small overhead of 2 qubits per stabilizer.

        Params:
            * noise (dict): noise object (see PushCircuit documentation)
            * num_rounds (int): number of rounds of syndrome extraction.
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        QUBITS_PER_STAB = 2
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)

        sec = PushCircuit(noise, n, QUBITS_PER_STAB)

        circuit_solution = cached_schedule(self)[4]
        max_tick = 0
        for i in circuit_solution:
            for j in i:
                if j is not None:
                    max_tick = max(max_tick, j)

        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        
        for i in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            sec.gate_round("RX", [0], tick_after=False)
            sec.gate_round("RZ", [1])
            sec.gate_round("CX", [0, 1])
            for tick in range(1, max_tick + 1):
                for k in range(n):
                    for j, stab in enumerate(stabilizers):
                        if circuit_solution[j][k] == tick:
                            if stab[k] == 1 and stab[n + k] == 0:
                                sec.push_gate("CZ", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 0 and stab[n + k] == 1:
                                sec.push_gate("CX", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 1 and stab[n + k] == 1:
                                sec.push_gate("CY", [n + j * QUBITS_PER_STAB, k])
                            else:
                                assert False, f"Invalid stabilizer entry {stab[k]}{stab[n + k]} at stabilizer {j} and qubit {k}"
                sec.tick()
            sec.gate_round("CX", [0, 1])
            sec.gate_round("MX", [0], tick_after=False)
            sec.gate_round("MZ", [1])
            for j, stab in enumerate(stabilizers):
                sec.push_gate("DETECTOR", [stim.target_rec(-(n - j))])
                if i == 0:
                    sec.push_gate("DETECTOR", [stim.target_rec(-n * QUBITS_PER_STAB + j), stim.target_rec(-n * QUBITS_PER_STAB - n + j)])
                else:
                    sec.push_gate("DETECTOR", [stim.target_rec(-n * QUBITS_PER_STAB + j), stim.target_rec(-2 * n * QUBITS_PER_STAB + j)])

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        for j, stab in enumerate(stabilizers):
            sec.push_gate("DETECTOR", [stim.target_rec(-n + j), stim.target_rec(-(QUBITS_PER_STAB+1) * n + j)])

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)
        return sec.circuit
    
    def ft_for_w6_css_sec(self, noise, num_rounds=3) -> stim.Circuit:
        """
        Adds on a third qubit for each stabilizer which adds an additional flag on top of the
        loop flag SEC to detect more extraction-time errors.

        Key features: provably FT for CSS codes w/ weight <= 6 stabilizers, 3 qubits per stabilizer.

        Params:
            * noise (dict): noise object (see PushCircuit documentation)
            * num_rounds (int): number of rounds of syndrome extraction.
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        QUBITS_PER_STAB = 3
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)

        sec = PushCircuit(noise, n, QUBITS_PER_STAB)

        circuit_solution = cached_schedule(self)[4]
        max_tick = 0
        for i in circuit_solution:
            for j in i:
                if j is not None:
                    max_tick = max(max_tick, j)

        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        
        for i in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            sec.gate_round("RX", [0], tick_after=False)
            sec.gate_round("RZ", [1, 2])
            sec.gate_round("CX", [0, 1])
            sec.gate_round("CX", [0, 2])
            for tick in range(1, max_tick + 1):
                for k in range(n):
                    for j, stab in enumerate(stabilizers):
                        indices = [l for l in circuit_solution[j] if l is not None]
                        indices.sort()
                        if circuit_solution[j][k] == tick:
                            control = 0
                            if tick in indices[2:4]:
                                control = 1
                            if tick in indices[4:]:
                                control = 2
                            if stab[k] == 1 and stab[n + k] == 0:
                                sec.push_gate("CZ", [n + j * QUBITS_PER_STAB + control, k])
                            elif stab[k] == 0 and stab[n + k] == 1:
                                sec.push_gate("CX", [n + j * QUBITS_PER_STAB + control, k])
                            elif stab[k] == 1 and stab[n + k] == 1:
                                sec.push_gate("CY", [n + j * QUBITS_PER_STAB + control, k])
                            else:
                                assert False, f"Invalid stabilizer entry {stab[k]}{stab[n + k]} at stabilizer {j} and qubit {k}"
                sec.tick()
            sec.gate_round("CX", [0, 1])
            sec.gate_round("CX", [0, 2])
            sec.gate_round("MX", [0], tick_after=False)
            sec.gate_round("MZ", [1, 2])
            for j, stab in enumerate(stabilizers):
                sec.push_gate("DETECTOR", [stim.target_rec(-2 * n + 2 * j)])
                sec.push_gate("DETECTOR", [stim.target_rec(-2 * n + 2 * j + 1)])
                if i == 0:
                    sec.push_gate("DETECTOR", [stim.target_rec(-3 * n + j), stim.target_rec(-4 * n + j)])
                else:
                    sec.push_gate("DETECTOR", [stim.target_rec(-3 * n + j), stim.target_rec(-6 * n + j)])
            sec.tick()

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        for j, stab in enumerate(stabilizers):
            sec.push_gate("DETECTOR", [stim.target_rec(-n + j), stim.target_rec(-4 * n + j)])

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)
        return sec.circuit

    def ft_for_w6_sec(self, noise, num_rounds=3) -> stim.Circuit:
        """
        Uses a more complicated FT gadget to deal with the fact that CNOT gates cause more 
        complicated correlated errors on stabilizer codes relative to CSS codes.
        Key features: provably FT for all weight <= 6 codes, but larger overhead of 6 qubits per stabilizer.

        Params:
            * noise (dict): noise object (see PushCircuit documentation)
            * num_rounds (int): number of rounds of syndrome extraction.
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        QUBITS_PER_STAB = 6
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)

        sec = PushCircuit(noise, n, QUBITS_PER_STAB)

        circuit_solution = cached_schedule(self)[4]
        max_tick = 0
        for i in circuit_solution:
            for j in i:
                if j is not None:
                    max_tick = max(max_tick, j)
    
        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)

        for i in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            sec.gate_round("RX", [0], tick_after=False)
            sec.gate_round("RZ", [1, 2, 3, 4, 5])
            sec.gate_round("CX", [0, 1])
            for tick in range(1, max_tick + 1):
                for j, stab in enumerate(stabilizers):
                    if tick in circuit_solution[j]:
                        k = circuit_solution[j].index(tick)
                    else:
                        continue
                    base = n + j * QUBITS_PER_STAB
                    indices = [l for l in circuit_solution[j] if l is not None]
                    indices.sort()
                    num = indices.index(tick)
                    if stab[k] == 1 and stab[n + k] == 0:
                        sec.push_gate("CZ", [base + (num % 3), k])
                    elif stab[k] == 0 and stab[n + k] == 1:
                        sec.push_gate("CX", [base + (num % 3), k])
                    elif stab[k] == 1 and stab[n + k] == 1:
                        sec.push_gate("CY", [base + (num % 3), k])
                    else:
                        assert False, f"Invalid stabilizer entry {stab[k]}{stab[n + k]} at stabilizer {j} and qubit {k}"
                    if num == 0:
                        sec.push_gate("CX", [base + 1, base + 2])
                    elif num == 1:
                        sec.push_gate("CX", [base, base + 3])
                    elif num == 2:
                        sec.push_gate("CX", [base + 1, base + 3])
                        sec.push_gate("CX", [base, base + 4])
                    elif num == 3:
                        sec.push_gate("CX", [base + 2, base + 4])
                        sec.push_gate("CX", [base + 1, base + 5])
                    elif num == 4:
                        sec.push_gate("CX", [base + 2, base + 5])
                sec.tick()
            sec.gate_round("MX", [0, 1, 2], tick_after=False)
            sec.gate_round("MZ", [3, 4, 5])
            for j, stab in enumerate(stabilizers):
                sec.push_gate("DETECTOR", [stim.target_rec(-3 * n + 3 * j),
                                           stim.target_rec(-3 * n + 3 * j + 1)])
                sec.push_gate("DETECTOR", [stim.target_rec(-3 * n + 3 * j + 2)])
                if i == 0:
                    sec.push_gate("DETECTOR", [stim.target_rec(-6 * n + 3 * j),
                                               stim.target_rec(-6 * n + 3 * j + 1),
                                               stim.target_rec(-6 * n + 3 * j + 2),
                                               stim.target_rec(-7 * n + j)])
                else:
                    sec.push_gate("DETECTOR", [stim.target_rec(-6 * n + 3 * j),
                                               stim.target_rec(-6 * n + 3 * j + 1),
                                               stim.target_rec(-6 * n + 3 * j + 2),
                                               stim.target_rec(-12 * n + 3 * j),
                                               stim.target_rec(-12 * n + 3 * j + 1),
                                               stim.target_rec(-12 * n + 3 * j + 2)])
            sec.tick()

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        for j, stab in enumerate(stabilizers):
            sec.push_gate("DETECTOR", [stim.target_rec(-n + j), 
                                       stim.target_rec(-7 * n + 3 * j),
                                       stim.target_rec(-7 * n + 3 * j + 1),
                                       stim.target_rec(-7 * n + 3 * j + 2)])

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)
        return sec.circuit

    def superdense_sec(self, noise, num_rounds=3) -> stim.Circuit:
        """
        !! This is a heuristic !!

        Initialize one qubit per stabilizer, but in pairs which are Bell pairs.
        Then do bare syndrome extraction on each pair. 
        This adds FT relative to bare SEC, in the sense that the paired
        qubits "flag" each other.

        Params:
            * noise (dict): noise object (see PushCircuit documentation)
            * num_rounds (int): number of rounds of syndrome extraction.
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        """
        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        QUBITS_PER_STAB = 1
        stabilizers = self.get_stabilizers()
        n = self.get_n()
        stabilizers_stim = stimify_symplectic(stabilizers)

        sec = PushCircuit(noise, n, QUBITS_PER_STAB)

        circuit_solution = cached_schedule(self)[4]
        max_tick = 0
        for i in circuit_solution:
            for j in i:
                if j is not None:
                    max_tick = max(max_tick, j)

        all_logical_paulis = self.get_stim_logical_paulis()

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)

        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        
        for i in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            sec.gate_round("RX", [0], tick_after=False)
            if n % 2 == 1:
                sec.push_gate("RX", [n + n * QUBITS_PER_STAB])
            sec.tick()
            for j in range((len(stabilizers) + 1) // 2):
                if j % 2 == 0:
                    sec.push_gate("CZ", [n + j * QUBITS_PER_STAB, n + j * QUBITS_PER_STAB + 1])
            sec.tick()
            for tick in range(1, max_tick + 1):
                for k in range(n):
                    for j, stab in enumerate(stabilizers):
                        if circuit_solution[j][k] == tick:
                            if stab[k] == 1 and stab[n + k] == 0:
                                sec.push_gate("CZ", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 0 and stab[n + k] == 1:
                                sec.push_gate("CX", [n + j * QUBITS_PER_STAB, k])
                            elif stab[k] == 1 and stab[n + k] == 1:
                                sec.push_gate("CY", [n + j * QUBITS_PER_STAB, k])
                            else:
                                assert False, f"Invalid stabilizer entry {stab[k]}{stab[n + k]} at stabilizer {j} and qubit {k}"
            for j in range((len(stabilizers) + 1) // 2):
                if j % 2 == 0:
                    sec.push_gate("CZ", [n + j * QUBITS_PER_STAB, n + j * QUBITS_PER_STAB + 1])
            sec.tick()
            sec.gate_round("MX", [0], tick_after=False)
            if n % 2 == 1:
                sec.push_gate("MX", [n + n * QUBITS_PER_STAB])
            for j, stab in enumerate(stabilizers):
                offset = 0
                if n % 2 == 1:
                    sec.push_gate("DETECTOR", [stim.target_rec(-1)])
                    offset = 1
                if i == 0:
                    sec.push_gate("DETECTOR", [stim.target_rec(-n - offset + j), stim.target_rec(-2 * n - offset + j)])
                else:
                    sec.push_gate("DETECTOR", [stim.target_rec(-n - offset + j), stim.target_rec(-2 * n - 2 * offset + j)])
            sec.tick()
        
        for stabilizer_stim in stabilizers_stim:
            sec.push_gate("MPP", stabilizer_stim, noiseless=True)
        for j, stab in enumerate(stabilizers):
            sec.push_gate("DETECTOR", [stim.target_rec(-n + j), stim.target_rec(-2 * n - offset + j)])

        append_observable_includes_for_paulis(circuit=sec.circuit, paulis=all_logical_paulis)
        return sec.circuit

    def benchmark(self, noise_model_name : str, p : float, num_rounds : int = 3, num_shots : int = 1000, phenomenological=False):
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

        sec = self.bare_ancilla_sec(make_noise_model(p, noise_model_name), num_rounds)
        #sec = self.loop_flag_sec(noise(p, noise_model_name), num_rounds)
        #sec = self.ft_for_w6_css_sec(noise(p, noise_model_name), num_rounds)
        #sec = self.ft_for_w6_sec(make_noise_model(p, noise_model_name), num_rounds)
        #sec = self.superdense(noise(p, noise_model_name), num_rounds)

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

    # code = MirrorCode(
    #     group = [2, 2, 3, 3],
    #     z0 = [[0, 0, 0, 0],
    #    [0, 1, 0, 1],
    #    [1, 0, 0, 2]],
    #     x0 = [[0, 0, 0, 0],
    #    [0, 1, 1, 0],
    #    [1, 1, 2, 0]]
    # )
    
    G = [2, 2, 4, 3, 3]
    A = [[0, 0, 0, 0, 0],
     [0, 0, 1, 0, 1],
     [0, 1, 0, 0, 2]]
    B = [[1, 0, 0, 0, 0],
     [1, 0, 1, 1, 0],
     [1, 1, 3, 2, 0]]
    
    code = MirrorCode(
        group = G,
        z0 = A,
        x0 = B
    )
    print(code)

    code.benchmark(
        noise_model_name = 'SI1000',
        p = 0.001,
        num_rounds = 3,
        num_shots = 1000
    )


    # Make some non-CSS codes and check if they are CSS
