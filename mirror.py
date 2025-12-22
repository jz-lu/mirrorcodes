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


"""
Part I: Mapping the mirror code parameterization to a canonical stabilizer tableau.

We specify a mirror code by giving an abelian group description (a tuple of prime powers describing a 
direct product of cyclic groups) and two subsets, the Z subset and the X subset. This fully describes a mirror code.
The functions in Part I (a) compute the stabilizer tableau of a mirror code description (`find_stabilizers`), 
(b) check if it is CSS (`css_flips`), and (c) check if the code is "canonical". 
By canonical, we mean that it is the unique representative under a set of operations which preserve the code.
These include swapping the Z and X subsets, permuting the elements of the subsets, automorphisms of the group, etc.
By having a canonical representation, we save a great deal of space during numerical search of the code.
"""

def _pair_lex_key(group, z0, x0):
    """
    Internal utility. Build a lex key for comparing pairs (Z0, X0).
    Shorter first list first, then lex on Z0, then shorter second list, then lex on X0.
    """
    n = int(np.prod(group))
    strides = find_strides(group)
    z_combiner = find_strides([n] * len(z0))
    x_combiner = find_strides([n] * len(x0))
    z_idx = int(z_combiner @ z0 @ strides) if len(z0) else 0
    x_idx = int(x_combiner @ x0 @ strides) if len(x0) else 0
    return (len(z0), z_idx, len(x0), x_idx)


def _twoG_iter(group):
    """
    Iterate over all elements of 2G = {2t : t in G} componentwise.
    For odd modulus m, this is all residues 0..m-1 (step 1).
    For even modulus m, this is the even residues 0,2,4,...,m-2 (step 2).
    """
    steps = [1 if m % 2 else 2 for m in group]
    return it.product(*[range(0, m, s) for m, s in zip(group, steps)])


def _column_permutations_by_size(group):
    """
    Build all column permutations that permute only among equal-size cyclic factors.
    This models all isomorphisms that arise from reordering isomorphic factors.
    """
    # collect indices per modulus
    by_size = {}
    for idx, m in enumerate(group):
        by_size.setdefault(int(m), []).append(idx)
    # list of permutations per block
    per_block = [list(it.permutations(block)) for block in by_size.values()]
    # product to yield a full-column permutation each time
    for choice in it.product(*per_block):
        # start with identity mapping
        perm = list(range(len(group)))
        for block_perm in choice:
            # block_perm is a permutation tuple of original indices belonging to one block
            # place them into the same positions they originally occupied, reordered
            for new_pos, old_idx in zip(sorted(block_perm), block_perm):
                perm[new_pos] = old_idx
        yield tuple(perm)


def canonicalize_perms(group, z0, x0, isos, strides):
    """
    Secondary canonicalizer. This accepts the sets z0 and x0 and returns a guess
    for the canonical z0 and x0. Applies all tricks other than swapping z0 and x0.
    Various equivalences that this tries are reordering Z0 and X0, applying an
    automorphism to any component group, including permutations of equal-size
    cyclic factors, and shifting the X0's by 2g, for some g (with the Z/X common
    torsor anchor already applied).

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions. Number of columns should match length of group
          and not exceed the terms of group. Number of rows is the Z weight. Also
          accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions. Number of columns should match length of group
          and not exceed the terms of group. Number of rows is the X weight. Also
          accepts tuples at any level.
        * isos (np.ndarray): list of lists containing per-component multiplicative
          automorphisms (units). Column permutations among equal-size factors
          are also enumerated here.
        * strides (np.ndarray): strides for indexing qubits

    Returns:
        * Tuple containinng two elements. The first is the canonical version of z0,
          the second is the canonical version of x0.
    """
    group = np.array(group)
    z0 = np.array(z0)
    x0 = np.array(x0)
    n = int(np.prod(group))

    # initialize current minimum
    best_z, best_x = z0.copy(), x0.copy()
    best_key = _pair_lex_key(group, best_z, best_x)

    z_combiner = find_strides([n] * len(z0))
    x_combiner = find_strides([n] * len(x0))

    # Enumerate column permutations within equal-size factor blocks
    for col_perm in _column_permutations_by_size(group):
        z_cols = z0[:, col_perm]
        x_cols = x0[:, col_perm]

        # Iterate over all permutations for z-rows
        for z_rows in it.permutations(z_cols):
            z_rows = np.array(z_rows)
            # Anchor torsor: subtract first z to make z'[0]=0
            z_anch = np.mod(z_rows - z_rows[0], group)

            # Apply all component automorphisms (same multiplier vector to both sets)
            for a in it.product(*isos):
                a = np.array(a)
                z_iso = np.mod(z_anch * a, group)

                # Early pruning on Z
                z_idx = int(z_combiner @ z_iso @ strides)
                if (len(z_iso), z_idx) > (len(best_z), int(z_combiner @ best_z @ strides)):
                    continue

                # Prepare X with same anchor (subtract original first z before iso)
                x_base = np.mod(x_cols - z_rows[0], group)

                # Enumerate X-row permutations
                for x_rows in it.permutations(x_base):
                    x_rows = np.array(x_rows)
                    x_iso = np.mod(x_rows * a, group)

                    # Sweep residual 2G: identity-choice degree of freedom
                    # Use util.shift_X to minimize over all X -> X + 2q
                    x_shift = shift_X(group, x_iso)

                    key = _pair_lex_key(group, z_iso, x_shift)
                    if key < best_key:
                        best_key = key
                        best_z, best_x = z_iso, x_shift

    return best_z, best_x


def canonicalize(group, z0, x0):
    """
    Main canonicalizer. This accepts the sets z0 and x0 and returns the canonical
    isomorphic z0 and x0. The canonical form over equivalent codes is not unique.
    Guarantees about this function: after canonicalization arrays z0[0] will only
    contain 0s. Arrays of z0 and x0 are individually sorted by our lex rule.
    Various equivalences tried are:
      * swapping Z0 and X0 (only compared when lengths are equal),
      * reordering Z0 and X0,
      * automorphisms of any component group, including permutations of equal-size
        factors (i.e., reindexing indistinguishable cyclic components),
      * common torsor translation (Z,X)->(Z+p, X+p),
      * identity-choice shift (Z,X)->(Z+h, X-h) which yields X->X+2q residual.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions.

    Returns:
        * Tuple containinng two elements. The first is the canonical version of z0,
          the second is the canonical version of x0.
    """
    group = np.array(group)
    z0 = np.array(z0)
    x0 = np.array(x0)
    isos = find_isos(group)  # units per component
    strides = find_strides(group)

    # Always orient so the shorter list is first unless lengths are equal.
    if len(z0) > len(x0):
        return canonicalize(group, x0, z0)

    # Option 1: keep (Z, X)
    z1, x1 = canonicalize_perms(group, z0, x0, isos, strides)

    # Option 2: swap (X, Z) only when same length, mirroring original behavior
    if len(z0) == len(x0):
        z2, x2 = canonicalize_perms(group, x0, z0, isos, strides)
        return (z1, x1) if _pair_lex_key(group, z1, x1) <= _pair_lex_key(group, z2, x2) else (z2, x2)

    return (z1, x1)


def is_Z_canonical(group, z0, isos):
    """
    Canonicalization checker for just Z's. Does not check legality. Merely checks
    whether it can find a smaller equivalent z0 instance under:
      * reordering within Z0,
      * automorphisms of component groups (units),
      * permutations among equal-size cyclic factors,
      * a single torsor anchor (shift so first row is 0).

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions.
        * isos (list of lists): list of per-component units

    Returns:
        * Boolean with whether z0 is canonical.
    """
    group = np.array(group)
    z0 = np.array(z0)

    strides = find_strides(group)
    combiner = find_strides([int(np.prod(group))] * len(z0))
    base_index = int(combiner @ z0 @ strides)

    for col_perm in _column_permutations_by_size(group):
        z_cols = z0[:, col_perm]
        for rows in it.permutations(z_cols):
            rows = np.array(rows)
            shifted = np.mod(rows - rows[0], group)
            for a in it.product(*isos):
                a = np.array(a)
                cand = np.mod(shifted * a, group)
                if int(combiner @ cand @ strides) < base_index:
                    return False
    return True


def is_X_canonical(group, x0, isos):
    """
    Canonicalization checker for just X's. Does not check legality. Merely checks
    whether it can find a smaller equivalent x0 instance under:
      * reordering within X0,
      * automorphisms of component groups (units),
      * permutations among equal-size cyclic factors,
      * the same torsor anchor (first row to 0),
      * an independent residual sweep over 2G.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions.
        * isos (list of lists): list of per-component units

    Returns:
        * Boolean with whether x0 is canonical.
    """
    group = np.array(group)
    x0 = np.array(x0)

    strides = find_strides(group)
    combiner = find_strides([int(np.prod(group))] * len(x0))

    # Baseline: anchor by subtracting first row, then minimize over residual 2G only
    base_anch = np.mod(x0 - x0[0], group)
    base_idx = min(int(combiner @ (np.mod(base_anch + np.array(delta), group)) @ strides)
                   for delta in _twoG_iter(group))

    for col_perm in _column_permutations_by_size(group):
        x_cols = x0[:, col_perm]
        for rows in it.permutations(x_cols):
            rows = np.array(rows)
            anch = np.mod(rows - rows[0], group)
            for a in it.product(*isos):
                a = np.array(a)
                cand = np.mod(anch * a, group)
                cand_min_idx = min(int(combiner @ (np.mod(cand + np.array(delta), group)) @ strides)
                                   for delta in _twoG_iter(group))
                if cand_min_idx < base_idx:
                    return False
    return True


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
    
def append_noisy_gate(circuit : stim.Circuit, gate : str, locality : int, qubits : list, p : float, after : bool = False):
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

    if locality == 1:
        if after:
            circuit.append(gate, qubits)
            circuit.append("DEPOLARIZE1", qubits, p)
        else:
            circuit.append("DEPOLARIZE1", qubits, p)
            circuit.append(gate, qubits)
    else:
        if after:
            circuit.append(gate, qubits)
            circuit.append("DEPOLARIZE2", qubits, p)
        else:
            circuit.append("DEPOLARIZE2", qubits, p)
            circuit.append(gate, qubits)
    
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
                append_noisy_gate(sec, "RX", 1, [ancilla_block_qubit, ancilla_block_qubit + 1], p1, after=True)

                # Initialize last 3 qubits to |0>
                append_noisy_gate(sec, "RZ", 1, [ancilla_block_qubit + 2, ancilla_block_qubit + 3, ancilla_block_qubit + 4], p1, after=True)

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
                                    append_noisy_gate(sec, "RZ", 1, [(j+1)*n + 3], p1, after=True)
                                    append_noisy_gate(sec, "RX", 1, [(j+1)*n + 4], p1, after=True)
                
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
                append_noisy_gate(sec, "RX", 1, [n+j], p1, after=True)
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

                append_noisy_gate(sec, "RX", 1, [base], p1, after=True)
                append_noisy_gate(sec, "RZ", 1, [base + 1], p1, after=True)
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

                append_noisy_gate(sec, "RX", 1, [base], p1, after=True)
                append_noisy_gate(sec, "RZ", 1, [base + 1, base + 2, base + 3], p1, after=True)
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
    

    def barely_dressed_ancilla_sec(self, p_data, p1, p2, num_rounds=3) -> stim.Circuit:
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

                append_noisy_gate(sec, "RX", 1, [base], p1, after=True)
                append_noisy_gate(sec, "RZ", 1, [base + 1], p1, after=True)
                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)

                for q in X_supp:
                    append_noisy_gate(sec, "CNOT", 2, [base, q], p2)
                for q in Z_supp:
                    append_noisy_gate(sec, "CZ", 2, [base, q], p2)
                    if q in X_supp:
                        # raise LookupError("You found me :)")
                        append_noisy_gate(sec, "S", 1, [base+1], p1)

                append_noisy_gate(sec, "CNOT", 2, [base, base + 1], p2)
                append_noisy_gate(sec, "MRX", 1, [base], p1)
                append_noisy_gate(sec, "MRZ", 1, [base + 1], p1)
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])
                if round_idx == 0:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(n + j + 2))])
                else:
                    sec.append("DETECTOR", targets=[stim.target_rec(-2), stim.target_rec(-(2*n + 2))])

        append_observable_includes_for_paulis(circuit=sec, paulis=all_logical_paulis)
        return sec

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

    def benchmark(self, p_data : float, p1 : float, p2 : float, num_rounds : int = 3, num_shots : int = 1000):
        """
        Use a decoder to numerically compute the logical error rate of the code.
        We use Tesseract, a heuristic-enhanced BP+OSD-type decoder which natively interacts with stim.
        Oversimplified version of circuit-level noise with no idle noise currently implemented.

        Params:
            * p_data (float): 1-qubit data error probability.
            * p1 (float): 1-qubit error probability.
            * p2 (float): 2-qubit error probability.
            * num_rounds (int): number of syndrome extraction rounds.
            * num_shots (int): number of trials to make during benchmarking

        """
        assert 0 <= p1 <= 3/4, f"1-qubit error probability {p1} must be within [0, 3/4]"
        assert 0 <= p1 <= 15/16, f"2-qubit error probability {p2} must be within [0, 15/16]"

        print("Making the syndrome extraction circuit...")
        # sec = self.syndrome_extraction_circuit(p_data, p1, p2, num_rounds, option=1)
        # sec = self.new_sec(p_data, p1, p2, num_rounds)
        sec = self.smart_casual_ancilla_sec(p_data, p1, p2, num_rounds)
        # sec = self.barely_dressed_ancilla_sec(p_data, p1, p2, num_rounds)
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
        p_data = 0.04,
        p1 = 0.0001,
        p2 = 0.002,
        num_rounds = 6,
        num_shots = 1000
    )


    # Make some non-CSS codes and check if they are CSS
