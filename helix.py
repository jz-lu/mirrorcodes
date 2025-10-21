"""
`helix.py`
Code file to construct a helix code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} x ... x Z_{aN}
    * Z0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to Z0
    * X0 [{u1, u2, ..., uQ}]: a set of N-tuples specifying elements of the group which belong to X0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the helix code.
The check weight is exactly |Z0| + |X0| - |Z0 ^ X0| <= |Z0| + |X0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form---there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
"""
import itertools as it
import numpy as np
from util import find_isos, find_strides, shift_X
from util import binary_rank, symp2Pauli, stimify_symplectic
from distance import distance
import stim


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

    if can_flip:
        for g in flips:
            index = g @ strides
            stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]]
    return stabilizers, can_flip


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
    def __init__(self, group, z0, x0, n=None, k=None, d=None, is_css=None):
        self.group = group
        self.z0 = z0
        self.x0 = x0
        self.wx = len(x0)
        self.wz = len(z0)
        self.stabilizers = None
        self.stim_tableau = None
        self.CSS = is_css

        self.n = int(n) if n is not None else None
        self.k = k
        self.d = d

    def get_stabilizers(self):
        if self.stabilizers is None:
            self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
        return self.stabilizers
    
    def get_stim_tableau(self):
        if self.stim_tableau is None:
            self.stim_tableau = stimify_symplectic(self.get_stabilizers())
        return self.stim_tableau

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
    
    def is_CSS(self):
        if self.CSS is None:
            self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
        return self.CSS

    def get_rate(self):
        return self.get_k() / self.get_n()
    
    def get_rel_dist(self):
        return self.get_d() / self.get_n()
    
    def syndrome_extraction_circuit(self, num_rounds=3) -> stim.Circuit:
        """
        Make a syndrome extraction circuit corresponding to the mirror code
        instantiated in this class.

        Params:
            * num_rounds (int): number of rounds of syndrome extraction. 
        
        Returns:
            * stim.Circuit object of the syndrome extraction circuit for the mirror code.
        
        TODO: add noise in the relevant parts of the circuit
        """
        assert self.wz == 3 and self.wx == 3, f"Idk how to make short circuits otherwise"

        # The first n qubits are the data qubits, and will be the controls for the syndromes.
        sec = stim.Circuit()
        stabilizers = self.get_stabilizers()
        ANCILLA_PER_STAB = 5
        n = self.get_n()
        stabilizer_stim = stimify_symplectic(stabilizers)

        # Do some perfect measurements before we get into actual extraction for detection purposes.
        # append_observable_includes_for_paulis(circuit=sec, paulis=all_logicals_paulis)
        sec.append("MPP", stabilizer_stim)
        # append_observable_includes_for_paulis(circuit=sec, paulis=all_logicals_paulis)

        # Initialize ancillary system
        for ancilla_block_qubit in range(n, (ANCILLA_PER_STAB+1)*n, ANCILLA_PER_STAB):
            # Initialize first 2 qubits to |+>
            sec.append("RX", [ancilla_block_qubit, ancilla_block_qubit + 1])

            # Initialize last 3 qubits to |0>
            sec.append("RZ", [ancilla_block_qubit + 2, ancilla_block_qubit + 3, ancilla_block_qubit + 4])

            # Add a CNOT to make a Bell pair
            sec.append("CNOT", [ancilla_block_qubit, ancilla_block_qubit + 2])
        
        for round_idx in range(num_rounds):
            # Do the syndrome extraction in parallel for each stabilizer
            for op in range(6):
                for j, stab in enumerate(stabilizers):
                    Z_part, X_part = stab[:n], stab[n:]

                    # X part first
                    X_supp = [i for i in range(n) if X_part[i] != 0]
                    Z_supp = [i for i in range(n) if Z_part[i] != 0]
                    assert len(X_supp) == len(Z_supp) == 3
                    if op <= 2:
                        # CNOT between X check and ancilla
                        sec.append("CNOT", [X_supp[op], (j+1)*n + op])
                        if op == 1:
                            # Do a CNOT between ancillas 0 and 3
                            sec.append("CNOT", [(j+1)*n, (j+1)*n + 3])
                        elif op == 2:
                            # Do a CNOT between ancillas 1 and 4
                            sec.append("CNOT", [(j+1)*n + 1, (j+1)*n + 4])
                    elif op <= 5:
                        # CZ between Z check and anncilla
                        if op == 3:
                            # Also add 2 CNOT gates between the ancillas
                            sec.append("CZ", [Z_supp[0], (j+1)*n])
                            sec.append("CNOT", [(j+1)*n + 1, (j+1)*n + 3])
                            sec.append("CNOT", [(j+1)*n + 2, (j+1)*n + 4])
                        elif op == 4:
                            # Also add measurements of last 2 ancillas and detections for some ancillas, then reset
                            sec.append("CZ", [Z_supp[1], (j+1)*n + 2])
                            sec.append("MZ", [(j+1)*n + 3, (j+1)*n + 4])
                            sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2)])
                        elif op == 5:
                            sec.append("CZ", [Z_supp[2], (j+1)*n + 1])
                            sec.append("CNOT", [(j+1)*n, (j+1)*n + 2])
                            # Reset the last 2 ancillas
                            if round_idx < num_rounds - 1:
                                sec.append("RZ", [(j+1)*n + 3])
                                sec.append("RX", [(j+1)*n + 4])
            
            # If this is the last round, measure the first 3 ancillas and call it a day.
            # If this is not the last round, also initialize a new Bell pair in parallel.
            for j in range(n):
                sec.append("MX", [(j+1)*n, (j+1)*n + 1]) # measure the first 2 ancillas 
                sec.append("DETECTOR", targets=[stim.target_rec(-1), stim.target_rec(-2)])
                sec.append("MZ", [(j+1)*n + 2]) # measure the third ancilla
                sec.append("DETECTOR", targets=[stim.target_rec(-1)])

            if round_idx < num_rounds - 1: # more rounds to go
                for j in range(n):
                    sec.append("CNOT", [(j+1)*n + 3, (j+1)*n + 4])

                    sec.append("SWAP", [(j+1)*n + 3, (j+1)*n]) #! this action should be noiseless
                    sec.append("SWAP", [(j+1)*n + 4, (j+1)*n + 2]) #! this action should be noiseless

        return sec


if __name__ == "__main__":
    """
    Run unit tests.
    """

    # Make some CSS codes and check if they are CSS
    CSS_group = (4, 7, 3)
    n = int(np.prod(CSS_group))
    X0 = ((0, 4, 1), (2, 3, 2))
    Z0 = ((1, 6, 2), (3, 1, 0), (1, 1, 1))
    print(canonicalize(CSS_group, X0, Z0))
    CSS_stabs = find_stabilizers(CSS_group, Z0, X0)
    print(f"Your CSS stabs are:")
    for stab in CSS_stabs[0]:
        print(symp2Pauli(stab, n))

    # Make some non-CSS codes and check if they are CSS
