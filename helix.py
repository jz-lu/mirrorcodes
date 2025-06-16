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

def canonicalize_perms(group, z0, x0, isos, strides):
    """
    Secondary canonicalizer. This accepts the sets z0 and x0 and returns a guess
    for the canonical z0 and x0. Applies all tricks other than swapping z0 and x0.
    Various equivalences that this tries are reordering Z0 and X0, applying an
    automorphism to any component group, and shifting the X0's by 2g, for some g.

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
        * isos (np.ndarray): list of lists containing isomorphisms of group
        * strides (np.ndarray): strides for indexing qubits

    Returns:
        * Tuple containinng two elements. The first is the canonical version of z0,
          the second is the canonical version of x0.
    """
    n = np.prod(group)

    #initialize variables to do indexing and hold current min
    z0 = np.array(z0)
    x0 = np.array(x0)
    min_z, min_x = z0, x0
    z_combiner = find_strides([n] * len(z0))
    x_combiner = find_strides([n] * len(x0))
    min_z_index = z_combiner @ z0 @ strides
    min_x_index = x_combiner @ x0 @ strides
    max_x_index = n ** len(x0)

    #check all permutations for z
    for i in it.permutations(z0):
        #shift first term to 0
        new_z_shuffled = np.mod(i - i[0], group)
        #check all isomorphisms
        for j in it.product(*isos):
            new_z_isoed = np.mod(new_z_shuffled * j, group)
            new_z_index = z_combiner @ new_z_isoed @ strides
            #new z is larger, skip
            if new_z_index > min_z_index:
                continue
            #new z is smaller, set z min and set x index to be above max value
            if new_z_index < min_z_index:
                min_z = new_z_isoed
                min_z_index = new_z_index
                min_x_index = max_x_index
            #check all x permutations
            for k in it.permutations(np.mod(x0 - i[0], group)):
                #apply isomorphism and shift
                new_x_isoed = shift_X(group, np.mod(np.array(k) * j, group))
                new_x_index = x_combiner @ new_x_isoed @ strides
                if new_x_index > min_x_index:
                    continue
                #found a new minimum, write it down
                min_x_index = new_x_index
                min_x = new_x_isoed
    return min_z, min_x


def canonicalize(group, z0, x0):
    """
    Main canonicalizer. This accepts the sets z0 and x0 and returns the canonical
    isomorphic z0 and x0. The canonical form over equivalent codes is not unique.
    Here are some guarantees about this function. z0[0] will only contain 0s. The
    arrays z0[0], z0[1], z0[2], ... will be sorted. The same is true for the arrays
    in x0. z0[1] will only contain entries x such that x is a divisor of the group
    size (0 counts). x0[1] will only contain 0 or 1 entries, with 1 entries only
    being acceptable if the corresponding group size is even. Various equivalences
    that this tries are swapping Z0 and X0, reordering Z0 and X0, applying an
    automorphism to any component group, and shifting the X0's by 2g, for some g.

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

    Returns:
        * Tuple containinng two elements. The first is the canonical version of z0,
          the second is the canonical version of x0.
    """
    #make sure that z0 is the shorter one
    if len(z0) > len(x0):
        return canonicalize(group, x0, z0)

    n = np.prod(group)
    group = np.array(group)
    z0 = np.array(z0)
    x0 = np.array(x0)
    isos = find_isos(group)
    strides = find_strides(group)
    
    #case where z0 has fewer terms than x0
    if len(z0) < len(x0):
        return canonicalize_perms(group, z0, x0, isos, strides)

    #common case, where lengths are equal
    #finds two options for the canonical form, one where z0 and x0 are switched,
    #each iterating over automorphisms and permutations inside each of z0 and x0
    z1, x1 = canonicalize_perms(group, z0, x0, isos, strides)
    z2, x2 = canonicalize_perms(group, x0, z0, isos, strides)
    z_combiner = find_strides([n] * len(z0))
    x_combiner = find_strides([n] * len(x0))
    z1_index = z_combiner @ z1 @ strides
    z2_index = z_combiner @ z2 @ strides
    x1_index = x_combiner @ x1 @ strides
    x2_index = x_combiner @ x2 @ strides
    if z1_index < z2_index or (z1_index == z2_index and x1_index < x2_index):
        return z1, x1
    return z2, x2


def is_Z_canonical(group, z0, isos):
    """
    Canonicalization checker for just Z's. Does not check legality. Merely checks
    whether it can find a smaller equivalent z0 instance.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the Z weight.
          Also accepts tuples at any level.
        * isos (list of lists): list of all isomorphisms of each factor of group

    Returns:
        * Boolean with whether z0 is canonical.
    """
    group = np.array(group)
    z0 = np.array(z0)
    
    #sorted index calculation
    strides = find_strides(group)
    combiner = find_strides([np.prod(group)] * len(z0))
    z0_index = combiner @ z0 @ strides
    
    #check over all permutations and isomorphisms
    for i in it.permutations(z0):
        shifted = np.mod(i - i[0], group)
        for j in it.product(*isos):
            if combiner @ np.mod(shifted * j, group) @ strides < z0_index:
                return False
    return True


def is_X_canonical(group, x0, isos):
    """
    Canonicalization checker for just X's. Does not check legality. Merely checks
    whether it can find a smaller equivalent x0 instance.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored
          version of the group. Also accepts tuple.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
          along the group dimensions. Number of columns should match length of
          group and not exceed the terms of group. Number of rows is the X weight.
          Also accepts tuples at any level.
        * isos (list of lists): list of all isomorphisms we wish to consider

    Returns:
        * Boolean with whether z0 is canonical.
    """
    group = np.array(group)
    x0 = np.array(x0)
    
    #sorted index calculation
    strides = find_strides(group)
    combiner = find_strides([np.prod(group)] * len(x0))
    x0_index = combiner @ x0 @ strides
    
    #check over all permutations and isomorphisms
    for i in it.permutations(x0):
        shifted = np.mod(i - np.array([j if k % 2 == 1 else 2 * (j // 2)
                                       for j, k in zip(i[0], group)]), group)
        for j in it.product(*isos):
            if combiner @ np.mod(shifted * j, group) @ strides < x0_index:
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

    #throw away duplicates
    return np.unique(s, axis = 0)


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
    #build sets containing differences between two qubits with the same pauli on
    #them, then combine the sets
    same_diffs = np.unique(np.vstack([build_set(group, z0, z0),
                                      build_set(group, x0, x0)]), axis = 0)

    #build set of differences of qubits with different paulis on them
    zx = build_set(group, z0, x0)

    #for each difference between two elements of the same set...
    flips = np.zeros((1, len(group)), dtype = int)
    for g in same_diffs:
        #find the group generated by it...
        gen_g = [g]
        cur = g
        while np.max(cur) > 0:
            cur = np.mod(cur + g, group)
            gen_g.append(cur)
        cur_flips = flips.copy()
        #and use those groups to generate the full group of things connected by
        #steps between qubits that must be in the same block (of the hadamarded
        #and non-hadamarded blocks)
        for i in cur_flips:
            flips = np.append(flips, np.mod(i + gen_g, group), axis = 0)
        flips = np.unique(flips, axis = 0)
        #if all the qubits lie in the same block, the code cannot be CSS
        if len(flips) == n:
            return False, []

    #the full sets of differences between qubits in the different blocks is given
    #by the differences between Z0 and X0 PLUS any element of the group times 2,
    #since this generates differences between elements of Z0 + g and X0 - g
    #find full set of 2 * g for all g in the group
    #find full set of differences between hadamarded and non hadamarded differences.
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
    #convert to numpy arrays
    group = np.array(group, np.int64)
    z0 = np.array(z0, np.int64)
    x0 = np.array(x0, np.int64)
    
    #compute n and define d as shorthand, the number of groups in the product
    n = int(np.prod(group))
    d = len(group)

    #compute strides, the number by which the index must go up to increment a
    #particular index. Example: if the group is [4, 7, 3], strides is [21, 3, 1],
    #because incrementing the index by 21 increments the first component of the
    #qubit array exactly.
    strides = find_strides(group)
    stabilizers = np.zeros((n, 2 * n), dtype = np.uint8)

    #if can_flip is true, the code is CSS and flips contains the qubits that need
    #to be hadamarded (the format of these is not indices, but arrays
    can_flip, flips = css_flips(group, z0, x0)

    #iterate over all qubits / all tuples in the group / all stabilizers
    for i, g in enumerate(it.product(*[range(a) for a in group])):
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1

    #flip qubits that need hadamarding if code is css
    if can_flip:
        for g in flips:
            index = g @ strides
            stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]] 
    return stabilizers, can_flip


class HelixCode():
    def __init__(self, group, z0, x0, n=None, k=None, d=None, is_css=None):
        self.group = group
        self.z0 = z0
        self.x0 = x0
        self.stabilizers = None
        self.stim_tableau = None
        self.CSS = is_css

        self.n = n
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
            self.n = np.prod(self.group)
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
    
    def is_CSS(self):
        if self.CSS is None:
            self.stabilizers, self.CSS = find_stabilizers(self.group, self.z0, self.x0)
        return self.CSS

    def get_rate(self):
        return self.get_k() / self.get_n()
    
    def get_rel_dist(self):
        return self.get_d() / self.get_n()


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
    for stab in CSS_stabs:
        print(symp2Pauli(stab, n))

    # Make some non-CSS codes and check if they are CSS
