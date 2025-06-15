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

from util import find_strides

def canonicalize_with_mult(group, z0, x0, mult):
    """
    Quaternary canonicalizer. This accepts the sets z0 and x0 and returns a guess for the canonical z0 and x0. Applies the passed isomorhism and computes the X-shift by shifting the second argument by 2 * g.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.
        * mult (np.ndarray): 1D Array of the same shape as group, containing entries all relatively prime to the corresponding group size. These represent the automorhism we need to apply before finding the X-shift.

    Returns:
        * 2d array containing the z0 and x0 guesses in one stack
    """
    #apply automorphism
    z0 = np.mod(z0 * mult, group)
    x0 = np.mod(x0 * mult, group)
    subtract = []
    #for each group, the amount you subtract is either all of the corresponding term in x0[0], or it is almost all if the group size is even and x0[0, i] is odd
    #this gets called a lot, and builds an array of a known size element by element, so a lot can probably be saved here
    for i, k in enumerate(group):
        if k % 2 == 0:
            subtract += [2 * (x0[0, i] // 2)]
        else:
            subtract += [x0[0, i]]
    x0 = np.mod(x0 - subtract, group)
    return np.vstack([z0, x0])

def canonicalize_with_order(group, z0, x0):
    """
    Tertiary canonicalizer. This accepts the sets z0 and x0 and returns a guess for the canonical z0 and x0, without changing the order of the terms. Applies only isomorphisms and X-shifts.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.

    Returns:
        * 2d array containing the z0 and x0 guesses in one stack
    """
    #start by shifting z0[0] to be 0
    z0 = np.array(z0)
    x0 = np.array(x0)
    x0 = np.mod(x0 - z0[0], group)
    z0 = np.mod(z0 - z0[0], group)
    assert max(z0[0]) == 0

    #find automorphisms
    mults = [0] * len(group)
    #for each group dimension...
    for i, k in enumerate(group):
        candidates = set()
        #find all automorphisms of that group... (numbers relatively prime to group size)
        for j in range(k):
            if np.gcd(j, k) == 1:
                candidates.add(j)
        #and then for each term in z0...
        for j in range(1, len(z0)):
            m = min(np.array(list(candidates)) * z0[j, i] % k)
            #only keep the automorphisms that minimize the corresponding component of that term in z0 (out of the remaining candidates)
            #this implementation is probably really slow
            for l in candidates.copy():
                if z0[j, i] * l % k != m:
                    candidates.remove(l)
        mults[i] = list(candidates)

    #for each combination of automorphisms, find the canonical form by just shifting
    sets = []
    for i in it.product(*mults):
        new = [canonicalize_with_mult(group, z0, x0, i)]
        if len(sets) == 0:
            sets = np.array(new)
            continue
        sets = np.vstack([sets, new])
    #use numpy black magic to sort nd array and take the first element
    return sets[np.lexsort(sets.reshape(len(sets), -1).T[::-1])[0]]

def canonicalize_perms(group, z0, x0):
    """
    Secondary canonicalizer. This accepts the sets z0 and x0 and returns a guess for the canonical z0 and x0. Applies all tricks other than swapping z0 and x0. Various equivalences that this tries are reordering Z0 and X0, applying an automorphism to any component group, and shifting the X0's by 2g, for some g.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.

    Returns:
        * 2d array containing the z0 and x0 guesses in one stack
    """
    #first, find all canonical forms (isomorphisms and shifts only) over all possible permutations of z0 and x0
    results = []
    for i in it.permutations(z0):
        for j in it.permutations(x0):
            new = [canonicalize_with_order(group, i, j)]
            if len(results) == 0:
                results = new
                continue
            results = np.vstack([results, new])
    #use numpy black magic to sort nd array and take the first element
    return results[np.lexsort(results.reshape(len(results), -1).T[::-1])[0]]

def canonicalize(group, z0, x0):
    """
    Main canonicalizer. This accepts the sets z0 and x0 and returns the canonical isomorphic z0 and x0. The canonical form over equivalent codes is not unique. Here are some guarantees about this function. z0[0] will only contain 0s. The arrays z0[0], z0[1], z0[2], ... will be sorted. The same is true for the arrays in x0. z0[1] will only contain entries x such that x is a divisor of the group size (0 counts). x0[1] will only contain 0 or 1 entries, with 1 entries only being acceptable if the corresponding group size is even. Various equivalences that this tries are swapping Z0 and X0, reordering Z0 and X0, applying an automorphism to any component group, and shifting the X0's by 2g, for some g.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.

    Returns:
        * Tuple containinng two elements. The first is the canonical version of z0, the second is the canonical version of x0.
    """
    group = np.array(group)
    z0 = np.array(z0)
    x0 = np.array(x0)

    #make sure that z0 is the shorter one
    if len(z0) > len(x0):
        return canonicalize(group, x0, z0)

    #case where z0 has fewer terms than x0
    if len(z0) < len(x0):
        result = canonicalize_perms(group, z0, x0)
        return result[:len(z0)], result[len(z0):]

    #common case, where lengths are equal
    #finds two options for the canonical form, one where z0 and x0 are switched, each iterating over automorphisms and permutations inside each of z0 and x0
    options = np.array([canonicalize_perms(group, z0, x0),
                        canonicalize_perms(group, x0, z0)])
    #numpy black magic to sort nd array and take the first element
    options = options[np.lexsort(options.reshape(2, -1).T[::-1])[0]]
    #options contains all of z0 and x0 in one list, so we split it up
    return options[:len(z0)], options[len(z0):]

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
    Utility function. Given two sets of arrays, finds all possible differences between an element of one array and an element of the other.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * a (np.ndarray): 2D Array. The first array of the two whose differences we will be finding. It is a list of all the elements of the group in the first array. Number of columns should match length of group.
        * b (np.ndarray): 2D Array. The second array of the two whose differences we will be finding. It is a list of all the elements of the group in the first array. Number of columns should match length of group.
    
    Returns:
        * 2D numpy array containing all the elements of the group that are the difference between something in a and something in b.
    """
    s = []
    group = np.array(group)
    a = np.array(a)
    b = np.array(b)
    for i in a:
        for j in b:
            new = np.mod(j - i, group)
            if len(s) == 0:
                s = [new]
                continue
            s = np.vstack([s, new])

    #throw away duplicates
    return np.unique(s, axis = 0)

def css_flips(n, group, z0, x0, strides):
    """
    Compute the stabilizer tableau of a code. Tableau is returned in symplectic form with the convention [Z | X]. Automatically checks if the code is CSS and converts the tableau to CSS form if so.

    Params:
        * n (int): Number of qubits. Should be the product of group. Passed for simplicity
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.
        * strides (np.ndarray): 1D Array of strides of the group. It is the backwards cumulative product of the group sizes. This means that the ith entry tells us by how much we need to increment the index of a group element to increase the ith entry of the group element by 1. Example: if the group is [4, 7, 3], strides should be [21, 3, 1]. This could be computed from group, but is passed for simplicity.

    Returns:
        * Tuple containing two terms. The first is a bool which is True iff the code passed was CSS. The second is a numpy array of qubits (in tuple / array form) which need to be hadamarded in order to make the code CSS. The second argument is [] if the first is False.
    """
    #build sets containing differences between two qubits with the same pauli on them, then combine the sets
    zz = build_set(group, z0, z0)
    xx = build_set(group, x0, x0)
    same_diffs = np.unique(np.vstack([zz, xx]), axis = 0)

    #build set of differences of qubits with different paulis on them
    zx = build_set(group, z0, x0)

    #for each difference between two elements of the same set...
    flips = [np.zeros(len(group), np.int64)]
    for g in same_diffs:
        #find the group generated by it...
        gen_g = [g]
        cur = g
        while np.max(cur) > 0:
            cur = np.mod(cur + g, group)
            gen_g = np.vstack([gen_g, cur])
        cur_flips = flips.copy()
        #and use those groups to generate the full group of things connected by steps between qubits that must be in the same block (of the hadamarded and non-hadamarded blocks)
        for i in cur_flips:
            for j in gen_g:
                flips = np.vstack([flips, np.mod(i + j, group)])
        flips = np.unique(flips, axis = 0)
        #if all the qubits lie in the same block, the code cannot be CSS
        if len(flips) == n:
            return False, []

    #the full sets of differences between qubits in the different blocks is given by the differences between Z0 and X0 PLUS any element of the group times 2, since this generates differences between elements of Z0 + g and X0 - g
    #find full set of 2 * g for all g in the group
    group_times_two = []
    for g in it.product(*[range(a) for a in group]):
        new = np.mod(2 * np.array(g), group)
        if len(group_times_two) == 0:
            group_times_two = [new]
            continue
        group_times_two = np.vstack([group_times_two, new])
    group_times_two = np.unique(group_times_two, axis = 0)

    #find full set of differences between hadamarded and non hadamarded differences.
    bad = set()
    for i in zx:
        for j in group_times_two:
            bad.add(np.mod(i + j, group) @ strides)
    for i in flips:
        if i @ strides in bad:
            return False, []
    return True, flips


def find_stabilizers(group, z0, x0):
    """
    Compute the stabilizer tableau of a code. Tableau is returned in symplectic form with the convention [Z | X]. Automatically checks if the code is CSS and converts the tableau to CSS form if so.

    Params:
        * group (np.ndarray): 1D Array of all the group sizes in the factored version of the group. Also accepts tuple.
        * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the Z weight. Also accepts tuples at any level.
        * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed along the group dimensions. Number of columns should match length of group and not exceed the terms of group. Number of rows is the X weight. Also accepts tuples at any level.
    
    Returns:
        * 2D numpy array in symplectic form (Z|X) with the stabilizers of the code. Returns n stabilizers, so contains linearly dependent checks. Should automatically be in CSS form if code is CSS.
    """
    #convert to numpy arrays
    group = np.array(group, np.int64)
    z0 = np.array(z0, np.int64)
    x0 = np.array(x0, np.int64)
    
    #compute n and define d as shorthand, the number of groups in the product
    n = int(np.prod(group))
    d = len(group)

    #compute strides, the number by which the index must go up to increment a particular index. Example: if the group is [4, 7, 3], strides is [21, 3, 1], because incrementing the index by 21 increments the first component of the qubit array exactly.
    strides = np.zeros(d, np.int64)
    strides[:-1] = np.cumprod(group[::-1])[d - 2::-1]
    strides[-1] = 1
    stabilizers = np.zeros((n, 2 * n), dtype = np.uint8)

    #if can_flip is true, the code is CSS and flips contains the qubits that need to be hadamarded (the format of these is not indices, but arrays
    can_flip, flips = css_flips(n, group, z0, x0, strides)

    #iterate over all qubits / all tuples in the group / all stabilizers
    for i, g in enumerate(it.product(*[range(a) for a in group])):
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1

    #flip qubits that need hadamarding if code is css
    if can_flip:
        for g in flips:
            index = g @ strides
            stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]] 
    return stabilizers

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
