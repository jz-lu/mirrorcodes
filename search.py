"""
`search.py`
Code file for conducting a search for good mirror codes.
This consists of two steps:
    1) Generate a mirror code according to some systematic protocol.
    2) Evaluate if the code is good. If so, keep it. If not, delete and continue.
We can start by searching through codes for which n ~ 100 +/- 100, and check
weight <= 6.
But let n >> check weight so that the LDPCness kicks in, e.g. n >= 30.

The precise meaning of "good" is debatable, but we will adopt the following
two-stage filtering method.
Stage 1 (distance-rate tradeoff):
    * Evaluate the rate R of the code. If R < 1/16, discard.
    * Evaluate the distance d of the code. If evaluation of the distance takes >3
      min, keep the code.
    * If the distance is calculated successfully, discard if Rd < 1/2. Keep
      otherwise.

Stage 2 (practicality):
    * Evaluate the pseudo-threshold using BP-OSD. If it is above some TBD cutoff,
      keep.
    * Evaluate the circuit distance?
"""
import itertools as it
import numpy as np
from primefac import primefac

from distance import distance
from isomorphism import automorphisms_fixing_vectors, lex_minimal_vectors, push_to_lex_minimal
from mirror import canonicalize, find_stabilizers, is_X_canonical, is_Z_canonical, MirrorCode
from util import binary_rank, compute_rank_from_tuples, find_isos, find_strides, \
                 gcd, index_to_array, partitions


def is_canonical(mirror_code):
    """
    Check whether `mirror_code` is in canonical form.
    We say that a mirror code (group, Z0, X0) is in canonical form if it is the
    lexicographically earliest representative among all codes equivalent under:
        1) swapping Z0 and X0 (compared only when lengths are equal),
        2) reordering within Z0 and X0,
        3) torsor shift (Z, X)->(Z+p, X+p),
        4) identity-choice shift (Z, X)->(Z+h, X-h) so jointly (Z, X)->(Z+p, X+p+2q),
        5) component-group automorphisms: per-component units AND permutations of
           equal-size cyclic factors (reindexing indistinguishable factors).

    Input:
        * mirror_code (tuple): Tuple of (group, Z0, X0) where group is a tuple
          and Z0 and X0 are sets
    
    Returns:
        * Binary indication of whether `mirror_code` is in canonical form.
    """
    return mirror_code == canonicalize(*mirror_code)


def num_indices(n, Z_wt, X_wt):
    """
    Compute number of possible codes on n qubits with given weights. This does not
    assume any canonicalization whatsoever, other than setting Z0[0] to 0. This
    function exists to have nice ranges over which to index. This returns the
    number of partitions of the powers of n times n to the power of Z_wt + X_wt - 1.

    Params:
        * n (int): The number of qubits we want to search over
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
    
    Returns:
        * An int with the number of possible codes on n qubits. Calculates the
        number of partitions of the powers of n times number of ways to pick
        qubits in Z0 and X0, not counting Z0[0].
    """
    result = 1
    #find all the powers of primes in n, and count how many partitions they have
    for i in it.groupby(list(primefac(n))):
        result *= len(list(partitions(len(list(i[1])))))
    return result * n ** (Z_wt + X_wt - 1)


def multi_index_to_tuples(group, Z_wt, X_wt, index):
    """
    Compute several tuples representing some qubits in group, which correspond to
    the values in Z0 and X0. This is done given an index
    from 0 to n ** (Z_wt + X_wt - 1) - 1. This is no different than expressing
    several numbers "base group". Notably, this works for larger indices too, but
    will only consider the index mod n ** (Z_wt + X_wt). Regardless of what the
    index says, Z0[0] is force-set to the all 0's tuple.

    Params:
        * group (np.ndarray): the group we are decomposing the index into
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index (int): A number from 0 to n ** (Z_wt + X_wt - 1) - 1 corresponding
          to several tuples mod group.
    
    Returns:
        * Four lists. The first has length Z_wt and is a list of tuples containing
          the values of Z0 as given by index (after dividing by n ** X_wt) with a
          tuple of zeros prepended. The second has length X_wt and is a list of
          tuples containing the values of X0 as given by index (after modding by
          n ** X_wt). The third and fourth lists contain the indices used to
          construct the tuples of the first and second lists, respectively. Each
          entry in the third and fourth lists is from 0 to n - 1.
    """
    n = int(np.prod(group))
    Zs = []
    Xs = []
    Z_nums = []
    X_nums = []

    #build Xs list
    for _ in range(X_wt):
        Xs.append(index_to_tuple(group, index))
        X_nums.append(index % n)
        index /= n
        index = int(index)

    #build Zs list
    for _ in range(Z_wt - 1):
        Zs.append(index_to_tuple(group, index))
        Z_nums.append(index % n)
        index /= n
        index = int(index)

    #prepend Z0[0]
    Zs.append((0,) * len(group))
    Z_nums.append(0)

    return Zs[::-1], Xs[::-1], Z_nums[::-1], X_nums[::-1]


def n_partitions(n):
    """
    Find all non-isomorphic abelian groups of n qubits.

    Params:
        * n (int): The number of qubits
    
    Returns:
        * A list of tuples, each containing the sizes of various abelian groups,
          such that the product of all the groups is n. Each group must be a power
          of a prime.
    """
    primes = []
    powers = []

    #find prime factorization of n
    for i in it.groupby(list(primefac(n))):
        primes.append(i[0])
        powers.append(len(list(i[1])))

    #find all partitions of the powers in the prime factorization of n
    combos = [list(partitions(i)) for i in powers]

    #find all combinations of decompositions of the various prime powers according
    #to the above partitions
    result = []
    for i in it.product(*combos):
        group_sizes = []
        for j, p in zip(i, primes):
            for k in j:
                group_sizes.append(p ** k)
        result.append((*group_sizes,))
    
    return result


def process_codes(n, Z_wt, X_wt, index_start = 0, index_end = None):
    """
    Wrapper for processing only codes which are canonical. Currently returns the
    number of valid codes.

    Params:
        * n (int): The number of physical qubits of the desired codes
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index_start (int, optional): The index at which the loop should start
          counting codes. Indices can range from 0 to n ** (Z_wt + X_wt - 1) - 1. 0
          by default. 
        * index_end (int, optional): The index at which the loop should stop
          counting codes. The code at index_end is not counted. Indices can range
          from 0 to n ** (Z_wt + X_wt - 1) - 1. None by default, which is then set
          to the max value. 
    
    Returns:
        * Number of canonical codes with n qubits.
    """
    #find max index
    if index_end == None:
        index_end = num_indices(n, Z_wt, X_wt)

    #find all group partitions
    groups = n_partitions(n)

    index = index_start
    COUNTER = 0
    #for each index...
    while index < index_end:
        quotient, remainder = divmod(index, n ** (Z_wt + X_wt - 1))
        #find the relevant group
        group = groups[quotient]
        #find the values of Z0 and X0 from the index
        Zs, Xs, Z_nums, X_nums = multi_index_to_tuples(group, Z_wt, X_wt, remainder)
        jump =  -1
        #for each value of Z0 and X0, other than Z0[0]...
        for i in range(1, Z_wt + X_wt):
            #see if there are any immediate mistakes
            #mistakes include not being in order, or Z0[1] not consisting of
            #divisors of the group size, or X0[0] not consisting of 0's or 1's
            #where appropriate
            if (any([j > 0 and g % j > 0 for j, g in zip(Zs[1], group)]) or
                (i < Z_wt and Z_nums[i] <= Z_nums[i - 1]) or
                (i > Z_wt and X_nums[i - Z_wt] <= X_nums[i - Z_wt - 1]) or
                (i == Z_wt and (max(Xs[0]) > 1 or
                    any([j > 0 and g % 2 > 0 for j, g in zip(Xs[0], group)])))):
                #compute by how much the index should be incremented based on the
                #position of the mistake
                jump = n ** (Z_wt + X_wt - 1 - i)
                break
        group = np.array(group)
        Zs = np.array(Zs)
        Xs = np.array(Xs)
        #if this is a valid code candidate, check that it is canonical
        if jump < 0:
            canon_Z, canon_X = canonicalize(group, Zs, Xs)
            if not (np.all(Zs == canon_Z) and np.all(Xs == canon_X)):
                jump = 1
        #if code is not canonical, increment the index
        if jump >= 0:
            index = (index // jump + 1) * jump
            continue
        #if we have reached this point, the code is canonical and should be counted
        COUNTER += 1
        index += 1
    return COUNTER


def decomposed_Z0_candidates(Z_wt, size):
    """
    Finds possible elements of Z0[:, k] that are worth searching over for some k.

    Params:
        * Z_wt (int): The number of terms in Z0
        * size (int): The size of the cyclic group we are finding candidates for
    
    Returns:
        * List of tuples. Each tuple contains a list of entries of Z0[:, k] and
          the gcd mod size of the elements and the array size, needed for computing
          the list of isomorphisms that leaves them invariant.
    """
    #compute possible indices. first is always 0, second is always a divisor, etc
    candidates = [[0] for _ in range(Z_wt)]
    candidates[1] = [i for i in range(size) if i == 0 or size % i == 0]
    for i in range(2, Z_wt):
        candidates[i] = range(size)

    #compute isomorphisms that leave indices unchanged
    isos_by_gcd = [[] for _ in range(size)]
    for i in range(1, size):
        if size % i > 0:
            continue
        isos_by_gcd[i % size] = [j for j in range(1, size) 
                                 if (j - 1) % (size / i) == 0 and gcd(j, size) == 1]

    #check to make sure each candidate is minimal under isomorphisms and append
    result = []
    isos = find_isos(size)
    strides = find_strides([size] * Z_wt)
    for zs in it.product(*candidates):
        z_list = np.array(zs)
        if z_list @ strides == min(
                [np.mod(z_list * i, size) @ strides for i in isos]):
            result.append((z_list, gcd(*zs, size) % size))
    return result


def decomposed_X0_candidates(X_wt, size):
    """
    Finds possible elements of X0[:, k] that are worth searching over for some k.

    Params:
        * X_wt (int): The number of terms in X0
        * size (int): The size of the cyclic group we are finding candidates for
    
    Returns:
        * 3d array. The outer dimension is loops over all values from 0
          to size, inclusive. Only the entries which are factors of size have any
          elements. The second dimension is just listing candidates. The inner
          dimension loops over the terms of X0, and thus has length X_wt.
    """
    #compute possible indices. first is either 0 or 1 if size is even
    candidates = [[0] if size % 2 == 1 else [0, 1] for _ in range(X_wt)]
    for i in range(1, X_wt):
        candidates[i] = range(size)

    #compute isomorphisms that might leave Z indices unchanged
    isos_by_gcd = [[] for _ in range(size)]
    for i in range(1, size + 1):
        if size % i > 0:
            continue
        isos_by_gcd[i % size] = [j for j in range(1, size)
                                 if (j - 1) % (size / i) == 0 and gcd(j, size) == 1]

    #check to make sure each candidate is minimal under isomorphisms and append
    result = [[] for _ in range(size + 1)]
    strides = find_strides([size] * X_wt)
    for i in range(size):
        if len(isos_by_gcd[i]) == 0:
            continue
        for xs in it.product(*candidates):
            x_list = np.array(xs)
            if x_list @ strides == min(
                    [np.mod(x_list * j, size) @ strides for j in isos_by_gcd[i]]):
                result[i].append(x_list)
    return result


def build_Z0_candidates(Z_wt, group, min_k = 3):
    """
    Finds possible tuples Z0 that are worth searching over.

    Params:
        * Z_wt (int): The number of terms in Z0
        * group (np.ndarray): The group whose codes we are finding
        * min_k (int, optional): Whether codes should be filtered to exclude those with k < min_k
    
    Returns:
        * List of Z0 candidates with possible isomorphisms. Each candidate is a
          tuple containing (Z0, isomorphisms). Each Z0 is a sorted list/tuple of
          lists/tuples mod group. Each Z0[0] is always all zeros, and each Z0[1]
          only contains terms which divide the group size. Each candidate
          isomorphisms is a index of gcds for each dimension, communicating
          potential isomorphisms which canonicalize Z0 as much as possible.
    """
    n = np.prod(group)
    d = len(group)
    
    #compute possible indices. first is always 0, second is always a divisor, etc
    candidates = np.zeros((Z_wt, d, 1))
    for g, k in enumerate(group):
        candidates[1, k] = [i for i in range(g) if i == 0 or g % i == 0]
        for i in range(2, Z_wt):
            candidates[i, k] = range(g)
    candidates = [it.product(*c) for c in candidates]
    
    indices = find_strides(group)
    autos = list(list_automorphisms(group))

    result = []
    for z in it.product(*candidates):
        v = z @ indices
        if v != sorted(v):
            continue
    


    lengths = [len(c) for c in candidates]
    #index incrementing amount for each index
    strides = find_strides(lengths)
    
    gcd_strides = find_strides(group)

    max_index = np.prod(lengths)
    index_val = 0
    result = []
    isos = find_isos(group)
    odd_indices = [i for i in range(d) if group[i] % 2 == 1]
    #for each valid combination from each subgroup
    while index_val < max_index:
        jump = -1
        index = index_to_tuple(lengths, index_val)
        zs = np.zeros((Z_wt, d), dtype = int)
        z_indices = None
        #find first instance of an index sorted out of order...
        for i in range(d):
            zs[:, i] = candidates[i][index[i]][0]
            z_indices = zs @ strides
            if np.any(z_indices[:-1] > z_indices[1:]):
                jump = strides[i]
                break
        #and increment there if it is not canonical
        if (jump > 0 or np.any(z_indices[:-1] >= z_indices[1:])
            or not is_Z_canonical(group, zs, isos)
            or (min_k and (not np.any(zs[:, odd_indices])
                or compute_rank_from_tuples(group, zs) > n - min_k))):
            jump = max(jump, 1)
            index_val = (index_val // jump + 1) * jump
            continue
        #otherwise, write down the code and the gcds
        result.append((zs, int(np.array([candidates[i][index[i]][1]
                                         for i in range(d)]) @ gcd_strides)))
        index_val += 1
    return result


def build_X0_candidates(X_wt, group, min_k = 3):
    """
    Finds possible tuples X0 that are worth searching over. Split up by the gcd of
    the Zs, which defines isomorphisms under which the Xs are minimal.

    Params:
        * X_wt (int): The number of terms in X0
        * group (np.ndarray): The group whose codes we are finding
        * min_k (int, optional): Whether codes should be filtered to exclude those with k < min_k
    
    Returns:
        * List of lists of X0 candidates. The outer list is indexed by the gcd's of
          the Zs which determine the isomorphisms under which the Zs are equivalent,
          meaning we must minimize the Xs over this set. Each X0 is a sorted
          list/tuple of lists/tuples mod group. Each X0[0] is always all 0 or 1.
    """
    n = np.prod(group)
    d = len(group)
    candidates = [decomposed_X0_candidates(X_wt, i) for i in group]
    result = [[] for i in range(n)]
    for i in range(n):
        gcds = index_to_tuple(group, i)
        lengths = [len(candidates[j][gcds[j]]) for j in range(d)]
        
        #index incrementing amount for each index
        strides = find_strides(lengths)
        max_index = np.prod(lengths)
        if max_index == 0:
            continue
        isos = [[k for k in range(1, g) if (k - 1) %
                 (1 if gcds[j] == 0 else g / gcds[j]) == 0 and gcd(k, g) == 1]
                for j, g in enumerate(group)]
        odd_indices = [i for i in range(d) if group[i] % 2 == 1]
        index_val = 0
        #for each valid combination from each subgroup
        while index_val < max_index:
            jump = -1
            index = index_to_tuple(lengths, index_val)
            xs = np.zeros((X_wt, d), dtype = int)
            #find first instance of an index sorted out of order...
            for j in range(d):
                xs[:, j] = candidates[j][gcds[j]][index[j]]
                x_indices = xs @ strides
                if np.any(x_indices[:-1] > x_indices[1:]):
                    jump = strides[j]
                    break
            #and increment there if it is not increasing
            if (jump > 0 or np.any(x_indices[:-1] >= x_indices[1:])
                or not is_X_canonical(group, xs, isos)
                or (min_k > 0 and (not np.any(xs[:, odd_indices]) or
                                     compute_rank_from_tuples(group, xs) > n - min_k))):
                jump = max(jump, 1)
                index_val = (index_val // jump + 1) * jump
                continue
            #otherwise, write down the code
            result[i].append(xs)
            index_val += 1
    return result


def minimal_strings_for_subgroup(Z_wt, X_wt, subgroup):
    p = primefac(subgroup[0])[0]
    result = np.array([])
    candidates = np.array([[[0] * len(subgroup)]])
    vec_indices = [0]
    strides = find_strides(subgroup)
    while True:
        if vec_indices[-1] >= len(candidates[-1]):
            if len(vec_indices) == 1:
                break
            vec_indices = vec_indices[:-2] + [vec_indices[-2] + 1]
            candidates = candidates[:-1]
            continue
        Z_vals = np.array([candidates[i, val] for i, val in enumerate(vec_indices[:min(len(vec_indices), Z_wt)])]) @ strides
        if len(set(Z_vals)) < len(Z_vals):
            vec_indices[-1] += 1
            continue
        if len(vec_indices > Z_wt):
            X_vals = np.array([candidates[i + Z_wt, val] for i, val in enumerate(vec_indices[Z_wt:len(vec_indices)])]) @ strides
            if len(set(X_vals)) < len(X_vals):
                vec_indices[-1] += 1
                continue
        if len(vec_indices) < Z_wt + X_wt:
            if len(vec_indices) == 1:
                candidates = np.append(candidates, [lex_minimal_vectors(subgroup)], axis = 0)
            else:
                candidates = np.append(candidates, [[]], axis = 0)
                isos = automorphisms_fixing_vectors(subgroup, [candidates[i, val] for i, val in enumerate(vec_indices)])
                strides = find_strides(subgroup)
                for i in range(np.prod(subgroup)):
                    v = index_to_array(subgroup, i)
                    if len(vec_indices) == Z_wt and max(v) > (1 if p == 2 else 0):
                        if p > 2:
                            break
                    elif i <= min(np.array([np.mod(a @ v, subgroup) for a in isos]) @ strides):
                        candidates[-1] = np.append(candidates[-1], [v], axis = 0)
            vec_indices += [0]
        else:
            result += np.ndarray.copy(np.array([candidates[i, val] for i, val in enumerate(vec_indices)]))
            vec_indices[-1] += 1
    return result


def permutation_bins(Z_wt, X_wt, subgroup, perm, candidates):
    p = primefac(subgroup[0])[0]
    result = []
    strides = find_strides(subgroup)
    for cand in candidates:
        signs = [] #ADD TRACKING OF SIGNS. ADD IGNORING OF TRIVIAL ISOMORPHISM
        c = np.ndarray.copy(cand[perm])
        c -= c[0]
        min_1 = push_to_lex_minimal(subgroup, c[1])
        val = strides @ (min_1 - cand[1])
        if val != 0:
            result[0 if val < 0 else 2] += [cand]
            continue
        isos = []
        for i in range(2, Z_wt + X_wt):
            if i == 2:
                isos = automorphisms_fixing_vectors(subgroup, c[1:i])
            else:
                isos = np.array([iso for iso in isos if np.mod(iso @ c[i - 1], subgroup) == c[i - 1]])
            values = np.array([np.mod(iso @ c[i], subgroup) for iso in isos])
            if i == Z_wt:
                values %= 2 if p == 2 else 1
            diff = min(values @ strides) - strides @ cand[i]
            if diff != 0:
                result[0 if diff < 0 else 2] = np.append(result[0 if diff < 0 else 2], [cand], axis = 0)
                break
            elif i == Z_wt + X_wt - 1:
                result[1] = np.append(result[1], [cand], axis = 0)
    return result


def find_all_codes_in_group(Z_wt, X_wt, group, min_k = 3, return_k = True):
    """
    Finds all codes of weights Z_wt and X_wt for a given group.

    Params:
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * group (np.ndarray): The group whose codes we are finding
        * min_k (int, optional): Whether codes should be filtered to exclude those with k < min_k
        * return_k (bool, optional): Whether to return k or not
    
    Returns:
        * List of tuples of the form (group, Z0, X0, IS_CSS, k) for valid codes. Each of Z0 and
          X0 is a list/tuple of lists/tuples mod group. If return_k is true, also
          adds the k, the logical dimension of the code, to the tuple.
    """

    n = np.prod(group)
    d = len(group)

    #find prime blocks
    blocks = np.array([])
    for power in group:
        if len(blocks) == 0 or primefac(power)[0] != primefac(blocks[-1, 0])[0]:
            blocks = np.append(blocks, [[power]], axis = 0)
        else:
            blocks[-1] = np.append(blocks[-1], [power], axis = 0)
    
    candidates = [minimal_strings_for_subgroup(Z_wt, X_wt, block) for block in blocks]
    

    
    #compute possible indices. first is always 0, second is always a divisor, etc
    Z_candidates = np.zeros((Z_wt, d, 1))
    for g, k in enumerate(group):
        Z_candidates[1, k] = [i for i in range(g) if i == 0 or g % i == 0]
        for i in range(2, Z_wt):
            Z_candidates[i, k] = range(g)
    Z_candidates = [it.product(*c) for c in Z_candidates]
    
    indices = find_strides(group)
    autos = list(list_automorphisms(group))

    result = []
    for z in it.product(*Z_candidates):
        v = z @ indices
        if v != sorted(v):
            continue
        minrep = z
        for p in it.permutations(z):
            prod = np.tensordot(autos, p, axes = ([2], [1]))
            



    n = np.prod(group)
    zs = build_Z0_candidates(Z_wt, group)
    xs = build_X0_candidates(X_wt, group)
    codes = []
    for i in zs:
        for j in xs:
            code = MirrorCode(group, i, j, n = n)
            if min_k > 0 and code.get_k() < min_k:
                continue
            canon_Z, canon_X = canonicalize(group, i, j)
            if np.all(i == canon_Z) and np.all(j == canon_X):
                codes.append((group, i, j, code.is_CSS()) + ((code.get_k(),) if return_k else ()))
    return codes


def find_all_codes(n, Z_wt, X_wt, min_k = 3):
    """
    Finds all codes for a given number of qubits, n, of given weight.

    Params:
        * n (int): The number of physical qubits of the desired codes
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * min_k (int, optional): Whether codes should be filtered to exclude those with k < min_k
    
    Returns:
        * List of tuples of the form (group, Z0, X0) for valid codes. Each of Z0
          and X0 is a list/tuple of lists/tuples mod group. If return_k is true, the
          tuple also has k at the end, the logical dimension of the code.
    """
    if n < 2:
        return []
    #test if n is a power of 2 and if a weight is 3
    if min_k > 0 and (Z_wt == 3 or X_wt == 3):
        p = n
        while True:
            if p == 1:
                return []
            if p % 2 == 1:
                break
            p //= 2
    
    result = []
    for group in n_partitions(n):
        result += find_all_codes_in_group(Z_wt, X_wt, group, min_k, return_k = min_k > 0)
    return result


def main():
    for i in range(32):
        print(i, len(find_all_codes(i, 3, 3)))


if __name__ == "__main__":
    main()
