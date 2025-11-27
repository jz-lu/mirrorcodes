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
from mirror import find_stabilizers, MirrorCode
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


def minimal_strings_for_subgroup(Z_wt, X_wt, subgroup):
    p = list(primefac(subgroup[0]))[0]
    result = []
    candidates = [[[0] * len(subgroup)]]
    vec_indices = [0]
    strides = find_strides(subgroup)
    while True:
        if vec_indices[-1] >= len(candidates[-1]):
            if len(vec_indices) == 1:
                break
            vec_indices = vec_indices[:-2] + [vec_indices[-2] + 1]
            candidates = candidates[:-1]
            continue
        Z_vals = np.array([candidates[i][val] for i, val in enumerate(vec_indices[:min(len(vec_indices), Z_wt)])]) @ strides
        if len(set(Z_vals)) < len(Z_vals):
            vec_indices[-1] += 1
            continue
        if len(vec_indices) > Z_wt:
            X_vals = np.array([candidates[i + Z_wt][val] for i, val in enumerate(vec_indices[Z_wt:len(vec_indices)])]) @ strides
            if len(set(X_vals)) < len(X_vals):
                vec_indices[-1] += 1
                continue
        if len(vec_indices) < Z_wt + X_wt:
            if len(vec_indices) == 1:
                candidates += [lex_minimal_vectors(subgroup)]
            else:
                candidates += [[]]
                isos = automorphisms_fixing_vectors(subgroup, [candidates[i][val] for i, val in enumerate(vec_indices)])
                strides = find_strides(subgroup)
                for i in range(np.prod(subgroup)):
                    v = index_to_array(subgroup, i)
                    if len(vec_indices) == Z_wt and max(v) > (1 if p == 2 else 0):
                        if p > 2:
                            break
                    elif i <= min(np.array([np.mod(a @ v, subgroup) for a in isos]) @ strides):
                        candidates[-1] += [v]
            vec_indices += [0]
        else:
            result += [np.ndarray.copy(np.array([candidates[i][val] for i, val in enumerate(vec_indices)]))]
            vec_indices[-1] += 1
    return result


def permutation_bins(Z_wt, X_wt, subgroup, perms, candidates):
    p = list(primefac(subgroup[0]))[0]
    result = np.zeros((len(candidates), len(perms), Z_wt + X_wt))
    strides = find_strides(subgroup)
    for cand_ind, cand in enumerate(candidates):
        for perm_ind, perm in enumerate(perms):
            c = np.ndarray.copy(cand[perm])
            c -= c[0]
            c %= subgroup
            min_1 = push_to_lex_minimal(subgroup, c[1])
            result[cand_ind, perm_ind, 1] = np.sign(strides @ (min_1 - cand[1]))
            if result[cand_ind, perm_ind, 1] != 0:
                result[cand_ind, perm_ind, 2:] = result[cand_ind, perm_ind, 1]
                continue
            isos = []
            for i in range(2, Z_wt + X_wt):
                if i == 2:
                    isos = automorphisms_fixing_vectors(subgroup, c[1:i])
                else:
                    isos = np.array([iso for iso in isos if (np.mod(iso @ c[i - 1], subgroup) == c[i - 1]).all()])
                values = np.array([np.mod(iso @ c[i], subgroup) for iso in isos])
                if i == Z_wt:
                    values %= (2 - (p % 2))
                result[cand_ind, perm_ind, i] = np.sign(min(values @ strides) - strides @ cand[i])
                if result[cand_ind, perm_ind, i] != 0:
                    result[cand_ind, perm_ind, i + 1:] = result[cand_ind, perm_ind, i]
                    break
                if i == Z_wt:
                    c[i:] -= c[i] - cand[i]
                    c %= subgroup
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
    #find prime blocks
    blocks = []
    for power in group:
        if len(blocks) == 0 or list(primefac(power))[0] != list(primefac(blocks[-1][0]))[0]:
            blocks += [[power]]
        else:
            blocks[-1] += [power]
    perms = [list(i) for i in it.permutations(range(Z_wt + X_wt)) if max(i[:Z_wt]) == Z_wt - 1]
    subcodes = [minimal_strings_for_subgroup(Z_wt, X_wt, block) for block in blocks]
    subsigns = [permutation_bins(Z_wt, X_wt, block, perms, subcodes[i]) for i, block in enumerate(blocks)]
    codes = [([[] for _ in range(Z_wt + X_wt)], np.zeros((len(perms), Z_wt + X_wt)))]
    for i in range(len(blocks)):
        new_codes = []
        for code in codes:
            for code2_ind, code2 in enumerate(subcodes[i]):
                f = lambda x, y: y if x == 0 else x
                new_signs = np.array([[f(code[1][j, k], subsigns[i][code2_ind][j, k])
                                       for k in range(Z_wt + X_wt)] for j in range(len(perms))])
                if min(new_signs[:, 0]) >= 0:
                    new_codes += [(np.append(code[0], code2, axis = 1), new_signs)]
        codes = new_codes
    twos = find_strides([2] * (Z_wt + X_wt))
    codes = [MirrorCode(group, code[0][:Z_wt], code[0][Z_wt:]) for code in codes if
             min(code[1] @ twos) >= 0 and len(np.unique(code[0][:Z_wt])) == Z_wt and len(np.unique(code[0][Z_wt:])) == X_wt]
    return [(group, code.z0, code.x0, code.is_CSS()) + ((code.get_k(),) if return_k else ()) for code in codes]


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
