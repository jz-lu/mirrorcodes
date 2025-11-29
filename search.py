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

from isomorphism import automorphisms_fixing_vectors, is_single_equivalence_class_under_shifts, \
    lex_minimal_vectors, push_to_lex_minimal
from mirror import MirrorCode
from util import find_strides, index_to_array, partitions


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


def minimal_strings_for_subgroup(Z_wt, X_wt, subgroup):
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    powers = [len(size) for size in factors]
    result = []
    candidates = [[[0] * len(subgroup)]]
    vec_indices = [0]
    strides = find_strides(subgroup)
    lex_min = lex_minimal_vectors(p, powers)
    while True:
        if vec_indices[-1] >= len(candidates[-1]):
            if len(vec_indices) == 1:
                break
            vec_indices = vec_indices[:-2] + [vec_indices[-2] + 1]
            candidates = candidates[:-1]
            continue
        if len(vec_indices) < Z_wt + X_wt:
            if len(vec_indices) == 1:
                candidates += [lex_min]
            else:
                candidates += [[]]
                isos = automorphisms_fixing_vectors(p, powers, [candidates[i][val] for i, val in enumerate(vec_indices)])
                for i in range(np.prod(subgroup)):
                    v = index_to_array(subgroup, i)
                    if len(vec_indices) == Z_wt and max(v) > (1 if p == 2 else 0):
                        if p > 2:
                            break
                    elif i <= min(np.array([np.mod(a @ v, subgroup) for a in isos]) @ strides):
                        candidates[-1] += [v]
            vec_indices += [0]
        else:
            result += [np.ndarray.copy(np.array([candidates[i][val] for i, val in enumerate(vec_indices)], dtype = int))]
            vec_indices[-1] += 1
    return [code for code in result if is_single_equivalence_class_under_shifts(Z_wt, X_wt, subgroup, code)]


def permutation_bins(Z_wt, X_wt, subgroup, perms, candidates):
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    powers = [len(size) for size in factors]
    result = np.zeros((len(candidates), len(perms), Z_wt + X_wt))
    strides = find_strides(subgroup)
    for cand_ind, cand in enumerate(candidates):
        for perm_ind, perm in enumerate(perms):
            c = cand[perm] - cand[perm[0]]
            c[Z_wt:] -= c[Z_wt] - (c[Z_wt] % (2 if p == 2 else 1))
            c %= subgroup
            c = (push_to_lex_minimal(p, powers, c[1]) @ c.T).T % subgroup
            result[cand_ind, perm_ind, 1:] = np.sign(strides @ (c[1] - cand[1]))
            if result[cand_ind, perm_ind, 1] != 0:
                continue
            isos = automorphisms_fixing_vectors(p, powers, cand[1:2])
            for i in range(2, Z_wt + X_wt):
                if i > 2 and len(isos) > 1:
                    isos = np.array([iso for iso in isos if (np.mod(iso @ cand[i - 1], subgroup) == cand[i - 1]).all()])
                sign = np.sign(min([np.mod(iso @ c[i], subgroup) @ strides for iso in isos]) - strides @ cand[i])
                result[cand_ind, perm_ind, i:] = sign
                if sign != 0:
                    break
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
    perms = [list(i) for i in it.permutations(range(Z_wt + X_wt)) if max(i[:Z_wt]) == Z_wt - 1
             or (Z_wt != X_wt or min(i[:Z_wt]) == Z_wt)]
    subcodes = [minimal_strings_for_subgroup(Z_wt, X_wt, block) for block in blocks]
    subsigns = [permutation_bins(Z_wt, X_wt, block, perms, subcodes[i]) for i, block in enumerate(blocks)]
    codes = [([[] for _ in range(Z_wt + X_wt)], np.zeros((len(perms), Z_wt + X_wt)))]
    f = lambda x, y: y if x == 0 else x
    for i in range(len(blocks)):
        new_codes = []
        for code in codes:
            for code2_ind, code2 in enumerate(subcodes[i]):
                new_signs = np.array([[f(code[1][j, k], subsigns[i][code2_ind][j, k])
                                       for k in range(Z_wt + X_wt)] for j in range(len(perms))])
                if min(new_signs[:, 0]) >= 0:
                    new_codes += [(np.append(code[0], code2, axis = 1), new_signs)]
        codes = new_codes
    twos = find_strides([2] * (Z_wt + X_wt))
    codes = [MirrorCode(group, code[0][:Z_wt], code[0][Z_wt:]) for code in codes if
             min(code[1] @ twos) >= 0 and len(np.unique(code[0][:Z_wt], axis = 0)) == Z_wt
             and len(np.unique(code[0][Z_wt:], axis = 0)) == X_wt]
    return [(group, code.z0, code.x0, code.is_CSS()) + ((code.get_k(),) if return_k else ()) for code in codes
            if code.get_k() >= min_k]


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
    # test if n is a power of 2 and if a weight is 3
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
    for i in range(25):
        print(i, len(find_all_codes(i, 3, 3)))


if __name__ == "__main__":
    main()
