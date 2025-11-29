"""
search.py
Search for good mirror codes.
"""

import itertools as it
from functools import lru_cache

import numpy as np
from primefac import primefac

from isomorphism import (
    automorphisms_fixing_vectors,
    is_single_equivalence_class_under_shifts,
    lex_minimal_vectors,
    push_to_lex_minimal,
)
from mirror import MirrorCode
from util import find_strides, index_to_array, partitions


# ============================================================
# Small caches for permutations and [2,...,2] strides
# ============================================================

@lru_cache(maxsize=None)
def _get_perms(Z_wt: int, X_wt: int):
    """
    Permutations of {0, ..., Z_wt + X_wt - 1} with the original
    filtering condition from the code.
    """
    total = Z_wt + X_wt
    perms = []
    for perm in it.permutations(range(total)):
        if max(perm[:Z_wt]) == Z_wt - 1 or (Z_wt != X_wt or min(perm[:Z_wt]) == Z_wt):
            perms.append(perm)
    return tuple(tuple(p) for p in perms)


@lru_cache(maxsize=None)
def _get_twos(total: int):
    """Strides for the vector [2,2,...,2] of given length."""
    return tuple(find_strides([2] * total))


# ============================================================
# 1) Non-isomorphic abelian groups
# ============================================================

def n_partitions(n):
    """
    Find all non-isomorphic abelian groups of n qubits.

    Returns a list of tuples of prime powers whose product is n.
    """
    primes = []
    powers = []

    for i in it.groupby(list(primefac(n))):
        primes.append(i[0])
        powers.append(len(list(i[1])))

    combos = [list(partitions(i)) for i in powers]

    result = []
    for combo in it.product(*combos):
        group_sizes = []
        for part, p in zip(combo, primes):
            for k in part:
                group_sizes.append(p ** k)
        result.append(tuple(group_sizes))
    return result


# ============================================================
# 2) Minimal strings for a subgroup
# ============================================================

def minimal_strings_for_subgroup(Z_wt, X_wt, subgroup):
    """
    Compute lex-minimal strings for a given prime-power subgroup.

    subgroup: iterable of p-powers, all with the same prime p.
    """
    subgroup = tuple(subgroup)
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    lambdas = tuple(len(f) for f in factors)
    r = len(subgroup)

    strides = np.array(find_strides(subgroup), dtype=int)
    subgroup_np = np.array(subgroup, dtype=int)
    group_size = int(np.prod(subgroup_np))

    # All group elements as vectors
    elems = [np.array(index_to_array(subgroup, idx), dtype=int)
             for idx in range(group_size)]

    # All lex-minimal representatives for this p-group
    lex_min = [np.array(v, dtype=int) for v in lex_minimal_vectors(p, lambdas)]

    result = []
    # candidates[depth] is list of possible vectors at that depth
    candidates = [[np.zeros(r, dtype=int)]]
    vec_indices = [0]

    while True:
        last_idx = vec_indices[-1]
        current_layer = candidates[-1]

        if last_idx >= len(current_layer):
            # backtrack
            if len(vec_indices) == 1:
                break
            vec_indices = vec_indices[:-2] + [vec_indices[-2] + 1]
            candidates.pop()
            continue

        depth = len(vec_indices)
        if depth < Z_wt + X_wt:
            if depth == 1:
                # second layer: lex-minimal vectors
                candidates.append(lex_min)
            else:
                candidates.append([])
                # fixed vectors so far
                fixed = tuple(
                    tuple(int(x) for x in candidates[level][vec_indices[level]])
                    for level in range(depth)
                )
                isos = automorphisms_fixing_vectors(p, lambdas, fixed)
                if isos.size == 0:
                    vec_indices[-1] += 1
                    candidates.pop()
                    continue

                for idx, v in enumerate(elems):
                    if depth == Z_wt and v.max() > (1 if p == 2 else 0):
                        if p > 2:
                            break
                    else:
                        images = (isos @ v) % subgroup_np  # (#isos, r)
                        lex_vals = images @ strides
                        if idx <= int(lex_vals.min()):
                            candidates[-1].append(v)

            vec_indices.append(0)
        else:
            code = np.stack(
                [candidates[level][vec_indices[level]] for level in range(depth)],
                axis=0,
            )
            result.append(code.copy())
            vec_indices[-1] += 1

    # Filter by equivalence under shifts
    good = [
        code for code in result
        if is_single_equivalence_class_under_shifts(Z_wt, X_wt, subgroup, code)
    ]
    return good


# ============================================================
# 3) Permutation bins
# ============================================================

def permutation_bins(Z_wt, X_wt, subgroup, perms, candidates):
    """
    For each candidate code on a prime-power subgroup and each permutation,
    compute sign information used to glue blocks.
    """
    subgroup = tuple(subgroup)
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    lambdas = tuple(len(f) for f in factors)
    strides = np.array(find_strides(subgroup), dtype=int)
    subgroup_np = np.array(subgroup, dtype=int)

    num_cands = len(candidates)
    num_perms = len(perms)
    total = Z_wt + X_wt

    if num_cands == 0 or num_perms == 0:
        return np.zeros((num_cands, num_perms, total), dtype=np.int8)

    result = np.zeros((num_cands, num_perms, total), dtype=np.int8)

    for cand_ind, cand in enumerate(candidates):
        cand = np.asarray(cand, dtype=int)

        for perm_ind, perm in enumerate(perms):
            # Reorder and normalise
            c = cand[list(perm)] - cand[perm[0]]
            if p == 2:
                offset = c[Z_wt] - (c[Z_wt] % 2)
                c[Z_wt:] -= offset
            else:
                c[Z_wt:] -= c[Z_wt]

            A = push_to_lex_minimal(p, lambdas, tuple(int(x) for x in c[1]))
            A_np = np.array(A, dtype=int)
            c = (A_np @ c.T).T % subgroup_np

            # Compare first non-fixed coordinate (index 1)
            d1 = int(strides @ (c[1] - cand[1]))
            if d1 > 0:
                s1 = 1
            elif d1 < 0:
                s1 = -1
            else:
                s1 = 0
            result[cand_ind, perm_ind, 1:] = s1
            if s1 != 0:
                continue

            # Automorphisms fixing cand[1]
            fixed = (tuple(int(x) for x in cand[1]),)
            isos = automorphisms_fixing_vectors(p, lambdas, fixed)

            for i in range(2, total):
                if isos.size > 1 and i > 2:
                    v_prev = cand[i - 1]
                    images_prev = (isos @ v_prev) % subgroup_np
                    mask = (images_prev == v_prev).all(axis=1)
                    isos = isos[mask]
                    if isos.size == 0:
                        break

                images = (isos @ c[i]) % subgroup_np
                lex_vals = images @ strides
                base = int(strides @ cand[i])
                diff = int(lex_vals.min()) - base

                if diff > 0:
                    sign = 1
                elif diff < 0:
                    sign = -1
                else:
                    sign = 0

                result[cand_ind, perm_ind, i:] = sign
                if sign != 0:
                    break

    return result


# ============================================================
# 4) Find all codes in a given abelian group
# ============================================================

def find_all_codes_in_group(Z_wt, X_wt, group, min_k=3, return_k=True):
    """
    Finds all codes of weights Z_wt and X_wt for a given group.

    group: tuple of prime powers.
    """
    group = tuple(group)
    total = Z_wt + X_wt

    # Split into prime blocks
    blocks = []
    i = 0
    L = len(group)
    while i < L:
        power = group[i]
        factors = list(primefac(power))
        p = factors[0]
        block = [power]
        i += 1
        while i < L:
            nxt = group[i]
            if nxt % p != 0:
                break
            block.append(nxt)
            i += 1
        blocks.append(tuple(block))

    perms = _get_perms(Z_wt, X_wt)
    num_perms = len(perms)

    # For each block, compute minimal strings and permutation signatures
    subcodes = []
    subsigns = []
    for block in blocks:
        codes_block = minimal_strings_for_subgroup(Z_wt, X_wt, block)
        if not codes_block:
            return []
        subcodes.append(codes_block)
        subsigns.append(permutation_bins(Z_wt, X_wt, block, perms, codes_block))

    # Combine blocks
    # Each entry: (code_vectors, sign_matrix)
    #   code_vectors: np.array shape (total, num_coords_so_far)
    #   sign_matrix : np.int8 array shape (num_perms, total)
    codes = [
        (np.zeros((total, 0), dtype=int),
         np.zeros((num_perms, total), dtype=np.int8))
    ]

    for block_idx, block in enumerate(blocks):
        codes_block = subcodes[block_idx]
        signs_block = subsigns[block_idx]
        new_codes = []

        for vecs, signs in codes:
            for code2_ind, code2 in enumerate(codes_block):
                block_signs = signs_block[code2_ind]
                new_signs = signs.copy()
                mask = (new_signs == 0)
                new_signs[mask] = block_signs[mask]
                if new_signs[:, 0].min() >= 0:
                    combined_vecs = np.concatenate((vecs, code2), axis=1)
                    new_codes.append((combined_vecs, new_signs))

        if not new_codes:
            return []
        codes = new_codes

    # Final filters and build MirrorCode objects
    twos = np.array(_get_twos(total), dtype=int)
    good = []
    for vecs, signs in codes:
        if (signs @ twos).min() < 0:
            continue

        z_part = vecs[:Z_wt]
        x_part = vecs[Z_wt:]

        if len(np.unique(z_part, axis=0)) != Z_wt:
            continue
        if len(np.unique(x_part, axis=0)) != X_wt:
            continue

        code = MirrorCode(group, z_part, x_part)
        k_val = code.get_k()
        if k_val < min_k:
            continue
        good.append((code, k_val))

    if not good:
        return []

    if return_k:
        return [
            (group, code.z0, code.x0, code.is_CSS(), k_val)
            for code, k_val in good
        ]
    else:
        return [
            (group, code.z0, code.x0, code.is_CSS())
            for code, _ in good
        ]


# ============================================================
# 5) Top-level search over all groups of size n
# ============================================================

def find_all_codes(n, Z_wt, X_wt, min_k=3):
    """
    Finds all codes for a given number of qubits, n, of given weight.
    """
    if n < 2:
        return []

    # If n is a power of 2 and a weight is 3, there are no codes.
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
        result.extend(
            find_all_codes_in_group(Z_wt, X_wt, group, min_k, return_k=(min_k > 0))
        )
    return result


# ============================================================
# 6) Main
# ============================================================

def main():
    for i in range(25):
        print(i, len(find_all_codes(i, 3, 3)))


if __name__ == "__main__":
    main()
