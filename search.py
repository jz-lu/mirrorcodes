"""
search.py
Search for good mirror codes.
"""

import itertools as it
import os
import pickle
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
# Small cache for permutations
# ============================================================

@lru_cache(maxsize=None)
def get_perms(Z_wt: int, X_wt: int):
    """
    Permutations of {0, ..., Z_wt + X_wt - 1} with the original
    filtering condition from the code.
    """
    total = Z_wt + X_wt
    perms = []
    for perm in it.permutations(range(total)):
        if max(perm[:Z_wt]) == Z_wt - 1 or (Z_wt == X_wt and min(perm[:Z_wt]) == Z_wt):
            perms.append(tuple(perm))
    return tuple(perms)


# ============================================================
# 1) Non-isomorphic abelian groups
# ============================================================

@lru_cache(maxsize=None)
def n_partitions(n):
    """
    Find all non-isomorphic abelian groups of n qubits.

    Returns a tuple of tuples of prime powers whose product is n.
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
    return tuple(result)


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

    elems = [np.array(index_to_array(subgroup, idx), dtype=np.int16)
             for idx in range(group_size)]

    lex_min_raw = lex_minimal_vectors(p, lambdas)
    lex_min = [np.array(v, dtype=np.int16) for v in lex_min_raw]

    result = []
    candidates = [[np.zeros(r, dtype=np.int16)]]
    vec_indices = [0]

    total = Z_wt + X_wt

    while True:
        last_idx = vec_indices[-1]
        current_layer = candidates[-1]

        if last_idx >= len(current_layer):
            if len(vec_indices) == 1:
                break
            vec_indices = vec_indices[:-2] + [vec_indices[-2] + 1]
            candidates.pop()
            continue

        depth = len(vec_indices)

        if depth < total:
            candidates.append([])
            fixed = tuple(
                tuple(int(x) for x in candidates[level][vec_indices[level]])
                for level in range(depth)
            )

            all_fixed_zero = all(
                all(x == 0 for x in f) for f in fixed
            )

            if all_fixed_zero:
                if depth != Z_wt:
                    # Just all lex-minimal representatives
                    candidates[-1] = lex_min
                else:
                    # At depth == Z_wt, enforce the small-weight rule
                    for v_vec in lex_min:
                        if v_vec.max() > (1 if p == 2 else 0):
                            if p > 2:
                                break
                        else:
                            candidates[-1].append(v_vec)
            else:
                # Nontrivial constraints: use automorphisms_fixing_vectors with Z_wt
                isos, shifts = automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed)
                if isos.size == 0:
                    vec_indices[-1] += 1
                    candidates.pop()
                    continue

                num_isos = isos.shape[0]

                for v_vec in elems:
                    if depth == Z_wt and v_vec.max() > (1 if p == 2 else 0):
                        if p > 2:
                            break

                    base_index = int(v_vec @ strides)
                    min_index = base_index

                    if depth < Z_wt:
                        # Z-region: no shift in the comparison
                        for idx in range(num_isos):
                            img = (isos[idx] @ v_vec) % subgroup_np
                            idx_val = int(img @ strides)
                            if idx_val < min_index:
                                min_index = idx_val
                    else:
                        # X-region: apply automorphism then subtract shift
                        for idx in range(num_isos):
                            img = (isos[idx] @ v_vec) % subgroup_np
                            img_shifted = (img - shifts[idx]) % subgroup_np
                            idx_val = int(img_shifted @ strides)
                            if idx_val < min_index:
                                min_index = idx_val

                    if base_index <= min_index:
                        candidates[-1].append(v_vec)

            vec_indices.append(0)
        else:
            code = np.stack(
                [candidates[level][vec_indices[level]] for level in range(depth)],
                axis=0,
            )
            result.append(code.copy())
            vec_indices[-1] += 1

    good = [
        code for code in result
        if is_single_equivalence_class_under_shifts(Z_wt, X_wt, subgroup, code)
    ]
    return good


# ============================================================
# 3) Permutation bins (signed position, vectorized over automorphisms)
# ============================================================

def permutation_bins(Z_wt, X_wt, subgroup, perms, candidates):
    """
    For each candidate code on a prime-power subgroup and each permutation,
    compute the *earliest* non-zero sign and encode it as a single signed
    position:

        block_spos[cand, perm] = 0    : all zeros (no sign)
                                   k  : earliest non-zero sign = +1 at coord k
                                  -k  : earliest non-zero sign = -1 at coord k

    where 1 <= k <= total-1 and total = Z_wt + X_wt.
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
        return np.zeros((num_cands, num_perms), dtype=np.int8)

    perms_np = np.array(perms, dtype=int)
    block_spos = np.zeros((num_cands, num_perms), dtype=np.int8)

    for cand_ind, cand in enumerate(candidates):
        cand = np.asarray(cand, dtype=np.int64)
        # Precompute base lex values for cand rows
        base_lex = (cand @ strides).astype(int)

        # Precompute automorphisms (and shifts) for prefixes of cand:
        # prefix i means we fix cand[0], ..., cand[i-1]
        prefix_auts = [None] * total
        all_zero = [False] * total
        for i in range(1, total):
            fixed = tuple(
                tuple(int(x) for x in cand[j])
                for j in range(i)
            )
            all_zero[i] = all(all(x == 0 for x in f) for f in fixed)
            if not all_zero[i]:
                isos_i, shifts_i = automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed)
                prefix_auts[i] = (isos_i, shifts_i)

        for perm_ind in range(num_perms):
            perm = perms_np[perm_ind]

            # Reorder and normalise by translation so that first row is zero
            c = cand[perm] - cand[perm[0]]

            for i in range(1, total):
                base = int(base_lex[i])

                if all_zero[i]:
                    # No constraints yet: just push c[i] to lex-minimal
                    A = push_to_lex_minimal(p, lambdas, c[i])
                    A_np = np.array(A, dtype=np.int64)
                    img = (A_np @ c[i]) % subgroup_np
                    if i == Z_wt:
                        if p == 2:
                            img = img % 2
                        else:
                            img = img % 1
                    min_iso = A_np
                    min_shift = np.zeros_like(c[i])
                    min_val = int(img @ strides)
                else:
                    aut_pair = prefix_auts[i]
                    if aut_pair is None:
                        break
                    isos, shifts = aut_pair
                    if isos.size == 0:
                        break

                    # Vectorised search over automorphisms
                    # imgs shape: (#isos, r)
                    imgs = (isos @ c[i]) - shifts
                    imgs %= subgroup_np
                    if i == Z_wt:
                        if p == 2:
                            imgs = imgs % 2
                        else:
                            imgs = imgs % 1
                    lex_vals = imgs @ strides
                    min_idx = int(np.argmin(lex_vals))
                    min_val = int(lex_vals[min_idx])
                    min_iso = isos[min_idx]
                    min_shift = shifts[min_idx]

                # Apply chosen automorphism + shift to all rows
                c = ((min_iso @ c.T).T - min_shift) % subgroup_np

                if i == Z_wt:
                    # For p=2, normalise X-rows by subtracting even offset
                    if p == 2:
                        offset = c[Z_wt] - (c[Z_wt] % 2)
                        c[Z_wt:] = (c[Z_wt:] - offset) % subgroup_np
                    else:
                        c[Z_wt:] = (c[Z_wt:] - c[Z_wt]) % subgroup_np

                diff = min_val - base
                if diff > 0:
                    sign = 1
                elif diff < 0:
                    sign = -1
                else:
                    sign = 0

                if sign != 0:
                    block_spos[cand_ind, perm_ind] = np.int8(sign * i)
                    break

    return block_spos


# ============================================================
# 4) Subgroup data: codes + permutation bins with disk storage
# ============================================================

def _subgroup_cache_filename(Z_wt: int, X_wt: int, block):
    """
    Build a filename for this (Z_wt, X_wt, block) in subgroups/.
    We encode block = (p**λ1, ..., p**λr) as (p, λ1,...,λr).
    """
    block = tuple(block)
    first = block[0]
    p = list(primefac(first))[0]

    lambdas = []
    for n in block:
        m = n
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        lambdas.append(e)
    exps_str = "_".join(str(e) for e in lambdas)

    os.makedirs("subgroups", exist_ok=True)
    fname = f"Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
    return os.path.join("subgroups", fname)


def _subgroup_codes_and_bins(Z_wt: int, X_wt: int, block):
    """
    Compute (or load from disk) both the minimal strings and the permutation
    signed-position data for a given (Z_wt, X_wt, block).

    Returns:
        codes_block : list of np.ndarray, each shape (Z_wt+X_wt, len(block))
        block_spos  : np.ndarray, shape (num_cands, num_perms), int8
    """
    block = tuple(block)
    path = _subgroup_cache_filename(Z_wt, X_wt, block)

    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["codes_block"], data["block_spos"]

    perms = get_perms(Z_wt, X_wt)
    codes_block = minimal_strings_for_subgroup(Z_wt, X_wt, block)
    block_spos = permutation_bins(Z_wt, X_wt, block, perms, codes_block)

    codes_block = [np.asarray(code, dtype=np.int16) for code in codes_block]
    block_spos = np.asarray(block_spos, dtype=np.int8)

    data = {
        "codes_block": codes_block,
        "block_spos": block_spos,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return codes_block, block_spos


# ============================================================
# 5) Find all codes in a given abelian group
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

    perms = get_perms(Z_wt, X_wt)
    num_perms = len(perms)

    # (vecs, spos_agg)
    codes = [
        (
            np.zeros((total, 0), dtype=np.int16),
            np.zeros(num_perms, dtype=np.int8),
        )
    ]

    for block in blocks:
        codes_block, block_spos = _subgroup_codes_and_bins(Z_wt, X_wt, block)
        if not codes_block:
            return []

        new_codes = []
        for vecs, spos_agg in codes:
            for idx, code2 in enumerate(codes_block):
                spos_blk = block_spos[idx]

                new_spos = spos_agg.copy()

                blk_nonzero = (spos_blk != 0)
                if np.any(blk_nonzero):
                    agg_nonzero = (spos_agg != 0)

                    idx1 = (~agg_nonzero) & blk_nonzero
                    new_spos[idx1] = spos_blk[idx1]

                    idx2 = agg_nonzero & blk_nonzero
                    pos_agg = np.abs(spos_agg)
                    pos_blk = np.abs(spos_blk)
                    earlier = pos_blk < pos_agg
                    idx2 &= earlier
                    new_spos[idx2] = spos_blk[idx2]

                combined_vecs = np.concatenate((vecs, code2), axis=1)
                new_codes.append((combined_vecs, new_spos))

        if not new_codes:
            return []
        codes = new_codes

    good = []
    for vecs, spos_agg in codes:
        if np.any(spos_agg < 0):
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
# 6) Top-level search over all groups of size n
# ============================================================

def find_all_codes(n, Z_wt, X_wt, min_k=3):
    """
    Finds all codes for a given number of qubits, n, of given weight.
    """
    if n < 2:
        return []

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
# 7) Main (simple test harness)
# ============================================================

def main():
    for i in range(31):
        print(i, len(find_all_codes(i, 3, 3)))


if __name__ == "__main__":
    main()
