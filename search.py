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

    # All group elements as vectors (int16 for safe arithmetic)
    elems = [np.array(index_to_array(subgroup, idx), dtype=np.int16)
             for idx in range(group_size)]

    # All lex-minimal representatives for this p-group
    lex_min = [np.array(v, dtype=np.int16) for v in lex_minimal_vectors(p, lambdas)]

    result = []
    # candidates[depth] is list of possible vectors at that depth
    candidates = [[np.zeros(r, dtype=np.int16)]]
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

                # Check if all fixed vectors are zero
                all_fixed_zero = all(
                    all(x == 0 for x in f) for f in fixed
                )

                if all_fixed_zero:
                    # No constraints yet: use push_to_lex_minimal instead of
                    # enumerating all automorphisms with automorphisms_fixing_vectors.
                    for v in elems:
                        if depth == Z_wt and v.max() > (1 if p == 2 else 0):
                            if p > 2:
                                break
                        # Compute lex-minimal image of v under full Aut(G)
                        A = push_to_lex_minimal(p, lambdas, tuple(int(x) for x in v))
                        A_np = np.array(A, dtype=int)
                        v_img = (A_np @ v) % subgroup_np
                        # Keep v only if it is already lex-minimal in its orbit
                        if np.array_equal(v_img, v):
                            candidates[-1].append(v)
                else:
                    # Nontrivial constraints: use automorphisms_fixing_vectors
                    isos = automorphisms_fixing_vectors(p, lambdas, fixed)
                    if isos.size == 0:
                        vec_indices[-1] += 1
                        candidates.pop()
                        continue

                    for v in elems:
                        if depth == Z_wt and v.max() > (1 if p == 2 else 0):
                            if p > 2:
                                break
                        images = (isos @ v) % subgroup_np  # (#isos, r)
                        lex_vals = images @ strides
                        if (v @ strides) <= lex_vals.min():
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
# 3) Permutation bins (memory-efficient: signed position)
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

    # Signed earliest position (0 = none)
    block_spos = np.zeros((num_cands, num_perms), dtype=np.int8)

    ZERO_PREFIX = object()  # sentinel for "all fixed vectors so far are zero"

    for cand_ind, cand in enumerate(candidates):
        cand = np.asarray(cand, dtype=np.int16)

        # Precompute automorphisms that fix prefixes of cand:
        # isos_by_prefix[i] = automorphisms that fix cand[1],...,cand[i-1]
        isos_by_prefix = [None] * total
        if total > 2:
            # Handle the first prefix (just cand[1])
            if np.all(cand[1] == 0):
                # All vectors so far (just cand[1]) are zero:
                # do NOT call automorphisms_fixing_vectors here.
                isos_by_prefix[2] = ZERO_PREFIX
                isos_current = None
            else:
                fixed1 = (tuple(int(x) for x in cand[1]),)
                isos_current = automorphisms_fixing_vectors(p, lambdas, fixed1)
                isos_by_prefix[2] = isos_current

            # Longer prefixes
            for i in range(3, total):
                # Prefix cand[1..i-1]
                prefix_block = cand[1:i]
                if not np.any(prefix_block):
                    # All fixed vectors so far are zero: mark as ZERO_PREFIX
                    isos_by_prefix[i] = ZERO_PREFIX
                    isos_current = None
                    continue

                # We have at least one non-zero vector in the prefix
                if isos_current is None:
                    # Compute stabiliser from scratch using the non-zero prefix vectors
                    fixed_prefix = tuple(
                        tuple(int(x) for x in row)
                        for row in prefix_block
                        if np.any(row)
                    )
                    if fixed_prefix:
                        isos_current = automorphisms_fixing_vectors(p, lambdas, fixed_prefix)
                    else:
                        isos_current = None
                else:
                    if np.size(isos_current, axis=0) > 1:
                        v_prev = cand[i - 1]
                        images_prev = (isos_current @ v_prev) % subgroup_np
                        mask = (images_prev == v_prev).all(axis=1)
                        isos_current = isos_current[mask]

                isos_by_prefix[i] = isos_current

        for perm_ind in range(num_perms):
            perm = perms_np[perm_ind]

            # Reorder and normalise
            c = cand[perm] - cand[perm[0]]

            if p == 2:
                offset = c[Z_wt] - (c[Z_wt] % 2)
                c[Z_wt:] -= offset
            else:
                c[Z_wt:] -= c[Z_wt]

            # Push c[1] to lex-minimal and apply automorphism to all rows
            A1 = push_to_lex_minimal(p, lambdas, tuple(int(x) for x in c[1]))
            A1_np = np.array(A1, dtype=int)
            c = (A1_np @ c.T).T % subgroup_np
            c = c.astype(np.int16, copy=False)

            # First candidate coordinate: index 1
            d1 = int(strides @ (c[1] - cand[1]))
            if d1 > 0:
                s1 = 1
            elif d1 < 0:
                s1 = -1
            else:
                s1 = 0

            if s1 != 0:
                block_spos[cand_ind, perm_ind] = np.int8(s1 * 1)
                continue

            # Compare positions 2, 3, ... until a sign is determined
            for i in range(2, total):
                if total <= 2:
                    break

                isos = isos_by_prefix[i]

                # Case 1: prefix is all zeros -> use full Aut(G) via push_to_lex_minimal
                if isos is ZERO_PREFIX:
                    base = int(strides @ cand[i])
                    A_i = push_to_lex_minimal(p, lambdas, tuple(int(x) for x in c[i]))
                    A_i_np = np.array(A_i, dtype=int)
                    img = (A_i_np @ c[i]) % subgroup_np
                    lex_val = int(strides @ img)
                    diff = lex_val - base

                    if diff > 0:
                        sign = 1
                    elif diff < 0:
                        sign = -1
                    else:
                        sign = 0
                        # Align entire c with this automorphism for consistency
                        c = (A_i_np @ c.T).T % subgroup_np

                # Case 2: no automorphisms left or not computed -> no further sign
                elif isos is None or (hasattr(isos, "size") and isos.size == 0):
                    break

                # Case 3: nontrivial stabiliser group: enumerate isos
                else:
                    base = int(strides @ cand[i])
                    min_iso = isos[0]
                    min_val = 10**9
                    for iso in isos:
                        img = (iso @ c[i]) % subgroup_np
                        lex_val = int(strides @ img)
                        if lex_val < min_val:
                            min_iso = iso
                            min_val = lex_val
                            if min_val < base:
                                break
                    diff = min_val - base

                    if diff > 0:
                        sign = 1
                    elif diff < 0:
                        sign = -1
                    else:
                        sign = 0
                        c = (min_iso @ c.T).T % subgroup_np

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

    # exponents λ_i such that block[i] = p**λ_i
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
    signed-position data for a given (Z_wt, X_wt, block).  block must be a
    tuple of prime powers for a single prime.

    Returns:
        codes_block : list of np.ndarray, each shape (Z_wt+X_wt, len(block))
        block_spos  : np.ndarray, shape (num_cands, num_perms), int8

    Disk format (in subgroups/…pkl) is a dict with these keys.
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

    # Normalise dtypes to small ints
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

    # Each entry in `codes` is:
    #   (vecs, spos_agg)
    # where:
    #   vecs     : np.ndarray (total, num_coords_so_far), int16
    #   spos_agg : np.ndarray (num_perms,), int8
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

                # Combine signed positions:
                #  - If aggregate is zero and block non-zero: take block.
                #  - If both non-zero: take the earlier (smaller |pos|).
                new_spos = spos_agg.copy()

                blk_nonzero = (spos_blk != 0)
                if np.any(blk_nonzero):
                    agg_nonzero = (spos_agg != 0)

                    # Case 1: previously zero, block non-zero -> take block
                    idx1 = (~agg_nonzero) & blk_nonzero
                    new_spos[idx1] = spos_blk[idx1]

                    # Case 2: both non-zero -> take earlier position
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

    # Final filters and build MirrorCode objects.
    # A row (per permutation) has negative scalar in the original "signs @ twos"
    # iff its earliest sign is -1. In the signed-position encoding, that's
    # exactly "signed_pos < 0".
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
# 7) Main (test harness)
# ============================================================

def main():
    for i in range(37):
        print(i, len(find_all_codes(i, 3, 3)))


if __name__ == "__main__":
    main()
