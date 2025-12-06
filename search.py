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


@lru_cache(maxsize=None)
def _get_perm_data(Z_wt: int, X_wt: int):
    """
    Precompute permutation data for this (Z_wt, X_wt):

      * perms            : tuple of permutations (same as get_perms)
      * perms_np         : numpy array of shape (num_perms, total)
      * groups_by_depth  : tuple of length `total`, where
          groups_by_depth[i] is a list of numpy arrays of permutation
          indices. Each array is a group of perms that share the same
          prefix (perm[0], ..., perm[i]) (i.e. prefix length i+1).

    This depends only on Z_wt and X_wt, not on the candidate or subgroup,
    so we can reuse it across all calls to permutation_bins.
    """
    perms = get_perms(Z_wt, X_wt)
    perms_np = np.array(perms, dtype=int)
    total = Z_wt + X_wt

    groups_by_depth = [None] * total
    for i in range(1, total):
        prefix_to_inds = {}
        for idx, perm in enumerate(perms):
            key = perm[: i + 1]  # prefix of length i+1
            prefix_to_inds.setdefault(key, []).append(idx)
        # Store each group as a numpy array of indices
        groups_by_depth[i] = [np.array(v, dtype=np.int32) for v in prefix_to_inds.values()]

    return perms, perms_np, tuple(groups_by_depth)


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

    Optimised memory use:
      * We no longer build a huge intermediate `result` and then filter.
      * Instead, we test each finished candidate immediately with
        `is_single_equivalence_class_under_shifts` and only keep the
        ones that pass.
      * We also avoid an unnecessary `.copy()` on the stacked code.
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
    elems = [np.array(index_to_array(subgroup, idx), dtype=np.int16)
             for idx in range(group_size)]

    lex_min_raw = lex_minimal_vectors(p, lambdas)
    lex_min = [np.array(v, dtype=np.int16) for v in lex_min_raw]

    good = []  # only store codes that pass the equivalence-class test
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
            # Build a full code for this candidate
            code = np.stack(
                [candidates[level][vec_indices[level]] for level in range(depth)],
                axis=0,
            )

            # Filter immediately: only keep codes that form a single
            # equivalence class under shifts
            if is_single_equivalence_class_under_shifts(Z_wt, X_wt, subgroup, code):
                good.append(code)

            vec_indices[-1] += 1

    return good


# ============================================================
# 3) Permutation bins (signed position, prefix-group sharing)
# ============================================================

def permutation_bins(Z_wt, X_wt, subgroup, candidates):
    """
    For each candidate code on a prime-power subgroup and each permutation,
    compute the *earliest* non-zero sign and encode it as a single signed
    position:

        block_spos[cand, perm] = 0    : all zeros (no sign)
                                   k  : earliest non-zero sign = +1 at coord k
                                  -k  : earliest non-zero sign = -1 at coord k

    where 1 <= k <= total-1 and total = Z_wt + X_wt.

    Optimisation:
      * Permutations and their prefix groups are generated once via
        `_get_perm_data(Z_wt, X_wt)`, so we avoid repeated `np.unique`
        calls per candidate and per depth.
      * For each candidate, and for each coordinate i = 1..total-1, we:
          - take the precomputed groups of perms that share the same
            prefix (perm[0], ..., perm[i]);
          - restrict each group to still-unassigned perms;
          - run the original algorithm up to i once on a single
            representative of each non-empty group;
          - reuse that earliest sign (if nonzero) for all perms in the
            group.
      * This reuse is mathematically safe because, for a fixed candidate,
        the behaviour up to coordinate i depends only on perm[0..i].
    """
    subgroup = tuple(subgroup)
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    lambdas = tuple(len(f) for f in factors)
    strides = np.array(find_strides(subgroup), dtype=int)
    subgroup_np = np.array(subgroup, dtype=int)

    total = Z_wt + X_wt
    perms, perms_np, groups_by_depth = _get_perm_data(Z_wt, X_wt)
    num_perms = len(perms)

    num_cands = len(candidates)
    if num_cands == 0 or num_perms == 0:
        return np.zeros((num_cands, num_perms), dtype=np.int8)

    block_spos = np.zeros((num_cands, num_perms), dtype=np.int8)

    def _earliest_sign_up_to_i(cand_np, base_lex, prefix_auts, all_zero,
                               perm_vec, i_max):
        """
        Run the original permutation_bins inner loop for this candidate+perm,
        but only up to coordinate i_max. Return (sign, pos) where:

            sign in {-1, 0, +1}
            pos  in {0, 1, ..., i_max}

        with pos=0 meaning "no non-zero sign up to i_max".
        """
        # Initial translation: c[0] = 0, c[j] = cand[perm[j]] - cand[perm[0]]
        c = cand_np[perm_vec] - cand_np[perm_vec[0]]
        c = c.copy()  # ensure no aliasing back into cand_np

        limit = min(i_max, total - 1)
        for i in range(1, limit + 1):
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
                return 1, i
            elif diff < 0:
                return -1, i
            # else sign == 0: continue

        return 0, 0

    for cand_ind, cand in enumerate(candidates):
        cand_np = np.asarray(cand, dtype=np.int64)
        # Precompute base lex values for cand rows
        base_lex = (cand_np @ strides).astype(int)

        # Precompute automorphisms (and shifts) for prefixes of cand:
        # prefix i means we fix cand[0], ..., cand[i-1]
        prefix_auts = [None] * total
        all_zero = [False] * total
        for i in range(1, total):
            fixed = tuple(
                tuple(int(x) for x in cand_np[j])
                for j in range(i)
            )
            all_zero[i] = all(all(x == 0 for x in f) for f in fixed)
            if not all_zero[i]:
                isos_i, shifts_i = automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed)
                prefix_auts[i] = (isos_i, shifts_i)

        assigned = np.zeros(num_perms, dtype=bool)

        # For each coordinate i, use precomputed groups of perms with the same prefix
        for i in range(1, total):
            remaining = np.where(~assigned)[0]
            if remaining.size == 0:
                break

            groups = groups_by_depth[i]
            for g in groups:
                # Restrict to still-unassigned perms in this group
                mask = ~assigned[g]
                if not np.any(mask):
                    continue
                g_unassigned = g[mask]

                rep_idx = int(g_unassigned[0])
                perm_vec = perms_np[rep_idx]

                sign, pos = _earliest_sign_up_to_i(
                    cand_np,
                    base_lex,
                    prefix_auts,
                    all_zero,
                    perm_vec,
                    i_max=i,
                )

                if sign != 0:
                    val = np.int8(sign * pos)
                    block_spos[cand_ind, g_unassigned] = val
                    assigned[g_unassigned] = True
                # If sign == 0, we leave these perms unassigned and they
                # will be considered at the next coordinate i+1.

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

    codes_block = minimal_strings_for_subgroup(Z_wt, X_wt, block)
    block_spos = permutation_bins(Z_wt, X_wt, block, codes_block)

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

    Memory-optimised version:
      - Decomposes the group into prime blocks.
      - For each block, loads:
          * codes_block : list of candidate block codes (matrices)
          * block_spos  : (#cands, #perms) signed positions
      - Runs a DFS over choices of one candidate per block.
        Each DFS state stores ONLY:
          * idx_tuple: a tuple of candidate indices (one per processed block)
          * spos_agg: aggregated signed-position vector across those blocks
      - Only at DFS leaves (all blocks chosen) do we reconstruct the full
        code (concatenating columns) and build MirrorCode objects.

    Extra pruning:
      - For each block and permutation, precompute the *best possible*
        positive coordinate (if any). From that we derive, for each DFS
        depth, the best future positive coordinate available.
      - If at some DFS state, for a permutation p we have a current
        negative earliest sign at coordinate k and even in the best case
        across all remaining blocks we cannot get a positive sign at a
        coordinate < k, then no completion of this branch can yield a
        valid code, and we prune that branch immediately.
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

    # We still use get_perms to know how many permutations we have
    perms = get_perms(Z_wt, X_wt)
    num_perms = len(perms)
    num_blocks = len(blocks)

    # Preload all blocks' codes and signed positions once
    blocks_data = []
    for block in blocks:
        codes_block, block_spos = _subgroup_codes_and_bins(Z_wt, X_wt, block)
        if not codes_block:
            return []
        codes_block = [np.asarray(code, dtype=np.int16) for code in codes_block]
        block_spos = np.asarray(block_spos, dtype=np.int8)
        blocks_data.append((codes_block, block_spos))

    # ------------------------------------------------------------
    # Precompute best possible positive coordinate per block/perm
    # and the best from any future block.
    #
    # best_pos[b, p]  = min k>0 over candidates with +k at perm p
    #                   or 0 if no positive sign is ever possible.
    # best_from[b, p] = min_{b' >= b} best_pos[b', p]
    #               (best possible positive from block b or later).
    # ------------------------------------------------------------
    best_pos = np.zeros((num_blocks, num_perms), dtype=np.uint8)

    for b, (_, block_spos) in enumerate(blocks_data):
        # block_spos shape: (num_cands_b, num_perms), int8
        # We want the minimum positive k per perm.
        bs = block_spos
        # Use 127 as a sentinel (since coords <= total-1 <= 255 but we store in uint8).
        # First work in int16 to avoid int8 overflow weirdness.
        arr = np.where(bs > 0, bs.astype(np.int16), 127)
        min_pos = arr.min(axis=0)  # int16
        # Convert to uint8, map sentinel back to 0 ("no positive available").
        min_pos_u8 = min_pos.astype(np.uint8)
        min_pos_u8[min_pos_u8 == 127] = 0
        best_pos[b, :] = min_pos_u8

    # best_from[b, p] = best positive from blocks b, b+1, ...
    best_from = np.zeros((num_blocks + 1, num_perms), dtype=np.uint8)
    # Row num_blocks (no future blocks) is all zeros by construction.
    # Fill backwards.
    for b in range(num_blocks - 1, -1, -1):
        if b == num_blocks - 1:
            best_from[b, :] = best_pos[b, :]
        else:
            best_from[b, :] = np.minimum(best_pos[b, :], best_from[b + 1, :])

    good = []

    def dfs(block_index, idx_tuple, spos_agg):
        """
        DFS over blocks:

        block_index : which block we're choosing a candidate for now
        idx_tuple   : tuple of candidate indices chosen for blocks [0..block_index-1]
        spos_agg    : aggregated signed positions across those blocks
        """
        # If we've assigned all blocks, evaluate the full code
        if block_index == num_blocks:
            # Final sign filter: any negative signed position kills the code
            if (spos_agg < 0).any():
                return

            # Reconstruct full vecs by concatenating block codes along columns
            vecs_parts = []
            for b, cand_idx in enumerate(idx_tuple):
                codes_block_b, _ = blocks_data[b]
                vecs_parts.append(codes_block_b[cand_idx])
            vecs = np.concatenate(vecs_parts, axis=1)  # shape (total, len(group))

            z_part = vecs[:Z_wt]
            x_part = vecs[Z_wt:]

            # Z and X rows must be distinct
            if len(np.unique(z_part, axis=0)) != Z_wt:
                return
            if len(np.unique(x_part, axis=0)) != X_wt:
                return

            code = MirrorCode(group, z_part, x_part)
            k_val = code.get_k()
            if k_val < min_k:
                return

            good.append((code, k_val))
            return

        # Otherwise, choose a candidate for this block
        codes_block, block_spos = blocks_data[block_index]
        num_block_cands = block_spos.shape[0]

        # Best future positives from *later* blocks
        future_best = best_from[block_index + 1]  # shape (num_perms,)

        for cand_idx in range(num_block_cands):
            spos_blk = block_spos[cand_idx]  # shape (num_perms,)

            blk_nonzero = (spos_blk != 0)
            if not blk_nonzero.any():
                # This block doesn't change any signs at all
                new_spos = spos_agg
            else:
                agg_nonzero = (spos_agg != 0)
                new_spos = spos_agg.copy()

                # Case 1: previously zero, block non-zero -> take block
                idx1 = (~agg_nonzero) & blk_nonzero
                if idx1.any():
                    blk1 = spos_blk[idx1]
                    # A fresh -1 here means earliest negative at coord 1,
                    # which can never be fixed (no j < 1). Immediate prune.
                    if (blk1 == -1).any():
                        continue
                    new_spos[idx1] = blk1

                # Case 2: both non-zero -> take earlier position
                idx2 = agg_nonzero & blk_nonzero
                if idx2.any():
                    pos_agg = np.abs(spos_agg)
                    pos_blk = np.abs(spos_blk)
                    earlier = pos_blk < pos_agg
                    idx_update = idx2 & earlier
                    if idx_update.any():
                        blk2 = spos_blk[idx_update]
                        # Again, a new -1 at coord 1 is hopeless.
                        if (blk2 == -1).any():
                            continue
                        new_spos[idx_update] = blk2

            # Extra pruning: check if any negative entry can ever be fixed
            neg_mask = new_spos < 0
            if neg_mask.any():
                k = np.abs(new_spos).astype(np.uint8)        # coordinate of earliest sign
                fb = future_best                             # best future positive
                # For perms where we already have a negative earliest sign:
                # if fb == 0 => no future positive possible
                # or fb >= k => no future positive at a *earlier* coordinate
                cannot_fix = neg_mask & ((fb == 0) | (fb >= k))
                if cannot_fix.any():
                    continue

            dfs(block_index + 1, idx_tuple + (cand_idx,), new_spos)

    # Start DFS with no blocks chosen and all-zero sign vector
    dfs(0, tuple(), np.zeros(num_perms, dtype=np.int8))

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
    print(len(find_all_codes(48, 3, 3)))


if __name__ == "__main__":
    main()
