"""
search.py
Search for good mirror codes.
"""

import itertools as it
import os
import pickle
import time
import sys
from functools import lru_cache

import numpy as np
from primefac import primefac

from isomorphism import (
    TimeLimitExceeded,
    set_timeout_checker,
    automorphisms_fixing_vectors,
    is_single_equivalence_class_under_shifts,
    lex_minimal_vectors,
    push_to_lex_minimal,
)
from mirror import MirrorCode
from util import find_strides, index_to_array, partitions


# ============================================================
# Global time limit (per *call* to find_all_codes)
# ============================================================

TIME_LIMIT_SECONDS = 36000


def _elapsed(start_time: float) -> float:
    """
    Helper: elapsed time in seconds from a given start_time.
    Uses time.monotonic() to avoid issues with wallclock adjustments.
    """
    return time.monotonic() - start_time


def _install_timeout_checker(start_time: float):
    """
    Install a timeout checker in the isomorphism module so that
    deep routines there can participate in the same global time
    limit as find_all_codes.
    """

    def _checker():
        if _elapsed(start_time) > TIME_LIMIT_SECONDS:
            raise TimeLimitExceeded(
                "Global time limit exceeded in isomorphism routines."
            )

    set_timeout_checker(_checker)


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
# 2) Minimal strings for a subgroup (iterative DFS, resumable)
# ============================================================

def _minstrings_inprogress_filename(Z_wt: int, X_wt: int, subgroup):
    """
    In-progress filename for minimal_strings_for_subgroup on a given block.
    Uses the same (p, lambdas) encoding style as the subgroup cache.
    """
    block = tuple(subgroup)
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

    os.makedirs("in_progress", exist_ok=True)
    fname = f"mstr_Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
    return os.path.join("in_progress", fname)


def _minstrings_cache_filename(Z_wt: int, X_wt: int, subgroup):
    """
    Final cache filename for minimal_strings_for_subgroup on a given block.
    Lives under subgroups/ and shares the same naming convention.
    """
    block = tuple(subgroup)
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
    fname = f"mstr_Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
    return os.path.join("subgroups", fname)


def minimal_strings_for_subgroup(Z_wt, X_wt, subgroup, start_time: float):
    """
    Compute lex-minimal strings for a given prime-power subgroup.

    subgroup: iterable of p-powers, all with the same prime p.

    Resumable + memory-aware:
      * DFS over candidate strings with (candidates, vec_indices) state.
      * Immediately filters full candidates with
        is_single_equivalence_class_under_shifts and only keeps
        those that pass in `good`.
      * On hitting TIME_LIMIT_SECONDS (based on start_time), saves
        {good, candidates, vec_indices} into in_progress/mstr_*.pkl
        and raises TimeLimitExceeded so the caller can checkpoint
        higher-level state.
      * On success, saves `good` to subgroups/mstr_*.pkl for reuse,
        and removes any in-progress snapshot.
    """
    subgroup = tuple(subgroup)
    factors = [list(primefac(s)) for s in subgroup]
    p = factors[0][0]
    lambdas = tuple(len(f) for f in factors)
    r = len(subgroup)

    cache_path = _minstrings_cache_filename(Z_wt, X_wt, subgroup)
    inprog_path = _minstrings_inprogress_filename(Z_wt, X_wt, subgroup)

    # If we already have the final result on disk, load and return.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        good = [np.asarray(code, dtype=np.int16) for code in data.get("good", [])]
        return good

    strides = np.array(find_strides(subgroup), dtype=int)
    subgroup_np = np.array(subgroup, dtype=int)
    group_size = int(np.prod(subgroup_np))

    # All group elements as vectors
    elems = [np.array(index_to_array(subgroup, idx), dtype=np.int16)
             for idx in range(group_size)]

    lex_min_raw = lex_minimal_vectors(p, lambdas)
    lex_min = [np.array(v, dtype=np.int16) for v in lex_min_raw]

    # Resume from snapshot if present
    if os.path.exists(inprog_path):
        with open(inprog_path, "rb") as f:
            data = pickle.load(f)
        good = data.get("good", [])
        candidates = data.get("candidates", [[np.zeros(r, dtype=np.int16)]])
        vec_indices = data.get("vec_indices", [0])
        # Ensure numpy dtypes are correct after unpickling
        candidates = [
            [np.asarray(v, dtype=np.int16) for v in layer]
            for layer in candidates
        ]
        good = [np.asarray(code, dtype=np.int16) for code in good]
    else:
        # Start fresh
        good = []
        candidates = [[np.zeros(r, dtype=np.int16)]]  # level 0
        vec_indices = [0]                             # select the zero vector at level 0

    total = Z_wt + X_wt

    try:
        while True:
            # Global time limit check
            if _elapsed(start_time) > TIME_LIMIT_SECONDS:
                raise TimeLimitExceeded(
                    "Time limit reached during minimal_strings_for_subgroup."
                )

            depth = len(vec_indices)
            last_idx = vec_indices[-1]
            current_layer = candidates[-1]

            # Backtrack if we've exhausted the current layer
            if last_idx >= len(current_layer):
                if depth == 1:
                    # Fully done
                    break
                # Pop this level and advance the index at the previous level
                candidates.pop()
                vec_indices.pop()
                vec_indices[-1] += 1
                continue

            # If we are not yet at full length, expand one more level
            if depth < total:
                fixed = tuple(
                    tuple(int(x) for x in candidates[level][vec_indices[level]])
                    for level in range(depth)
                )
                all_fixed_zero = all(all(x == 0 for x in f) for f in fixed)

                new_layer = []

                if all_fixed_zero:
                    # All-zero prefix.
                    if depth != Z_wt:
                        # Before the Z_wt-th vector: any lex-minimal representative
                        new_layer = lex_min
                    else:
                        # At depth == Z_wt, enforce the small-weight rule
                        for v_vec in lex_min:
                            if v_vec.max() > (1 if p == 2 else 0):
                                if p > 2:
                                    break
                            else:
                                new_layer.append(v_vec)
                else:
                    # Nontrivial constraints: use automorphisms_fixing_vectors with Z_wt
                    isos, shifts = automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed)
                    if isos.size != 0:
                        num_isos = isos.shape[0]
                        for v_vec in elems:
                            if _elapsed(start_time) > TIME_LIMIT_SECONDS:
                                raise TimeLimitExceeded(
                                    "Time limit reached during minimal_strings_for_subgroup "
                                    "(element loop)."
                                )

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
                                new_layer.append(v_vec)

                if not new_layer:
                    # No possible extension for this prefix: move on at this depth
                    vec_indices[-1] += 1
                    continue

                candidates.append(new_layer)
                vec_indices.append(0)
                continue

            # depth == total: we have a full candidate code
            code = np.stack(
                [candidates[level][vec_indices[level]] for level in range(total)],
                axis=0,
            )

            if is_single_equivalence_class_under_shifts(Z_wt, X_wt, subgroup, code):
                good.append(code)

            # Advance last coordinate at this depth
            vec_indices[-1] += 1

    except TimeLimitExceeded:
        # Save snapshot on any timeout (including those raised in isomorphism.py)
        snapshot = {
            "good": good,
            "candidates": candidates,
            "vec_indices": vec_indices,
        }
        with open(inprog_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        raise

    # Finished successfully: persist final result and clean up snapshots
    data = {"good": [np.asarray(code, dtype=np.int16) for code in good]}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(inprog_path):
        os.remove(inprog_path)

    return good


# ============================================================
# 3) Permutation bins (signed position, permutations inside)
#    + internal timeout checks
# ============================================================

def permutation_bins(
    Z_wt,
    X_wt,
    subgroup,
    candidates,
    precomputed=None,
    start_time: float | None = None,
):
    """
    For each candidate code on a prime-power subgroup and each permutation,
    compute the *earliest* non-zero sign and encode it as a single signed
    position:

        block_spos[cand, perm] = 0    : all zeros (no sign)
                                   k  : earliest non-zero sign = +1 at coord k
                                  -k  : earliest non-zero sign = -1 at coord k

    where 1 <= k <= total-1 and total = Z_wt + X_wt.

    Optimisation:
      * If `precomputed` is provided (a tuple
          (p, lambdas, strides, subgroup_np, perms_np, total)),
        we reuse it instead of recomputing subgroup data and permutations.
      * We also check TIME_LIMIT_SECONDS against `start_time` and raise
        TimeLimitExceeded if over budget.
    """
    if start_time is None:
        start_time = time.monotonic()

    if precomputed is None:
        subgroup = tuple(subgroup)
        factors = [list(primefac(s)) for s in subgroup]
        p = factors[0][0]
        lambdas = tuple(len(f) for f in factors)
        strides = np.array(find_strides(subgroup), dtype=int)
        subgroup_np = np.array(subgroup, dtype=int)
        total = Z_wt + X_wt
        perms = get_perms(Z_wt, X_wt)
        perms_np = np.array(perms, dtype=np.int32)
    else:
        p, lambdas, strides, subgroup_np, perms_np, total = precomputed

    num_perms = perms_np.shape[0]
    num_cands = len(candidates)

    if num_cands == 0 or num_perms == 0:
        return np.zeros((num_cands, num_perms), dtype=np.int8)

    block_spos = np.zeros((num_cands, num_perms), dtype=np.int8)

    for cand_ind, cand in enumerate(candidates):
        # Time check per candidate
        if _elapsed(start_time) > TIME_LIMIT_SECONDS:
            raise TimeLimitExceeded(
                "Time limit reached during permutation_bins (per candidate)."
            )

        cand = np.asarray(cand, dtype=np.int64)
        # Precompute base lex values for cand rows
        base_lex = (cand @ strides).astype(int)

        # Precompute automorphisms (and shifts) for prefixes of cand:
        # prefix i means we fix cand[0], ..., cand[i-1]
        total_rows = base_lex.shape[0]
        prefix_auts = [None] * total_rows
        all_zero = [False] * total_rows
        for i in range(1, total_rows):
            fixed = tuple(
                tuple(int(x) for x in cand[j])
                for j in range(i)
            )
            all_zero[i] = all(all(x == 0 for x in f) for f in fixed)
            if not all_zero[i]:
                isos_i, shifts_i = automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed)
                prefix_auts[i] = (isos_i, shifts_i)

        for perm_ind in range(num_perms):
            # Time check per permutation
            if _elapsed(start_time) > TIME_LIMIT_SECONDS:
                raise TimeLimitExceeded(
                    "Time limit reached during permutation_bins (per permutation)."
                )

            perm = perms_np[perm_ind]

            # Reorder and normalise by translation so that first row is zero
            c = cand[perm] - cand[perm[0]]

            for i in range(1, total_rows):
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


def _subgroup_inprogress_filename(Z_wt: int, X_wt: int, block):
    """
    Same naming convention as _subgroup_cache_filename, but under in_progress/.
    Used to store partial results for (Z_wt, X_wt, block) across permutation_bins.
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

    os.makedirs("in_progress", exist_ok=True)
    fname = f"Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
    return os.path.join("in_progress", fname)


def _subgroup_codes_and_bins(Z_wt: int, X_wt: int, block, start_time: float):
    """
    Compute (or load from disk) both the minimal strings and the permutation
    signed-position data for a given (Z_wt, X_wt, block).

    Returns:
        codes_block : list of np.ndarray, each shape (Z_wt+X_wt, len(block))
        block_spos  : np.ndarray, shape (num_cands, num_perms), int8

    Checkpointing + time limit:
      - Uses minimal_strings_for_subgroup (now resumable) and permutation_bins
        (resumable per candidate).
      - If TIME_LIMIT_SECONDS exceeded at any point, saves snapshot and raises
        TimeLimitExceeded up the call stack.
    """
    block = tuple(block)
    cache_path = _subgroup_cache_filename(Z_wt, X_wt, block)
    inprog_path = _subgroup_inprogress_filename(Z_wt, X_wt, block)

    # Precompute subgroup + permutation context once per block
    factors = [list(primefac(s)) for s in block]
    p = factors[0][0]
    lambdas = tuple(len(f) for f in factors)
    strides = np.array(find_strides(block), dtype=int)
    subgroup_np = np.array(block, dtype=int)
    total = Z_wt + X_wt
    perms = get_perms(Z_wt, X_wt)
    perms_np = np.array(perms, dtype=np.int32)
    num_perms = perms_np.shape[0]
    perm_ctx = (p, lambdas, strides, subgroup_np, perms_np, total)

    # 1) If we already have the final result, just load and return.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        codes_block = [np.asarray(code, dtype=np.int16) for code in data["codes_block"]]
        block_spos = np.asarray(data["block_spos"], dtype=np.int8)
        return codes_block, block_spos

    # 2) Check for an in-progress snapshot of this subgroup's permutation_bins.
    if os.path.exists(inprog_path):
        with open(inprog_path, "rb") as f:
            data = pickle.load(f)
        codes_block = [np.asarray(code, dtype=np.int16) for code in data["codes_block"]]
        block_spos = np.asarray(data["block_spos"], dtype=np.int8)
        next_cand = int(data.get("next_cand", 0))
    else:
        # No permutation-bins snapshot: start by computing minimal strings
        codes_block = minimal_strings_for_subgroup(Z_wt, X_wt, block, start_time=start_time)
        codes_block = [np.asarray(code, dtype=np.int16) for code in codes_block]

        if not codes_block:
            block_spos = np.zeros((0, num_perms), dtype=np.int8)
            data = {
                "codes_block": codes_block,
                "block_spos": block_spos,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(inprog_path):
                os.remove(inprog_path)
            return codes_block, block_spos

        block_spos = np.zeros((len(codes_block), num_perms), dtype=np.int8)
        next_cand = 0

    # 3) Process remaining candidates one by one, checking the time limit.
    num_cands = len(codes_block)

    while next_cand < num_cands:
        if _elapsed(start_time) > TIME_LIMIT_SECONDS:
            snapshot = {
                "codes_block": codes_block,
                "block_spos": block_spos,
                "next_cand": next_cand,
            }
            with open(inprog_path, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            raise TimeLimitExceeded(
                "Time limit reached during _subgroup_codes_and_bins; "
                "partial state saved in in_progress/."
            )

        cand_code = [codes_block[next_cand]]

        try:
            cand_spos = permutation_bins(
                Z_wt,
                X_wt,
                block,
                cand_code,
                precomputed=perm_ctx,
                start_time=start_time,
            )
        except TimeLimitExceeded:
            # Save the candidates processed so far (0..next_cand-1) and propagate.
            snapshot = {
                "codes_block": codes_block,
                "block_spos": block_spos,
                "next_cand": next_cand,
            }
            with open(inprog_path, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            raise

        cand_spos = np.asarray(cand_spos, dtype=np.int8)
        if cand_spos.shape != (1, num_perms):
            raise RuntimeError(
                f"permutation_bins returned shape {cand_spos.shape} "
                f"for a single candidate; expected (1, {num_perms})."
            )

        block_spos[next_cand: next_cand + 1, :] = cand_spos
        next_cand += 1

    data = {
        "codes_block": codes_block,
        "block_spos": block_spos,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(inprog_path):
        os.remove(inprog_path)

    return codes_block, block_spos


# ============================================================
# 5) Find all codes in a given abelian group (with checkpointing)
# ============================================================

def _group_inprogress_filename(Z_wt: int, X_wt: int, group):
    """
    In-progress filename for a single abelian group in find_all_codes_in_group.
    One file per (Z_wt, X_wt, group) under in_progress/.
    """
    group = tuple(int(g) for g in group)
    group_str = "_".join(str(g) for g in group)
    os.makedirs("in_progress", exist_ok=True)
    fname = f"group_Z{Z_wt}_X{X_wt}_g{group_str}.pkl"
    return os.path.join("in_progress", fname)


def find_all_codes_in_group(
    Z_wt,
    X_wt,
    group,
    min_k=3,
    return_k=True,
    start_time: float | None = None,
):
    """
    Finds all codes of weights Z_wt and X_wt for a given group.

    group: tuple of prime powers.

    Memory-optimised and checkpoint-aware version:
      - Uses iterative DFS with a stack.
      - Checkpoints DFS state + results to in_progress/ if time limit exceeded.
      - On time limit, raises TimeLimitExceeded.
    """
    if start_time is None:
        start_time = time.monotonic()

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
    num_blocks = len(blocks)

    # Preload all blocks' codes and signed positions once
    blocks_data = []
    for block in blocks:
        if _elapsed(start_time) > TIME_LIMIT_SECONDS:
            raise TimeLimitExceeded(
                "Time limit reached during find_all_codes_in_group (preload blocks)."
            )
        codes_block, block_spos = _subgroup_codes_and_bins(Z_wt, X_wt, block, start_time=start_time)
        if not codes_block:
            return []
        codes_block = [np.asarray(code, dtype=np.int16) for code in codes_block]
        block_spos = np.asarray(block_spos, dtype=np.int8)
        blocks_data.append((codes_block, block_spos))

    # Precompute best possible positive coordinate per block & perm
    best_pos = np.zeros((num_blocks, num_perms), dtype=np.uint8)
    for b, (_, block_spos) in enumerate(blocks_data):
        bs = block_spos  # (num_cands_b, num_perms), int8
        arr = np.where(bs > 0, bs.astype(np.int16), 127)
        min_pos = arr.min(axis=0)  # int16
        min_pos_u8 = min_pos.astype(np.uint8)
        min_pos_u8[min_pos_u8 == 127] = 0
        best_pos[b, :] = min_pos_u8

    best_from = np.zeros((num_blocks + 1, num_perms), dtype=np.uint8)
    for b in range(num_blocks - 1, -1, -1):
        if b == num_blocks - 1:
            best_from[b, :] = best_pos[b, :]
        else:
            best_from[b, :] = np.minimum(best_pos[b, :], best_from[b + 1, :])

    inprog_path = _group_inprogress_filename(Z_wt, X_wt, group)

    good_raw = []
    stack = []

    # Resume DFS if we have a saved group-level snapshot
    if os.path.exists(inprog_path):
        with open(inprog_path, "rb") as f:
            data = pickle.load(f)
        good_raw = data.get("good_raw", [])
        raw_stack = data.get("stack", [])
        stack = []
        for frame in raw_stack:
            block_index, cand_idx, idx_tuple, spos_list = frame
            spos_arr = np.asarray(spos_list, dtype=np.int8)
            stack.append((int(block_index), int(cand_idx), tuple(idx_tuple), spos_arr))
    else:
        spos0 = np.zeros(num_perms, dtype=np.int8)
        stack.append((0, 0, tuple(), spos0))

    try:
        while stack:
            # Group-level time check; any timeout here is handled
            # by the outer except block which snapshots the DFS state.
            if _elapsed(start_time) > TIME_LIMIT_SECONDS:
                raise TimeLimitExceeded(
                    "Time limit reached during find_all_codes_in_group; "
                    "partial state will be saved in in_progress/."
                )

            block_index, cand_idx, idx_tuple, spos_agg = stack.pop()

            if block_index == num_blocks:
                if (spos_agg < 0).any():
                    continue

                vecs_parts = []
                for b, block_cand_idx in enumerate(idx_tuple):
                    codes_block_b, _ = blocks_data[b]
                    vecs_parts.append(codes_block_b[block_cand_idx])
                vecs = np.concatenate(vecs_parts, axis=1)

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

                is_css = code.is_CSS()
                good_raw.append((z_part, x_part, bool(is_css), int(k_val)))
                continue

            codes_block, block_spos = blocks_data[block_index]
            num_block_cands = block_spos.shape[0]

            if cand_idx >= num_block_cands:
                continue

            # Schedule the next candidate at this block on the stack
            stack.append((block_index, cand_idx + 1, idx_tuple, spos_agg))

            # Combine signs with this candidate
            spos_blk = block_spos[cand_idx]
            blk_nonzero = (spos_blk != 0)

            new_spos = spos_agg
            if blk_nonzero.any():
                agg_nonzero = (spos_agg != 0)
                new_spos = spos_agg.copy()

                # Case 1: previously zero, block non-zero -> take block
                idx1 = (~agg_nonzero) & blk_nonzero
                if idx1.any():
                    blk1 = spos_blk[idx1]
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
                        if (blk2 == -1).any():
                            continue
                        new_spos[idx_update] = blk2

            # Prune branches where a negative earliest sign cannot be fixed
            neg_mask = new_spos < 0
            if neg_mask.any():
                k = np.abs(new_spos).astype(np.uint8)
                fb = best_from[block_index + 1]
                cannot_fix = neg_mask & ((fb == 0) | (fb >= k))
                if cannot_fix.any():
                    continue

            stack.append((block_index + 1, 0, idx_tuple + (cand_idx,), new_spos))

    except TimeLimitExceeded:
        # Snapshot the current DFS state so we can resume this group later.
        raw_stack = []
        for (b_idx, c_idx, idx_tuple, spos_agg) in stack:
            raw_stack.append((b_idx, c_idx, tuple(idx_tuple), spos_agg.tolist()))
        snapshot = {
            "good_raw": good_raw,
            "stack": raw_stack,
        }
        with open(inprog_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        raise

    # Finished successfully for this group: remove any group-level in_progress file
    if os.path.exists(inprog_path):
        os.remove(inprog_path)

    if not good_raw:
        return []

    if return_k:
        return [
            (group, z_part, x_part, is_css, k_val)
            for (z_part, x_part, is_css, k_val) in good_raw
        ]
    else:
        return [
            (group, z_part, x_part, is_css)
            for (z_part, x_part, is_css, k_val) in good_raw
        ]


# ============================================================
# 6) Top-level search over all groups of size n (single-threaded)
# ============================================================

def _codes_partial_filename(n: int, Z_wt: int, X_wt: int, min_k: int):
    """
    Filename for saving partial results (codes found so far) for a given
    (n, Z_wt, X_wt, min_k). Lives under in_progress/.
    """
    os.makedirs("in_progress", exist_ok=True)
    return os.path.join(
        "in_progress",
        f"codes_n{n}_Z{Z_wt}_X{X_wt}_k{min_k}.pkl",
    )


def find_all_codes(n, Z_wt, X_wt, min_k=3):
    """
    Finds all codes for a given number of qubits, n, of given weight.

    Single-threaded, but with checkpointing via in_progress/ for:
      - minimal_strings_for_subgroup
      - _subgroup_codes_and_bins
      - find_all_codes_in_group

    Time limit:
      - Each call to find_all_codes(n, Z_wt, X_wt) has a 10-hour budget
        (TIME_LIMIT_SECONDS) measured from the beginning of this call.
      - If exceeded during this call, we:
          * save the codes found so far (plus index of next group) to a file
            under in_progress/,
          * call sys.exit(msg) with a short message (printed to stderr),
            so the caller cannot proceed.
      - On success (no timeout), we return the full list of codes and
        clean up any stale partial "codes_*" file.
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

    groups = n_partitions(n)
    if not groups:
        return []

    start_time = time.monotonic()
    _install_timeout_checker(start_time)

    partial_path = _codes_partial_filename(n, Z_wt, X_wt, min_k)

    # Resume from a previous top-level snapshot if present
    if os.path.exists(partial_path):
        with open(partial_path, "rb") as f:
            data = pickle.load(f)
        results = data.get("results", [])
        next_group_index = int(data.get("next_group_index", 0))
    else:
        results = []
        next_group_index = 0

    try:
        for idx in range(next_group_index, len(groups)):
            group = groups[idx]

            # Quick coarse check before going into group-level work
            if _elapsed(start_time) > TIME_LIMIT_SECONDS:
                raise TimeLimitExceeded(
                    "Time limit reached before processing next group."
                )

            more = find_all_codes_in_group(
                Z_wt,
                X_wt,
                group,
                min_k=min_k,
                return_k=(min_k > 0),
                start_time=start_time,
            )
            results.extend(more)

            # We have successfully finished group idx
            next_group_index = idx + 1

    except TimeLimitExceeded:
        # We hit the time limit somewhere. All lower-level state has
        # already been snapshot to in_progress/; we now save the codes
        # found so far and the index of the next group, then terminate.
        snapshot = {
            "results": results,
            "next_group_index": next_group_index,
        }
        with open(partial_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        msg = (
            f"Time limit reached in find_all_codes(n={n}, Z_wt={Z_wt}, "
            f"X_wt={X_wt}, min_k={min_k}); partial results and progress "
            f"saved to '{partial_path}'."
        )
        sys.exit(msg)

    # Completed within time: clean up any stale partial file, then return
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return results


# ============================================================
# 7) Main (simple test harness)
# ============================================================

def main():
    print(len(find_all_codes(24, 3, 3)))


if __name__ == "__main__":
    main()
