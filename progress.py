#!/usr/bin/env python3
"""
progress_report.py

Estimate progress of find_all_codes(n, Z_wt, X_wt) from checkpoint filenames
in the in_progress/ directory. Only uses filenames, never opens files.
"""

import os
import re
from typing import Dict, List, Tuple

from search import n_partitions  # your existing function


# ---------- helpers ----------

def factor_prime_power(n: int) -> Tuple[int, int]:
    """Return (p, k) such that n = p**k assuming n is a prime power."""
    if n < 2:
        raise ValueError(f"{n} is not a valid prime power >= 2")
    p = 2
    while p * p <= n and n % p != 0:
        p += 1
    if n % p != 0:
        p = n
    k = 0
    m = n
    while m % p == 0:
        m //= p
        k += 1
    if m != 1:
        raise ValueError(f"{n} is not a pure prime power")
    return p, k


def split_into_blocks(group: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Split group into prime blocks as in find_all_codes_in_group."""
    group = tuple(group)
    blocks: List[Tuple[int, ...]] = []
    i = 0
    L = len(group)
    while i < L:
        power = group[i]
        p, _ = factor_prime_power(power)
        block = [power]
        i += 1
        while i < L:
            nxt = group[i]
            if nxt % p != 0:
                break
            block.append(nxt)
            i += 1
        blocks.append(tuple(block))
    return blocks


def block_name_components(block: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """For block = (p**λ1,...,p**λr), return (p, (λ1,...,λr))."""
    p, first_exp = factor_prime_power(block[0])
    lambdas = [first_exp]
    for n in block[1:]:
        p2, k2 = factor_prime_power(n)
        if p2 != p:
            raise ValueError(f"block {block} mixes primes")
        lambdas.append(k2)
    return p, tuple(lambdas)


def possible_inprogress_names_for_job(Z_wt: int, X_wt: int, n: int) -> List[str]:
    """
    For job (n, Z_wt, X_wt), return the ordered list of all filenames
    that find_all_codes(n, Z_wt, X_wt) could create in in_progress/.
    """
    groups = n_partitions(n)
    seen = set()
    order: List[str] = []

    for group in groups:
        group = tuple(int(g) for g in group)
        blocks = split_into_blocks(group)

        # block-level files
        for block in blocks:
            p, lambdas = block_name_components(block)
            exps_str = "_".join(str(e) for e in lambdas)

            mstr_name = f"mstr_Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
            if mstr_name not in seen:
                seen.add(mstr_name)
                order.append(mstr_name)

            sub_name = f"Z{Z_wt}_X{X_wt}_p{p}_l{exps_str}.pkl"
            if sub_name not in seen:
                seen.add(sub_name)
                order.append(sub_name)

        # group-level file
        group_str = "_".join(str(g) for g in group)
        g_name = f"group_Z{Z_wt}_X{X_wt}_g{group_str}.pkl"
        if g_name not in seen:
            seen.add(g_name)
            order.append(g_name)

    return order


# ---------- parse filenames in in_progress/ ----------

GROUP_RE = re.compile(r"^group_Z(\d+)_X(\d+)_g([\d_]+)\.pkl$")
MSTR_RE = re.compile(r"^mstr_Z(\d+)_X(\d+)_p(\d+)_l([\d_]+)\.pkl$")
SUBG_RE = re.compile(r"^Z(\d+)_X(\d+)_p(\d+)_l([\d_]+)\.pkl$")


def classify_inprogress_files(inprog_dir: str):
    """
    Return dict with keys: 'group', 'mstr', 'subg', 'unknown'.
    Each value is a list of small dicts with parsed metadata.
    """
    group_entries = []
    mstr_entries = []
    subg_entries = []
    unknown_entries = []

    for fname in os.listdir(inprog_dir):
        if not fname.endswith(".pkl"):
            continue

        m = GROUP_RE.match(fname)
        if m:
            Z = int(m.group(1))
            X = int(m.group(2))
            group_str = m.group(3)
            group = tuple(int(x) for x in group_str.split("_"))
            n = 1
            for g in group:
                n *= g
            group_entries.append(
                {"name": fname, "Z": Z, "X": X, "group": group, "n": n}
            )
            continue

        m = MSTR_RE.match(fname)
        if m:
            Z = int(m.group(1))
            X = int(m.group(2))
            p = int(m.group(3))
            lambdas = tuple(int(x) for x in m.group(4).split("_"))
            mstr_entries.append(
                {"name": fname, "Z": Z, "X": X, "p": p, "lambdas": lambdas}
            )
            continue

        m = SUBG_RE.match(fname)
        if m:
            Z = int(m.group(1))
            X = int(m.group(2))
            p = int(m.group(3))
            lambdas = tuple(int(x) for x in m.group(4).split("_"))
            subg_entries.append(
                {"name": fname, "Z": Z, "X": X, "p": p, "lambdas": lambdas}
            )
            continue

        unknown_entries.append({"name": fname})

    return {
        "group": group_entries,
        "mstr": mstr_entries,
        "subg": subg_entries,
        "unknown": unknown_entries,
    }


# ---------- main ----------

def main():
    inprog_dir = "in_progress"
    if not os.path.isdir(inprog_dir):
        print("no in_progress directory")
        return

    entries = classify_inprogress_files(inprog_dir)
    group_entries = entries["group"]
    mstr_entries = entries["mstr"]
    subg_entries = entries["subg"]

    # infer jobs (n, Z, X) only from group_* files
    job_keys = sorted(set((e["Z"], e["X"], e["n"]) for e in group_entries))
    if not job_keys:
        print("no group_* checkpoints, no jobs inferred")
        return

    # build conceptual sequences for each job
    jobs_data: Dict[Tuple[int, int, int], Tuple[List[str], Dict[str, int]]] = {}
    for (Z, X, n) in job_keys:
        names = possible_inprogress_names_for_job(Z, X, n)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        jobs_data[(Z, X, n)] = (names, name_to_idx)

    # gather all in_progress filenames once
    all_files = [f for f in os.listdir(inprog_dir) if f.endswith(".pkl")]

    print("n, Z, X, progress:")
    for (Z, X, n) in sorted(job_keys, key=lambda t: t[2]):  # sort by n
        names, name_to_idx = jobs_data[(Z, X, n)]

        best_frac = 0.0
        total_steps = len(names)
        if total_steps == 0:
            continue

        # use any in-progress file that belongs to this job and matches a step
        for fname in all_files:
            # must match this Z, X
            if f"_Z{Z}_" not in fname or f"_X{X}_" not in fname:
                continue
            if fname not in name_to_idx:
                continue
            idx = name_to_idx[fname]
            # model "within this step" as 50% of the step
            frac = (idx + 0.5) / total_steps
            if frac > best_frac:
                best_frac = frac

        # never print exactly 100%
        if best_frac >= 1.0:
            best_frac = 0.99

        print(f"n={n} Z={Z} X={X} progress={best_frac*100:5.1f}%")

if __name__ == "__main__":
    main()
