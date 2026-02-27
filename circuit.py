"""
SMT (Z3) solver for the value-assignment problem described, with:
  1) a wrapper that relaxes the allowed value range from 1..w to 1..W for W = w, w+1, ...
  2) a cached wrapper that accepts a MirrorCode-like object and memoizes results on disk.

Install:
  pip install z3-solver

Input:
  bits: list[list[int]] of shape n x (2n), entries in {0,1}

Main entry points:
  - solve_value_assignment(bits, value_cap=None)
      * If value_cap is None: uses W = w (row max weight)
      * Else: uses W = value_cap

  - solve_with_relaxation(bits, max_extra=None, start_extra=0, verbose=True)
      * Tries W = w + start_extra, w + start_extra + 1, ...
      * Stops when a solution is found or after max_extra (if provided)

  - cached_schedule(code)
      * Calls solve_with_relaxation(code.get_stabilizers(), max_extra=6, verbose=False)
      * Persists results under ./schedules/ keyed by stabilizer content
      * Reuses existing cached file if stabilizers match

This file written by ChatGPT-5.2 under guidance of ABK.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
from util import find_strides
import itertools as it

from z3 import And, Distinct, Int, Not, Solver, Xor, group, sat


def compress_binary_matrix(bits: List[List[int]]) -> Tuple[List[List[str]], int]:
    n = len(bits)
    if n == 0:
        return [], 0
    if any(len(r) != 2 * n for r in bits):
        raise ValueError("Input must have n rows and exactly 2n columns.")

    compressed = [["I"] * n for _ in range(n)]

    def term(a: int, b: int) -> str:
        # a = bits[i][j], b = bits[i][j+n]
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("Input entries must be 0/1.")
        if a == 0 and b == 0:
            return "I"
        if a == 1 and b == 0:
            return "Z"
        if a == 0 and b == 1:
            return "X"
        return "Y"  # a==1 and b==1

    row_weights = [0] * n
    for i in range(n):
        for j in range(n):
            t = term(bits[i][j], bits[i][j + n])
            compressed[i][j] = t
            if t != "I":
                row_weights[i] += 1

    w = max(row_weights) if n > 0 else 0
    return compressed, w


def solve_value_assignment(
    code,
    value_cap: Optional[int] = None,
):
    """
    Solve the instance with allowed assigned values in {1..W},
    where W = value_cap if provided, else W = w (max row weight).

    Returns:
      (is_sat, compressed, w, W, values_or_none)

    where values_or_none is an n x n matrix: int for non-I, None for I.
    """
    bits = _normalize_bits(code.get_stabilizers())
    compressed, w = compress_binary_matrix(bits)
    n = len(compressed)

    if code.is_CSS() and value_cap >= 2 * (code.wz + code.wx):
        # For CSS codes, the anti-commutation structure is simple enough that
        # we can assign values greedily without needing a full SMT solver.
        values: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]
        val = 1
        strides = find_strides(code.group)
        for i in 'ZX':
            for j in code.z0:
                for k, g in enumerate(it.product(*[range(a) for a in code.group])):
                    assert compressed[k][np.mod(j + g, code.group) @ strides] != 'I'
                    if compressed[k][np.mod(j + g, code.group) @ strides] == i:
                        values[k][np.mod(j + g, code.group) @ strides] = val
                val += 1
            for j in code.x0:
                for k, g in enumerate(it.product(*[range(a) for a in code.group])):
                    assert compressed[k][np.mod(j - g, code.group) @ strides] != 'I'
                    if compressed[k][np.mod(j - g, code.group) @ strides] == i:
                        values[k][np.mod(j - g, code.group) @ strides] = val
                val += 1
        return True, compressed, w, 2 * (code.wz + code.wx), values

    # Trivial cases
    if n == 0:
        return True, compressed, 0, 0, []
    if w == 0:
        return True, compressed, 0, 0, [[None for _ in range(n)] for _ in range(n)]

    W = w if value_cap is None else int(value_cap)
    if W <= 0:
        raise ValueError("value_cap must be a positive integer (or None).")
    if W < w:
        # Allowed to ask for W<w, but it cannot help; treat as UNSAT early.
        return False, compressed, w, W, None

    # Collect sparse structure: at most w non-I per row/col.
    row_cols: List[List[int]] = [[] for _ in range(n)]                 # columns with non-I in each row
    col_entries: List[List[Tuple[int, str]]] = [[] for _ in range(n)]  # (row, term) for each col

    for i in range(n):
        for j in range(n):
            t = compressed[i][j]
            if t != "I":
                row_cols[i].append(j)
                col_entries[j].append((i, t))

    # Sanity check on the prompt's guarantees
    if any(len(col_entries[j]) > w for j in range(n)):
        raise ValueError("Input violates guarantee: a column has > w non-I terms.")

    s = Solver()

    # Create integer vars for each non-I site
    var: Dict[Tuple[int, int], "Int"] = {}
    for i in range(n):
        for j in row_cols[i]:
            v = Int(f"v_{i}_{j}")
            var[(i, j)] = v
            s.add(And(v >= 1, v <= W))

    # Row uniqueness: no two equal values in same row
    for i in range(n):
        vs = [var[(i, j)] for j in row_cols[i]]
        if len(vs) >= 2:
            s.add(Distinct(*vs))

    # Column uniqueness: no two equal values in same column
    for j in range(n):
        vs = [var[(i, j)] for (i, _t) in col_entries[j]]
        if len(vs) >= 2:
            s.add(Distinct(*vs))

    # Anti-commutation sites per row-pair by scanning columns.
    pair_sites = defaultdict(list)  # (a,b) -> [j1, j2, ...]
    for j in range(n):
        entries = col_entries[j]  # size <= w
        for (i, ti), (k, tk) in combinations(entries, 2):
            if ti != tk:  # anti-commutation site
                a, b = (i, k) if i < k else (k, i)
                pair_sites[(a, b)].append(j)

    # Parity constraint:
    # For each pair (a,b), the number of anti-comm sites where value[a,j] > value[b,j] is even.
    for (a, b), cols in pair_sites.items():
        # If the input truly has commuting rows, len(cols) should be even.
        if len(cols) % 2 == 1:
            return False, compressed, w, W, None

        bs = [(var[(a, j)] > var[(b, j)]) for j in cols]
        if bs:
            s.add(Not(Xor(*bs)))  # even parity

    if s.check() != sat:
        return False, compressed, w, W, None

    m = s.model()
    values: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]
    for (i, j), v in var.items():
        values[i][j] = m.evaluate(v, model_completion=True).as_long()

    return True, compressed, w, W, values


# def solve_with_relaxation(
#     code,
#     max_extra: Optional[int] = None,
#     start_extra: int = 0,
#     verbose: bool = True,
# ):
#     """
#     Repeatedly try increasing caps W = w + extra until SAT is found.

#     Args:
#       max_extra: if None, loop forever until SAT; else stop after extra > max_extra
#       start_extra: start at extra = start_extra (default 0 means try W=w first)
#       verbose: print status lines each iteration

#     Returns:
#       Same tuple as solve_value_assignment: (True, compressed, w, W, values)
#       or, if max_extra is set and no solution is found up to it:
#         (False, compressed, w, W_last_tried, None)
#     """
#     # Compute w once (also validates shape).
#     bits = _normalize_bits(code.get_stabilizers())
#     compressed, w = compress_binary_matrix(bits)

#     # Handle degenerate cases quickly.
#     n = len(compressed)
#     if n == 0:
#         if verbose:
#             print("[relax] n=0: vacuously SAT")
#         return True, compressed, 0, 0, []
#     if w == 0:
#         if verbose:
#             print("[relax] w=0 (all I): vacuously SAT")
#         return True, compressed, 0, 0, [[None for _ in range(n)] for _ in range(n)]

#     extra = int(start_extra)
#     if extra < 0:
#         raise ValueError("start_extra must be >= 0.")
#     if max_extra is not None and max_extra < extra:
#         raise ValueError("max_extra must be >= start_extra (or None).")

#     last_W = None
#     while True:
#         W = w + extra
#         last_W = W
#         if verbose:
#             print(f"[relax] trying value range 1..{W} (w={w}, extra={extra}) ...", flush=True)

#         ok, _compressed2, _w2, W2, values = solve_value_assignment(code, value_cap=W)

#         if ok:
#             if verbose:
#                 print(f"[relax] SAT found with W={W2}")
#             return True, compressed, w, W2, values

#         if max_extra is not None and extra >= max_extra:
#             if verbose:
#                 print(f"[relax] UNSAT up to W={W} (max_extra={max_extra}). Stopping.")
#             return False, compressed, w, last_W, None

#         extra += 1

# ------------------------- Disk cache wrapper -------------------------

def _normalize_bits(stabilizers: Any) -> List[List[int]]:
    """
    Normalize various array-like inputs into list[list[int]] with 0/1 entries.
    Intended to handle common cases like numpy arrays, tuples, etc.
    """
    if stabilizers is None:
        raise ValueError("code.get_stabilizers() returned None")

    # Convert outer container to list
    rows = stabilizers.tolist() if hasattr(stabilizers, "tolist") else stabilizers
    rows = list(rows)

    bits: List[List[int]] = []
    for r in rows:
        rr = r.tolist() if hasattr(r, "tolist") else r
        rr = list(rr)
        bits.append([int(x) for x in rr])

    # Basic validation (and ensure entries are 0/1)
    n = len(bits)
    if n == 0:
        return bits
    if any(len(r) != 2 * n for r in bits):
        raise ValueError("Stabilizers must have shape n x (2n).")
    for i in range(n):
        for j in range(2 * n):
            if bits[i][j] not in (0, 1):
                raise ValueError("Stabilizer entries must be 0/1.")
    return bits


def _stabilizers_fingerprint(code) -> str:
    """
    Generate a human-readable key suitable for use as a filename.

    The old implementation returned a SHA256 hash of the stabilizer
    matrix.  The new behaviour encodes the three defining pieces of a
    MirrorCode-like object instead:

        * ``code.group`` – a list of integers
        * ``code.z0``   – a 2‑D array of integers
        * ``code.x0``   – a 2‑D array of integers

    These three values are JSON‑serialized and then sanitised so that the
    resulting string contains only alphanumeric characters, hyphens,
    underscores or dots.  That makes the cache files readable by a human
    and still safe as filenames on most platforms.
    """
    # collect the required attributes; let Python raise an AttributeError
    # if any of them is missing so that callers notice soon.
    group = list(code.group)

    def _to_list2d(arr: Any) -> List[List[int]]:
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        arr = list(arr)
        return [list(row) for row in arr]

    z0 = _to_list2d(code.z0)
    x0 = _to_list2d(code.x0)

    payload = {"group": group, "z0": z0, "x0": x0}
    s = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    # make filesystem friendly: keep only a small safe subgroup
    safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in s)
    return safe


def cached_schedule(code: Any):
    """
    Short, descriptive cached wrapper.

    Accepts a MirrorCode-like object `code` and runs:
      solve_with_relaxation(code.get_stabilizers(), max_extra=6, verbose=False)

    Results are cached under ./schedules/ keyed by the stabilizer content.
    If the same stabilizers appear again, returns the cached result instead of recomputing.

    Returns:
      (ok, compressed, w, W, values)
    """
    stabs = code.get_stabilizers()
    bits = _normalize_bits(stabs)
    key = _stabilizers_fingerprint(code)
    _, best_w = compress_binary_matrix(bits)

    schedules_dir = Path("schedules")
    schedules_dir.mkdir(parents=True, exist_ok=True)

    path = schedules_dir / f"{key}.json"

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Minimal structural validation; if it fails, we recompute.
            if (
                isinstance(data, dict)
                and "ok" in data
                and "compressed" in data
                and "w" in data
                and "W" in data
                and "values" in data
            ):
                return (
                    bool(data["ok"]),
                    data["compressed"],
                    int(data["w"]),
                    int(data["W"]),
                    data["values"],
                )
        except Exception:
            # Fall through to recompute on any read/parse error.
            pass

    value_cap = 16
    while value_cap > best_w:
        value_cap -= 1
        ok, compressed, w, W, values = solve_value_assignment(code, value_cap=value_cap)
        if ok:
            data = {
                "ok": ok,
                "compressed": compressed,
                "w": w,
                "W": W,
                "values": values,
                "stabilizers": bits,  # stored for debugging / collision resistance
            }

            # Atomic-ish write: write temp then replace.
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            tmp.replace(path)
        else:
            # No need to try smaller value_cap if this is already UNSAT.
            break

    return data["ok"], data["compressed"], data["w"], data["W"], data["values"]
