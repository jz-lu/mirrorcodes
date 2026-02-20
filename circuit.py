"""
SMT (Z3) solver for the value-assignment problem described, with a wrapper that
relaxes the allowed value range from 1..w to 1..W for W = w, w+1, w+2, ...

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
"""

from __future__ import annotations

from mirror import MirrorCode
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from z3 import And, Distinct, Int, Not, Solver, Xor, sat


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
    bits: List[List[int]],
    value_cap: Optional[int] = None,
):
    """
    Solve the instance with allowed assigned values in {1..W},
    where W = value_cap if provided, else W = w (max row weight).

    Returns:
      (is_sat, compressed, w, W, values_or_none)

    where values_or_none is an n x n matrix: int for non-I, None for I.
    """
    compressed, w = compress_binary_matrix(bits)
    n = len(compressed)

    # Trivial cases
    if n == 0:
        return True, compressed, 0, 0, []
    if w == 0:
        return True, compressed, 0, 0, [[None for _ in range(n)] for _ in range(n)]

    W = w if value_cap is None else int(value_cap)
    if W <= 0:
        raise ValueError("value_cap must be a positive integer (or None).")
    if W < w:
        # It's allowed to ask for W<w, but it can never help; treat as UNSAT early.
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


def solve_with_relaxation(
    bits: List[List[int]],
    max_extra: Optional[int] = None,
    start_extra: int = 0,
    verbose: bool = True,
):
    """
    Repeatedly try increasing caps W = w + extra until SAT is found.

    Args:
      max_extra: if None, loop forever until SAT; else stop after extra > max_extra
      start_extra: start at extra = start_extra (default 0 means try W=w first)
      verbose: print status lines each iteration

    Returns:
      Same tuple as solve_value_assignment: (True, compressed, w, W, values)
      or, if max_extra is set and no solution is found up to it:
        (False, compressed, w, W_last_tried, None)
    """
    # Compute w once (also validates shape).
    compressed, w = compress_binary_matrix(bits)

    # Handle degenerate cases quickly.
    n = len(compressed)
    if n == 0:
        if verbose:
            print("[relax] n=0: vacuously SAT")
        return True, compressed, 0, 0, []
    if w == 0:
        if verbose:
            print("[relax] w=0 (all I): vacuously SAT")
        return True, compressed, 0, 0, [[None for _ in range(n)] for _ in range(n)]

    extra = int(start_extra)
    if extra < 0:
        raise ValueError("start_extra must be >= 0.")
    if max_extra is not None and max_extra < extra:
        raise ValueError("max_extra must be >= start_extra (or None).")

    last_W = None
    while True:
        W = w + extra
        last_W = W
        if verbose:
            print(f"[relax] trying value range 1..{W} (w={w}, extra={extra}) ...", flush=True)

        ok, _compressed2, _w2, W2, values = solve_value_assignment(bits, value_cap=W)

        if ok:
            if verbose:
                print(f"[relax] SAT found with W={W2}")
            return True, compressed, w, W2, values

        if max_extra is not None and extra >= max_extra:
            if verbose:
                print(f"[relax] UNSAT up to W={W} (max_extra={max_extra}). Stopping.")
            return False, compressed, w, last_W, None

        extra += 1


# ------------------------- Example -------------------------
if __name__ == "__main__":
    # Replace this with your input. n=3 => bits is 3 x 6
    code = MirrorCode(
        group = [2, 2, 3, 3],
        z0 = [[0, 0, 0, 0],
       [0, 1, 0, 1],
       [1, 0, 0, 2]],
        x0 = [[0, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 1, 2, 0]]
    )

    bits = code.get_stabilizers()
    print(bits)

    ok, compressed, w, W, values = solve_with_relaxation(bits, max_extra=10, verbose=True)

    print("\nResult:")
    print("SAT:", ok, "w:", w, "W used:", W)
    print("Compressed:")
    for r in compressed:
        print(" ".join(r))
    print("Values (None means I):")
    if ok:
        for r in values:
            print(r)
