# isomorphism.py
#
# Helpers for working with finite abelian p-groups of type
#   G = ⊕_i Z / p^{λ_i} Z
# where lambdas = (λ_1, ..., λ_r).
#
# Public API used by search.py:
#   - lex_minimal_vectors(p, lambdas)
#   - automorphisms_fixing_vectors(p, lambdas, fixed_vectors)
#   - push_to_lex_minimal(p, lambdas, v)
#   - is_single_equivalence_class_under_shifts(Z_wt, X_wt, prime_powers, vectors)
#
# All other functions are internal.

from itertools import product
from functools import lru_cache
import numpy as np


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def factor_prime_power(n: int):
    """Return (p, k) such that n = p**k with p prime."""
    if n <= 1:
        raise ValueError("Group factors must be prime powers ≥ 2.")
    m = n
    p = None
    # find smallest prime divisor
    d = 2
    while d * d <= m:
        if m % d == 0:
            p = d
            break
        d += 1
    if p is None:
        p = m  # n is prime
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    if n != 1:
        raise ValueError(f"{m} is not a prime power")
    return p, k


def _p_adic_valuation(x: int, p: int):
    """v_p(x) for x != 0, with x an integer."""
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


# ----------------------------------------------------------------------
# Endomorphism / automorphism machinery
# ----------------------------------------------------------------------

def endo_entry_options(p: int, lam_i: int, lam_j: int):
    """
    Allowed values for matrix entry a_ij of a group endomorphism:

    - Row i lives in Z / p^{lam_i} Z.
    - For the homomorphism condition, p^{lam_j} * e_j must map to 0 in row i,
      i.e. p^{lam_j} * a_ij ≡ 0 (mod p^{lam_i}).

    That forces:
      - If lam_j < lam_i: a_ij must be divisible by p^{lam_i - lam_j}.
      - If lam_j >= lam_i: a_ij can be any element mod p^{lam_i}.
    """
    n_i = p ** lam_i
    if lam_j < lam_i:
        step = p ** (lam_i - lam_j)
        return list(range(0, n_i, step))
    else:
        return list(range(n_i))


def rank_mod_p(rows, p: int):
    """
    Compute the rank over F_p of a matrix given by its list of rows (mod p).
    rows: list of iterables of equal length.
    """
    if not rows:
        return 0
    M = [list(r) for r in rows]
    m = len(M[0])  # number of columns
    rank = 0

    for col in range(m):
        pivot = None
        for i in range(rank, len(M)):
            if M[i][col] % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        M[rank], M[pivot] = M[pivot], M[rank]
        inv = pow(M[rank][col], -1, p)

        for i in range(rank + 1, len(M)):
            factor = M[i][col] * inv % p
            if factor == 0:
                continue
            for j in range(col, m):
                M[i][j] = (M[i][j] - factor * M[rank][j]) % p

        rank += 1
        if rank == len(M):
            break

    return rank


def automorphisms_fixing_vectors(p: int, lambdas, fixed_vectors):
    """
    Enumerate all automorphisms of G = ⊕ Z/p^{λ_i}Z that fix each vector in fixed_vectors.

    Parameters
    ----------
    p : int
        Prime.
    lambdas : iterable of ints
        Exponents λ_i, so modulus at coordinate i is p**λ_i.
    fixed_vectors : iterable of vectors
        Each vector is an iterable of length r; entries are interpreted mod p**λ_i.

    Returns
    -------
    np.ndarray
        Array of shape (num_aut, r, r). If no such automorphisms exist, returns
        an empty array of shape (0, r, r).
    """
    lambdas = tuple(int(l) for l in lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]

    # Normalize fixed vectors mod the group.
    fixed = []
    for v in fixed_vectors:
        if len(v) != r:
            raise ValueError("Each fixed vector must have length equal to number of factors.")
        fixed.append(tuple(int(v_i) % n_i for v_i, n_i in zip(v, moduli)))

    # Precompute row candidates for each row i
    row_candidates = []
    for i in range(r):
        n_i = moduli[i]
        opts_per_j = [endo_entry_options(p, lambdas[i], lambdas[j]) for j in range(r)]
        candidates_i = []

        for entries in product(*opts_per_j):
            row = list(entries)
            ok = True
            for v in fixed:
                dot = sum(row[j] * v[j] for j in range(r)) % n_i
                if dot != v[i]:
                    ok = False
                    break
            if ok:
                candidates_i.append(tuple(row))

        row_candidates.append(candidates_i)

    autos = []

    def backtrack(i, current_rows, current_rows_mod_p):
        if i == r:
            autos.append(np.array(current_rows, dtype=int))
            return

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            # A zero row mod p cannot be part of an invertible matrix
            if all(c == 0 for c in row_mod_p):
                continue

            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)
            if new_rank != prev_rank + 1:
                continue

            backtrack(i + 1,
                      current_rows + [list(row)],
                      new_rows_mod_p)

    backtrack(0, [], [])

    if autos:
        return np.stack(autos, axis=0)
    else:
        return np.zeros((0, r, r), dtype=int)


def element_order_p_group(p: int, lambdas, v):
    """
    Compute the order of a vector v in the p-group G = ⊕ Z/p^{λ_i}Z.

    Parameters
    ----------
    p : int
    lambdas : iterable of ints
    v : iterable of length r

    Returns
    -------
    int
        p^k, where k is the maximal exponent among the coordinates,
        or 1 if v is the zero vector.
    """
    lambdas = tuple(int(l) for l in lambdas)
    moduli = [p ** lam for lam in lambdas]
    v_norm = [int(v_i) % n_i for v_i, n_i in zip(v, moduli)]

    max_exp = 0
    any_nonzero = False
    for coord, lam in zip(v_norm, lambdas):
        if coord == 0:
            continue
        any_nonzero = True
        val = _p_adic_valuation(coord, p)
        exp = lam - val
        if exp > max_exp:
            max_exp = exp

    if not any_nonzero:
        return 1
    return p ** max_exp


def _automorphism_sending_vector(p: int, lambdas, source, target):
    """
    Try to construct a single automorphism A of G such that A * source = target.

    Parameters
    ----------
    p : int
    lambdas : iterable of ints
    source, target : length-r iterables

    Returns
    -------
    list[list[int]] or None
        A matrix A (list of rows) if possible, or None if no such automorphism exists.
    """
    lambdas = tuple(int(l) for l in lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]

    if len(source) != r or len(target) != r:
        raise ValueError("source and target must have same length as lambdas.")

    src = tuple(int(x) % n for x, n in zip(source, moduli))
    tgt = tuple(int(x) % n for x, n in zip(target, moduli))

    # Easy impossibility checks
    if all(x == 0 for x in src) and any(x != 0 for x in tgt):
        return None
    if all(x == 0 for x in src) and all(x == 0 for x in tgt):
        # Identity works
        return [[int(i == j) for j in range(r)] for i in range(r)]

    # Build row candidates enforcing A*src = tgt row by row
    row_candidates = []
    for i in range(r):
        n_i = moduli[i]
        opts_per_j = [endo_entry_options(p, lambdas[i], lambdas[j]) for j in range(r)]
        cand_i = []
        for entries in product(*opts_per_j):
            row = list(entries)
            dot = sum(row[j] * src[j] for j in range(r)) % n_i
            if dot == tgt[i]:
                cand_i.append(tuple(row))
        row_candidates.append(cand_i)

    def backtrack(i, current_rows, current_rows_mod_p):
        if i == r:
            return [list(row) for row in current_rows]

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)
            if new_rank != prev_rank + 1:
                continue

            res = backtrack(i + 1, current_rows + [row], new_rows_mod_p)
            if res is not None:
                return res

        return None

    return backtrack(0, [], [])


# ----------------------------------------------------------------------
# Canonical representatives under full Aut(G)
# ----------------------------------------------------------------------

def _all_group_elements(p: int, lambdas):
    """Enumerate all elements of G = ⊕ Z/p^{λ_i}Z in lexicographic order."""
    lambdas = tuple(int(l) for l in lambdas)
    moduli = [p ** lam for lam in lambdas]
    ranges = [range(n) for n in moduli]
    for v in product(*ranges):
        yield tuple(int(x) for x in v)


def _lex_minimal_orbit_rep(p: int, lambdas, v, elems, orders):
    """
    For a given v in G, find the lexicographically minimal element w
    in its Aut(G)-orbit, together with an automorphism A s.t. A*v = w.

    Parameters
    ----------
    p : int
    lambdas : tuple[int]
    v : tuple[int]
    elems : list[tuple[int]]
        All group elements in lex order.
    orders : dict[tuple[int], int]
        Precomputed element orders.

    Returns
    -------
    (w, A)
      w : tuple[int]  (lex-minimal orbit representative)
      A : list[list[int]]  (automorphism matrix with A*v = w)
    """
    lambdas = tuple(int(l) for l in lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]
    v_norm = tuple(int(v_i) % n_i for v_i, n_i in zip(v, moduli))

    if all(x == 0 for x in v_norm):
        A_id = [[int(i == j) for j in range(r)] for i in range(r)]
        return v_norm, A_id

    ord_v = orders[v_norm]

    for w in elems:
        w_norm = w
        if all(x == 0 for x in w_norm):
            # nonzero element cannot map to zero
            continue
        if orders[w_norm] != ord_v:
            continue
        A = _automorphism_sending_vector(p, lambdas, v_norm, w_norm)
        if A is not None:
            return w_norm, A

    raise RuntimeError("No automorphism found mapping element to any candidate.")


@lru_cache(maxsize=None)
def _lex_minimal_vectors_cached(p: int, lambdas_tuple):
    """
    Cached core of lex_minimal_vectors: returns a tuple of tuples, each
    a lex-minimal Aut(G)-orbit representative for G = ⊕ Z/p^{λ_i}Z.
    """
    lambdas_tuple = tuple(int(l) for l in lambdas_tuple)
    elems = list(_all_group_elements(p, lambdas_tuple))

    # Precompute orders for all elements
    orders = {v: element_order_p_group(p, lambdas_tuple, v) for v in elems}

    reps_set = set()
    for v in elems:
        w, _ = _lex_minimal_orbit_rep(p, lambdas_tuple, v, elems, orders)
        reps_set.add(w)

    reps_list = sorted(reps_set)
    return tuple(reps_list)


def lex_minimal_vectors(p: int, lambdas):
    """
    Return all lexicographically minimal vectors for Aut(G)-orbits in
    G = ⊕ Z/p^{λ_i}Z, where lambdas = (λ_1, ..., λ_r).

    Each vector is returned as a list[int] of length r, and we include
    exactly one representative per orbit (including the zero vector).
    """
    reps = _lex_minimal_vectors_cached(p, tuple(lambdas))
    # Convert tuples to lists (to match previous API usage in search.py)
    return [list(v) for v in reps]


def push_to_lex_minimal(p: int, lambdas, v):
    """
    Given v in G = ⊕ Z/p^{λ_i}Z, find an automorphism A such that A * v
    is the lexicographically minimal representative in its Aut(G)-orbit.

    Returns
    -------
    A : list[list[int]]
        r x r automorphism matrix with A*v = w_min.
    """
    lambdas = tuple(int(l) for l in lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]
    v_norm = tuple(int(v_i) % n_i for v_i, n_i in zip(v, moduli))

    if all(x == 0 for x in v_norm):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    ord_v = element_order_p_group(p, lambdas, v_norm)
    candidates = _lex_minimal_vectors_cached(p, lambdas)

    for w in candidates:
        if element_order_p_group(p, lambdas, w) != ord_v:
            continue
        A = _automorphism_sending_vector(p, lambdas, v_norm, w)
        if A is not None:
            return A

    raise RuntimeError("No automorphism found mapping v to a lex-minimal representative.")


# ----------------------------------------------------------------------
# Equivalence relation under shifts (unchanged logically)
# ----------------------------------------------------------------------

def is_single_equivalence_class_under_shifts(Z_wt, X_wt, prime_powers, vectors):
    """
    Given:
        Z_wt, X_wt : nonnegative integers with len(vectors) == Z_wt + X_wt
        prime_powers: [n1, ..., nr], each ni a prime power (primes may differ)
        vectors     : list of group elements, each an iterable of length r

    We consider the group G = ⊕_i Z/ni Z and an equivalence relation generated by:

      For each v ∈ G, form
          S(v) = { vectors[i] + v   for i in range(Z_wt) }
               ∪ { vectors[i] - v   for i in range(Z_wt, Z_wt + X_wt) }

      (operations done component-wise modulo ni).
      All elements of S(v) are declared equivalent.

    The function returns True iff this equivalence relation has exactly ONE
    equivalence class (i.e. the whole group is identified), and False otherwise.
    """
    r = len(prime_powers)
    for n in prime_powers:
        factor_prime_power(n)  # will raise if not a prime power

    if len(vectors) != Z_wt + X_wt:
        raise ValueError("len(vectors) must be exactly Z_wt + X_wt.")

    # Normalize vectors mod each modulus
    base_vectors = []
    for v in vectors:
        if len(v) != r:
            raise ValueError("Each vector must have length equal to the number of group factors.")
        base_vectors.append(
            tuple(int(v_i) % n_i for v_i, n_i in zip(v, prime_powers))
        )

    # Enumerate all group elements
    elems = list(product(*(range(n_i) for n_i in prime_powers)))
    index_of = {e: i for i, e in enumerate(elems)}
    N = len(elems)

    # Union–find
    class DSU:
        __slots__ = ("parent", "rank")

        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, a, b):
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1

    dsu = DSU(N)

    # Generate equivalence from all v in G
    for v in elems:
        S_v = []

        # First Z_wt entries: vectors[i] + v
        for i in range(Z_wt):
            g = base_vectors[i]
            S_v.append(
                tuple((g[j] + v[j]) % prime_powers[j] for j in range(r))
            )

        # Last X_wt entries: vectors[i] - v
        for i in range(Z_wt, Z_wt + X_wt):
            g = base_vectors[i]
            S_v.append(
                tuple((g[j] - v[j]) % prime_powers[j] for j in range(r))
            )

        if not S_v:
            continue

        base_idx = index_of[S_v[0]]
        for w in S_v[1:]:
            dsu.union(base_idx, index_of[w])

    # Check if all group elements lie in one equivalence class
    root0 = dsu.find(0)
    for i in range(1, N):
        if dsu.find(i) != root0:
            return False
    return True
