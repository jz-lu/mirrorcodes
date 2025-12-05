from itertools import product, combinations
from functools import lru_cache
import numpy as np


# ---------- Basic helpers ----------

def factor_prime_power(n: int):
    """Return (p, k) such that n = p**k with p prime."""
    if n <= 1:
        raise ValueError("Group factors must be prime powers ≥ 2.")
    for p in range(2, n + 1):
        if n % p == 0:
            k = 0
            m = n
            while m % p == 0:
                m //= p
                k += 1
            if m == 1:
                return p, k
    raise ValueError(f"{n} is not a prime power")


# ---------- 1) Lexicographically minimal vectors for a p-group ----------

def lex_minimal_vectors(p, lambdas):
    """
    Given a finite abelian p-group described by

        G = ⊕_i Z / p^{λ_i} Z

    with λ_i in `lambdas` (non-decreasing), return all lexicographically minimal
    vectors (one per Aut(G)-orbit) as tuples (x_1, ..., x_r) with
    0 <= x_i < p^{λ_i}.
    """
    lambdas_t = tuple(int(l) for l in lambdas)
    return list(_lex_minimal_vectors_cached(p, lambdas_t))


@lru_cache(maxsize=None)
def _lex_minimal_vectors_cached(p, lambdas):
    """Cached core for lex_minimal_vectors."""
    lambdas = tuple(lambdas)
    r = len(lambdas)
    vectors = {tuple(0 for _ in range(r))}
    indices = list(range(r))

    for t in range(1, r + 1):
        for idxs in combinations(indices, t):

            def rec(pos, prev_alpha, prev_beta, alphas):
                if pos == t:
                    coords = [0] * r
                    for j, idx in enumerate(idxs):
                        alpha = alphas[j]
                        coords[idx] = p ** alpha
                    vectors.add(tuple(coords))
                    return

                i = idxs[pos]
                lam = lambdas[i]
                start_alpha = 0 if prev_alpha is None else prev_alpha + 1

                for alpha in range(start_alpha, lam):
                    beta = lam - alpha
                    if prev_beta is not None and beta <= prev_beta:
                        continue
                    rec(pos + 1, alpha, beta, alphas + [alpha])

            rec(0, None, None, [])

    return tuple(sorted(vectors))


# ---------- 2) Endomorphism entries and rank mod p ----------

def endo_entry_options(p, lam_i, lam_j):
    """
    Allowed values for matrix entry a_ij of a group endomorphism of
    ⊕_k Z/p^{λ_k}Z.
    """
    n_i = p ** lam_i
    if lam_j < lam_i:
        step = p ** (lam_i - lam_j)
        return list(range(0, n_i, step))
    else:
        return list(range(n_i))


def rank_mod_p(rows, p):
    """
    Compute the rank over F_p of a matrix given by its list of rows.
    rows: list of iterables of equal length, entries interpreted mod p.
    """
    if not rows:
        return 0

    m = len(rows[0])
    M = [list(r) for r in rows]
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


# ---------- 3) Automorphisms fixing given vectors (with shifts) ----------

def automorphisms_fixing_vectors(p, lambdas, Z_wt, fixed_vectors):
    """
    Public wrapper that normalizes arguments then delegates to the cached core.
    """
    lambdas_t = tuple(int(l) for l in lambdas)
    fixed_t = tuple(tuple(int(x) for x in v) for v in fixed_vectors)
    return _automorphisms_fixing_vectors_cached(p, lambdas_t, int(Z_wt), fixed_t)


@lru_cache(maxsize=None)
def _automorphisms_fixing_vectors_cached(p, lambdas, Z_wt, fixed):
    """
    Cached core for automorphisms_fixing_vectors.

    Additional rule (for p == 2): only keep automorphisms whose shift vector
    has *even* entries in every component (mod the corresponding modulus).
    """
    lambdas = tuple(lambdas)
    r = len(lambdas)
    if r == 0:
        return (np.empty((0, 0, 0), dtype=int),
                np.empty((0, 0), dtype=int))

    moduli = np.array([p ** lam for lam in lambdas], dtype=int)

    fixed = list(fixed)
    t = len(fixed)

    # Normalize fixed vectors
    fixed_norm = [
        tuple(int(v[j]) % moduli[j] for j in range(r))
        for v in fixed
    ]

    baseX = None
    diffs = []
    if t > Z_wt:
        baseX = fixed_norm[Z_wt]
        for k in range(Z_wt + 1, t):
            dv = tuple(
                (fixed_norm[k][j] - baseX[j]) % moduli[j]
                for j in range(r)
            )
            diffs.append(dv)

    row_candidates = []
    indices_Z = range(min(t, Z_wt))

    for i in range(r):
        n_i = int(moduli[i])
        opts_per_j = [
            endo_entry_options(p, lambdas[i], lambdas[j]) for j in range(r)
        ]
        candidates_i = []

        for entries in product(*opts_per_j):
            row = list(entries)
            ok = True

            # Z-row constraints
            for k in indices_Z:
                v = fixed_norm[k]
                dot = sum(row[j] * v[j] for j in range(r)) % n_i
                if dot != v[i]:
                    ok = False
                    break
            if not ok:
                continue

            # X-row differences
            if baseX is not None:
                for dv in diffs:
                    dot = sum(row[j] * dv[j] for j in range(r)) % n_i
                    if dot != dv[i]:
                        ok = False
                        break
                if not ok:
                    continue

            candidates_i.append(tuple(row))

        row_candidates.append(candidates_i)

    automorphisms = []

    def backtrack(i, current_rows, current_rows_mod_p):
        if i == r:
            automorphisms.append([list(row) for row in current_rows])
            return

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            if all(c == 0 for c in row_mod_p):
                continue

            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)
            if new_rank != prev_rank + 1:
                continue

            backtrack(i + 1, current_rows + [row], new_rows_mod_p)

    backtrack(0, [], [])

    if not automorphisms:
        return (np.empty((0, r, r), dtype=int),
                np.empty((0, r), dtype=int))

    mats = np.array(automorphisms, dtype=int)

    # Compute shifts if there are X-rows
    if baseX is not None:
        baseX_vec = np.array(baseX, dtype=int)
        images = np.einsum("aij,j->ai", mats, baseX_vec) % moduli
        shifts = (images - baseX_vec) % moduli

        # Enforce: when p == 2, all shift components must be even
        if p == 2:
            even_mask = (shifts % 2 == 0).all(axis=1)
            mats = mats[even_mask]
            shifts = shifts[even_mask]

            if mats.shape[0] == 0:
                return (np.empty((0, r, r), dtype=int),
                        np.empty((0, r), dtype=int))
    else:
        shifts = np.zeros((mats.shape[0], r), dtype=int)

    return mats, shifts


# ---------- 4) Element order and automorphism sending a vector ----------

def _p_adic_valuation(x, p):
    """
    v_p(x) for x != 0, with x an integer.
    """
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def element_order_p_group(p, lambdas, v):
    """
    Public wrapper with caching for element order.
    """
    lambdas_t = tuple(int(l) for l in lambdas)
    v_t = tuple(int(x) for x in v)
    return _element_order_p_group_cached(p, lambdas_t, v_t)


@lru_cache(maxsize=None)
def _element_order_p_group_cached(p, lambdas, v):
    lambdas = tuple(lambdas)
    r = len(lambdas)
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


def _automorphism_sending_vector(p, lambdas, source, target):
    """
    Internal helper:
    Try to construct a single automorphism A of the group (p, lambdas)
    such that A * source = target.
    """
    lambdas = tuple(lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]

    if len(source) != r or len(target) != r:
        raise ValueError("source and target must have same length as lambdas.")

    src = tuple(int(source[i]) % moduli[i] for i in range(r))
    tgt = tuple(int(target[i]) % moduli[i] for i in range(r))

    if all(x == 0 for x in src) and any(x != 0 for x in tgt):
        return None

    if all(x == 0 for x in src) and all(x == 0 for x in tgt):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    row_candidates = []
    for i in range(r):
        n_i = moduli[i]
        opts_per_j = [endo_entry_options(p, lambdas[i], lambdas[j]) for j in range(r)]
        candidates_i = []
        for entries in product(*opts_per_j):
            row = list(entries)
            dot = sum(row[j] * src[j] for j in range(r)) % n_i
            if dot == tgt[i]:
                candidates_i.append(tuple(row))
        row_candidates.append(candidates_i)

    def backtrack(i, current_rows, current_rows_mod_p):
        if i == r:
            return [list(row) for row in current_rows]

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            if all(c == 0 for c in row_mod_p):
                continue
            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)
            if new_rank != prev_rank + 1:
                continue

            result = backtrack(i + 1, current_rows + [row], new_rows_mod_p)
            if result is not None:
                return result

        return None

    return backtrack(0, [], [])


# ---------- 5) Main function: push to lex-minimal ----------

def push_to_lex_minimal(p, lambdas, v):
    """
    Public wrapper with caching.
    """
    lambdas_t = tuple(int(l) for l in lambdas)
    v_t = tuple(int(x) for x in v)
    return _push_to_lex_minimal_cached(p, lambdas_t, v_t)


@lru_cache(maxsize=None)
def _push_to_lex_minimal_cached(p, lambdas, v):
    """
    Cached core for push_to_lex_minimal.
    """
    lambdas = tuple(lambdas)
    r = len(lambdas)
    moduli = [p ** lam for lam in lambdas]

    if len(v) != r:
        raise ValueError("Vector length must match number of factors.")

    v_norm = tuple(int(v_i) % n_i for v_i, n_i in zip(v, moduli))

    if all(x == 0 for x in v_norm):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    ord_v = _element_order_p_group_cached(p, lambdas, v_norm)

    candidates = _lex_minimal_vectors_cached(p, lambdas)

    for w in candidates:
        if _element_order_p_group_cached(p, lambdas, w) != ord_v:
            continue

        A = _automorphism_sending_vector(p, lambdas, v_norm, w)
        if A is not None:
            return A

    raise RuntimeError("No automorphism found mapping v to a lex-minimal representative.")


# ---------- 6) Equivalence under shifts on a general abelian group ----------

def is_single_equivalence_class_under_shifts(Z_wt, X_wt, prime_powers, vectors):
    """
    Given:
        Z_wt, X_wt : nonnegative integers with len(vectors) == Z_wt + X_wt
        prime_powers: [n1, ..., nr], each ni a prime power (primes may differ)
        vectors     : list of group elements, each an iterable of length r

    We consider the group G = oplus_i Z/ni Z and an equivalence relation generated by:

      For each v ∈ G, form
          S(v) = { vectors[i] + v   for i in range(Z_wt) }
               U { vectors[i] - v   for i in range(Z_wt, Z_wt + X_wt) }

      (operations done component-wise modulo ni).
      All elements of S(v) are declared equivalent.

    The function returns True iff this equivalence relation has exactly ONE
    equivalence class (i.e. the whole group is identified), and False otherwise.

    NOTE: we now assume each ni is a prime power and *do not* re-check it,
    to avoid repeated factorization overhead.
    """
    r = len(prime_powers)

    if len(vectors) != Z_wt + X_wt:
        raise ValueError("len(vectors) must be exactly Z_wt + X_wt.")

    base_vectors = []
    for v in vectors:
        if len(v) != r:
            raise ValueError("Each vector must have length equal to the number of group factors.")
        base_vectors.append(
            tuple(int(v_i) % n_i for v_i, n_i in zip(v, prime_powers))
        )

    from itertools import product

    elems = list(product(*(range(n_i) for n_i in prime_powers)))
    index_of = {e: i for i, e in enumerate(elems)}
    N = len(elems)

    class DSU:
        __slots__ = ("parent", "rank")

        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            parent = self.parent
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(self, a, b):
            parent = self.parent
            rank = self.rank
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

    dsu = DSU(N)

    for v in elems:
        S_v = []

        for i in range(Z_wt):
            g = base_vectors[i]
            S_v.append(
                tuple((g[j] + v[j]) % prime_powers[j] for j in range(r))
            )

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

    root0 = dsu.find(0)
    for i in range(1, N):
        if dsu.find(i) != root0:
            return False
    return True
