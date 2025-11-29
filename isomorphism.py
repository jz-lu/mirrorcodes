from itertools import product, combinations
from functools import lru_cache
import numpy as np


# ============================================================
# Internal helpers
# ============================================================

@lru_cache(maxsize=None)
def _group_prime_powers(p: int, lambdas: tuple[int, ...]):
    """
    For a fixed prime p and lambdas = (λ1, ..., λr),
    return (p_pows, moduli) where:

        p_pows[k] = p**k for 0 <= k <= max(lambdas)
        moduli[i] = p**λ_i
    """
    if not lambdas:
        return (1,), ()
    max_lambda = max(lambdas)
    p_pows = [1]
    for _ in range(max_lambda):
        p_pows.append(p_pows[-1] * p)
    p_pows = tuple(p_pows)
    moduli = tuple(p_pows[lam] for lam in lambdas)
    return p_pows, moduli


def _extend_basis_mod_p(basis, row_mod_p, inv_mod_p, p: int):
    """
    Incrementally maintain an F_p row-echelon basis.

    basis:
        tuple of rows (each a tuple of ints), already reduced with
        leading 1 pivots.

    row_mod_p:
        candidate row (tuple of ints), entries already reduced mod p.

    Returns:
        - new_basis (basis with reduced row appended) if row is independent;
        - None if row is dependent.
    """
    r = list(row_mod_p)
    m = len(r)

    # Reduce r against current basis
    for b in basis:
        for j in range(m):
            bj = b[j]
            if bj:
                rj = r[j]
                if rj:
                    # subtract rj * b, pivot of b is 1
                    factor = rj
                    for k in range(j, m):
                        r[k] = (r[k] - factor * b[k]) % p
                break

    # Find first non-zero entry
    for j in range(m):
        val = r[j] % p
        if val:
            inv = inv_mod_p[val]
            for k in range(j, m):
                r[k] = (r[k] * inv) % p
            return basis + (tuple(r),)

    return None


def _p_adic_valuation(x: int, p: int) -> int:
    """v_p(x) for x != 0."""
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


# ============================================================
# 1) Lexicographically minimal vectors (p-group)
# ============================================================

@lru_cache(maxsize=None)
def lex_minimal_vectors(p: int, lambdas: tuple[int, ...]):
    """
    Given a finite abelian p-group specified by

        p, lambdas = p, (λ1, λ2, ..., λr)  with λ1 <= ... <= λr,

    return all lexicographically minimal vectors (one per Aut(G)-orbit)
    as a tuple of tuples. Each vector x = (x_1, ..., x_r) satisfies
        0 <= x_i < p**λ_i.
    """
    r = len(lambdas)
    p_pows, _ = _group_prime_powers(p, lambdas)

    vectors = {tuple(0 for _ in range(r))}
    indices = range(r)

    for t in range(1, r + 1):
        for idxs in combinations(indices, t):

            def rec(pos, prev_alpha, prev_beta, alphas):
                if pos == t:
                    coords = [0] * r
                    for j, idx in enumerate(idxs):
                        alpha = alphas[j]
                        coords[idx] = p_pows[alpha]
                    vectors.add(tuple(coords))
                    return

                i = idxs[pos]
                lam = lambdas[i]
                start_alpha = 0 if prev_alpha is None else prev_alpha + 1

                for alpha in range(start_alpha, lam):
                    beta = lam - alpha
                    if prev_beta is not None and beta <= prev_beta:
                        continue
                    rec(pos + 1, alpha, beta, alphas + (alpha,))

            rec(0, None, None, ())

    return tuple(sorted(vectors))


# ============================================================
# 2) Automorphisms fixing given vectors (p-group)
# ============================================================

def automorphisms_fixing_vectors(
    p: int,
    lambdas: tuple[int, ...],
    fixed_vectors: tuple[tuple[int, ...], ...],
):
    """
    Enumerate all automorphisms of the abelian p-group G specified by
        p, lambdas = p, (λ1, ..., λr)
    that fix each vector in `fixed_vectors`.

    Input:
        p            : prime
        lambdas      : tuple of exponents (λ1, ..., λr)
        fixed_vectors: tuple of vectors, each a tuple of length r

    Output:
        numpy array of shape (k, r, r), each slice an automorphism matrix.
    """
    r = len(lambdas)
    lambdas_t = lambdas
    p_pows, moduli = _group_prime_powers(p, lambdas_t)

    # Normalise fixed vectors modulo the group
    fixed = []
    for v in fixed_vectors:
        fixed.append(tuple(v_i % n_i for v_i, n_i in zip(v, moduli)))
    fixed = tuple(fixed)

    # Precompute inverses in F_p
    inv_mod_p = {a: pow(a, -1, p) for a in range(1, p)}

    row_candidates = []

    for i in range(r):
        n_i = moduli[i]
        lam_i = lambdas_t[i]

        # Allowed residues per column j
        opts_per_j = []
        for j in range(r):
            lam_j = lambdas_t[j]
            if lam_j < lam_i:
                step = p_pows[lam_i - lam_j]
                opts_per_j.append(range(0, n_i, step))
            else:
                opts_per_j.append(range(n_i))

        candidates_i = []
        if fixed:
            for entries in product(*opts_per_j):
                ok = True
                for v in fixed:
                    s = 0
                    for a, b in zip(entries, v):
                        s += a * b
                    if s % n_i != v[i]:
                        ok = False
                        break
                if ok:
                    candidates_i.append(tuple(entries))
        else:
            for entries in product(*opts_per_j):
                candidates_i.append(tuple(entries))

        row_candidates.append(tuple(candidates_i))

    automorphisms = []

    def backtrack(i, current_rows, basis):
        if i == r:
            automorphisms.append([list(row) for row in current_rows])
            return

        candidates_i = row_candidates[i]
        for row in candidates_i:
            row_mod_p = tuple(c % p for c in row)
            if not any(row_mod_p):
                continue
            new_basis = _extend_basis_mod_p(basis, row_mod_p, inv_mod_p, p)
            if new_basis is None:
                continue
            backtrack(i + 1, current_rows + (row,), new_basis)

    backtrack(0, (), ())
    if not automorphisms:
        return np.zeros((0, r, r), dtype=int)
    return np.array(automorphisms, dtype=int)


# ============================================================
# 3) Element order in a p-group
# ============================================================

def element_order_p_group(
    p: int,
    lambdas: tuple[int, ...],
    v: tuple[int, ...],
) -> int:
    """
    Compute the order of v in the p-group G determined by (p, lambdas).
    Returns p^k or 1 for zero vector.
    """
    lambdas_t = lambdas
    _, moduli = _group_prime_powers(p, lambdas_t)
    v_norm = [v_i % n_i for v_i, n_i in zip(v, moduli)]

    max_exp = 0
    any_nonzero = False
    for coord, lam in zip(v_norm, lambdas_t):
        if coord == 0:
            continue
        any_nonzero = True
        val = _p_adic_valuation(coord, p)
        exp = lam - val
        if exp > max_exp:
            max_exp = exp

    if not any_nonzero:
        return 1

    p_pows, _ = _group_prime_powers(p, lambdas_t)
    return p_pows[max_exp]


# ============================================================
# 4) Internal: construct automorphism sending one vector to another
# ============================================================

@lru_cache(maxsize=None)
def _automorphism_sending_vector(
    p: int,
    lambdas: tuple[int, ...],
    source: tuple[int, ...],
    target: tuple[int, ...],
):
    """
    Try to construct an automorphism A of the p-group (p, lambdas)
    with A * source = target.

    Returns:
        - matrix A (list of rows) if possible
        - None if no such automorphism exists
    """
    r = len(lambdas)
    lambdas_t = lambdas
    p_pows, moduli = _group_prime_powers(p, lambdas_t)

    src = tuple(source[i] % moduli[i] for i in range(r))
    tgt = tuple(target[i] % moduli[i] for i in range(r))

    # Nonzero -> zero impossible under automorphism
    if all(x == 0 for x in src) and any(x != 0 for x in tgt):
        return None

    # Zero -> zero: identity
    if all(x == 0 for x in src) and all(x == 0 for x in tgt):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    inv_mod_p = {a: pow(a, -1, p) for a in range(1, p)}

    row_candidates = []

    for i in range(r):
        n_i = moduli[i]
        lam_i = lambdas_t[i]

        opts_per_j = []
        for j in range(r):
            lam_j = lambdas_t[j]
            if lam_j < lam_i:
                step = p_pows[lam_i - lam_j]
                opts_per_j.append(range(0, n_i, step))
            else:
                opts_per_j.append(range(n_i))

        candidates_i = []
        for entries in product(*opts_per_j):
            s = 0
            for a, b in zip(entries, src):
                s += a * b
            if s % n_i == tgt[i]:
                candidates_i.append(tuple(entries))

        row_candidates.append(tuple(candidates_i))

    def backtrack(i, current_rows, basis):
        if i == r:
            return [list(row) for row in current_rows]

        candidates_i = row_candidates[i]
        for row in candidates_i:
            row_mod_p = tuple(c % p for c in row)
            if not any(row_mod_p):
                continue
            new_basis = _extend_basis_mod_p(basis, row_mod_p, inv_mod_p, p)
            if new_basis is None:
                continue
            result = backtrack(i + 1, current_rows + (row,), new_basis)
            if result is not None:
                return result
        return None

    return backtrack(0, (), ())


# ============================================================
# 5) Main: push to lex-minimal representative
# ============================================================

@lru_cache(maxsize=None)
def push_to_lex_minimal(
    p: int,
    lambdas: tuple[int, ...],
    v: tuple[int, ...],
):
    """
    Given a p-group (p, lambdas) and a vector v, find an automorphism A
    such that A*v is the lexicographically minimal element in the Aut(G)-orbit
    of v. Returns A as list-of-lists (rows).
    """
    r = len(lambdas)
    lambdas_t = lambdas
    _, moduli = _group_prime_powers(p, lambdas_t)

    v_norm = tuple(v_i % n_i for v_i, n_i in zip(v, moduli))

    # Zero is already lex-minimal
    if all(x == 0 for x in v_norm):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    ord_v = element_order_p_group(p, lambdas_t, v_norm)
    candidates = lex_minimal_vectors(p, lambdas_t)

    for w in candidates:
        if element_order_p_group(p, lambdas_t, w) != ord_v:
            continue
        A = _automorphism_sending_vector(p, lambdas_t, v_norm, w)
        if A is not None:
            return A

    raise RuntimeError("No automorphism found mapping v to a lex-minimal representative.")


# ============================================================
# 6) rank_mod_p helper (optional)
# ============================================================

def rank_mod_p(rows, p: int) -> int:
    """
    Compute the rank over F_p of a matrix given by its list of rows.
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


# ============================================================
# 7) Equivalence under shifts in a finite abelian group
# ============================================================

def is_single_equivalence_class_under_shifts(
    Z_wt: int,
    X_wt: int,
    prime_powers,
    vectors,
) -> bool:
    """
    Same semantics as in your original code, but assumes:
        - each modulus is a prime power,
        - product(prime_powers) <= 256,
        - len(vectors) == Z_wt + X_wt.

    No safety checks for speed.
    """
    r = len(prime_powers)

    base_vectors = [
        tuple(v_i % n_i for v_i, n_i in zip(v, prime_powers))
        for v in vectors
    ]

    from itertools import product as _prod

    elems = list(_prod(*(range(n_i) for n_i in prime_powers)))
    index_of = {e: i for i, e in enumerate(elems)}
    N = len(elems)

    parent = list(range(N))
    rank = [0] * N

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

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
            union(base_idx, index_of[w])

    root0 = find(0)
    for i in range(1, N):
        if find(i) != root0:
            return False
    return True
