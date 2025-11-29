from itertools import product, combinations
from functools import lru_cache
import numpy as np


# ============================================================
# Internal helpers and caches
# ============================================================

@lru_cache(maxsize=None)
def _group_prime_powers(p: int, lambdas_tuple: tuple[int, ...]):
    """
    Cache p^k and the moduli p**lambda_i for a given (p, lambdas).
    Returns:
        (p_pows, moduli)
        p_pows[k] = p**k for 0 <= k <= max(lambdas)
        moduli[i] = p**lambdas[i]
    """
    if not lambdas_tuple:
        return (1,), ()

    max_lambda = max(lambdas_tuple)
    p_pows = [1]
    for _ in range(max_lambda):
        p_pows.append(p_pows[-1] * p)
    p_pows = tuple(p_pows)

    moduli = tuple(p_pows[lam] for lam in lambdas_tuple)
    return p_pows, moduli


def _extend_basis_mod_p(
    basis: tuple[tuple[int, ...], ...],
    row_mod_p: tuple[int, ...],
    inv_mod_p: dict[int, int],
    p: int,
) -> tuple[tuple[int, ...], ...] | None:
    """
    Maintain an incremental row-echelon basis over F_p.

    basis:
        tuple of rows, each already in reduced form with a leading 1 pivot
        (not necessarily in strictly increasing pivot columns, but that's OK).

    row_mod_p:
        the candidate row (entries already reduced modulo p).

    Returns:
        - new_basis (basis with the reduced row appended) if row is independent;
        - None if row is dependent (i.e. reduces to 0).
    """
    r = list(row_mod_p)
    m = len(r)

    # Reduce r against current basis
    for b in basis:
        # pivot column: first non-zero entry of b
        for j in range(m):
            bj = b[j]
            if bj:  # pivot column
                rj = r[j]
                if rj:
                    # pivot of b is 1, so subtract rj * b
                    factor = rj
                    for k in range(j, m):
                        r[k] = (r[k] - factor * b[k]) % p
                break

    # Check if r is now the zero vector
    for j in range(m):
        if r[j] % p:
            # Non-zero: normalise pivot to 1
            pivot_val = r[j] % p
            inv = inv_mod_p[pivot_val]
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
#    API: (p, lambdas)
# ============================================================

def lex_minimal_vectors(p: int, lambdas: list[int]) -> list[list[int]]:
    """
    Given a finite abelian p-group specified by

        p, lambdas = p, [λ1, λ2, ..., λr]  with λ1 <= ... <= λr,

    return all lexicographically minimal vectors (one per Aut(G)-orbit) as
    described in your previous discussion.

    Each returned vector x = (x_1, ..., x_r) has
        0 <= x_i < p**λ_i,
    and lex order is the usual Python tuple order.
    """
    lambdas = list(lambdas)
    r = len(lambdas)

    # p_pows[k] = p**k, shared for this group
    p_pows, _ = _group_prime_powers(p, tuple(lambdas))

    # Always include the zero vector
    vectors = {tuple(0 for _ in range(r))}
    indices = range(r)

    # Characterisation:
    # Choose indices i_1 < ... < i_t and exponents α_j with
    #   0 <= α_1 < ... < α_t < λ_{i_t}
    # and
    #   λ_{i_1} - α_1 < ... < λ_{i_t} - α_t
    # Then x_{i_j} = p**α_j, others 0.
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
                    rec(pos + 1, alpha, beta, alphas + [alpha])

            rec(0, None, None, [])

    return [list(v) for v in sorted(vectors)]


# ============================================================
# 2) Automorphisms fixing given vectors (p-group)
#    API: (p, lambdas)
# ============================================================

def automorphisms_fixing_vectors(
    p: int,
    lambdas: list[int],
    fixed_vectors: list[list[int]] | list[tuple[int, ...]],
) -> np.ndarray:
    """
    Enumerate all automorphisms of the abelian p-group G specified by
        p, lambdas = p, [λ1, ..., λr]

    that fix each vector in `fixed_vectors`.

    Input:
        p           : prime
        lambdas     : list of exponents [λ1, ..., λr] with λ1 <= ... <= λr
        fixed_vectors:
            list of vectors of length r.
            Entry v[i] is taken modulo p**λ_i internally.

    Output:
        A numpy array of shape (k, r, r). Each slice A[t] is a matrix
        representing an automorphism of G that fixes all fixed_vectors.
    """
    lambdas = list(lambdas)
    r = len(lambdas)
    lambdas_t = tuple(lambdas)
    p_pows, moduli = _group_prime_powers(p, lambdas_t)

    # Normalise fixed vectors modulo the group
    fixed = [
        tuple(v_i % n_i for v_i, n_i in zip(v, moduli))
        for v in fixed_vectors
    ]

    # Precompute inverses in F_p (p is prime)
    inv_mod_p = {a: pow(a, -1, p) for a in range(1, p)}

    # Precompute candidate rows for each row i, subject to:
    #   - endomorphism constraints
    #   - A(v) = v in coordinate i for all fixed v
    row_candidates: list[list[tuple[int, ...]]] = []

    for i in range(r):
        n_i = moduli[i]
        lam_i = lambdas[i]

        # allowed residues in column j
        opts_per_j = []
        for j in range(r):
            lam_j = lambdas[j]
            if lam_j < lam_i:
                # entry must be multiple of p^{lam_i - lam_j}
                step = p_pows[lam_i - lam_j]
                opts_per_j.append(range(0, n_i, step))
            else:
                # any residue mod p^{lam_i}
                opts_per_j.append(range(n_i))

        candidates_i: list[tuple[int, ...]] = []

        if fixed:
            for entries in product(*opts_per_j):
                ok = True
                # enforce (row · v) ≡ v[i] (mod n_i) for each fixed v
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
            # No fixed vectors: every endomorphism row is allowed
            for entries in product(*opts_per_j):
                candidates_i.append(tuple(entries))

        row_candidates.append(candidates_i)

    automorphisms: list[list[list[int]]] = []

    def backtrack(i: int,
                  current_rows: list[tuple[int, ...]],
                  basis: tuple[tuple[int, ...], ...]):
        if i == r:
            automorphisms.append([list(row) for row in current_rows])
            return

        candidates_i = row_candidates[i]
        for row in candidates_i:
            row_mod_p = tuple(c % p for c in row)
            if not any(row_mod_p):
                # Zero row not allowed in an invertible matrix
                continue

            new_basis = _extend_basis_mod_p(basis, row_mod_p, inv_mod_p, p)
            if new_basis is None:
                continue

            backtrack(i + 1,
                      current_rows + [row],
                      new_basis)

    backtrack(0, [], ())
    return np.array(automorphisms, dtype=int)


# ============================================================
# 3) Element order in a p-group
#    API: (p, lambdas)
# ============================================================

def element_order_p_group(
    p: int,
    lambdas: list[int],
    v: list[int] | tuple[int, ...],
) -> int:
    """
    Compute the order of a vector v in the p-group determined by (p, lambdas).

    Input:
        p       : prime
        lambdas : [λ1, ..., λr]
        v       : iterable of length r

    Returns:
        p^k, where k is the maximal exponent among the coordinates,
        or 1 if v is the zero vector.
    """
    lambdas = list(lambdas)
    p_pows, moduli = _group_prime_powers(p, tuple(lambdas))

    v_norm = [v_i % n_i for v_i, n_i in zip(v, moduli)]

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

    return p_pows[max_exp]


# ============================================================
# 4) Internal: construct automorphism sending one vector to another
#    API: (p, lambdas)
# ============================================================

def _automorphism_sending_vector(
    p: int,
    lambdas: list[int],
    source: list[int] | tuple[int, ...],
    target: list[int] | tuple[int, ...],
) -> list[list[int]] | None:
    """
    Try to construct an automorphism A of the p-group (p, lambdas) with A*source = target.

    Returns:
        - matrix A (list of rows) if possible
        - None if no such automorphism exists
    """
    lambdas = list(lambdas)
    r = len(lambdas)
    lambdas_t = tuple(lambdas)
    p_pows, moduli = _group_prime_powers(p, lambdas_t)

    src = tuple(source[i] % moduli[i] for i in range(r))
    tgt = tuple(target[i] % moduli[i] for i in range(r))

    # Nonzero -> zero is impossible under an automorphism
    if all(x == 0 for x in src) and any(x != 0 for x in tgt):
        return None

    # Zero -> zero: identity automorphism
    if all(x == 0 for x in src) and all(x == 0 for x in tgt):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    # Precompute inverses in F_p
    inv_mod_p = {a: pow(a, -1, p) for a in range(1, p)}

    # Build candidate rows with A*src = tgt, row by row
    row_candidates: list[list[tuple[int, ...]]] = []

    for i in range(r):
        n_i = moduli[i]
        lam_i = lambdas[i]

        # allowed residues per column j
        opts_per_j = []
        for j in range(r):
            lam_j = lambdas[j]
            if lam_j < lam_i:
                step = p_pows[lam_i - lam_j]
                opts_per_j.append(range(0, n_i, step))
            else:
                opts_per_j.append(range(n_i))

        candidates_i: list[tuple[int, ...]] = []
        for entries in product(*opts_per_j):
            s = 0
            for a, b in zip(entries, src):
                s += a * b
            if s % n_i == tgt[i]:
                candidates_i.append(tuple(entries))

        row_candidates.append(candidates_i)

    def backtrack(i: int,
                  current_rows: list[tuple[int, ...]],
                  basis: tuple[tuple[int, ...], ...]):
        if i == r:
            return [list(row) for row in current_rows]

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            if not any(row_mod_p):
                continue

            new_basis = _extend_basis_mod_p(basis, row_mod_p, inv_mod_p, p)
            if new_basis is None:
                continue

            result = backtrack(i + 1,
                               current_rows + [row],
                               new_basis)
            if result is not None:
                return result

        return None

    return backtrack(0, [], ())


# ============================================================
# 5) Main: push to lex-minimal representative
#    API: (p, lambdas)
# ============================================================

def push_to_lex_minimal(
    p: int,
    lambdas: list[int],
    v: list[int] | tuple[int, ...],
) -> list[list[int]]:
    """
    Given a p-group (p, lambdas) and a vector v, find the lexicographically
    minimal element w in the Aut(G)-orbit of v and return a matrix A with
    A*v = w.

    Input:
        p       : prime
        lambdas : [λ1, ..., λr]
        v       : iterable of ints, length r

    Output:
        A : list of rows, each a list of ints (an automorphism matrix).
    """
    lambdas = list(lambdas)
    r = len(lambdas)
    _, moduli = _group_prime_powers(p, tuple(lambdas))

    v_norm = tuple(v_i % n_i for v_i, n_i in zip(v, moduli))

    # Zero is already lex-minimal and fixed by every automorphism
    if all(x == 0 for x in v_norm):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    ord_v = element_order_p_group(p, lambdas, v_norm)

    # All lex-minimal representatives for this group, in lex order
    candidates = lex_minimal_vectors(p, lambdas)

    for w in candidates:
        if element_order_p_group(p, lambdas, w) != ord_v:
            continue

        A = _automorphism_sending_vector(p, lambdas, v_norm, tuple(w))
        if A is not None:
            return A

    # Should not happen if everything is consistent
    raise RuntimeError("No automorphism found mapping v to a lex-minimal representative.")


# ============================================================
# 6) (Optional) rank_mod_p helper (kept for compatibility)
#    NOTE: not used by the optimized code above.
# ============================================================

def rank_mod_p(rows, p: int) -> int:
    """
    Compute the rank over F_p of a matrix given by its list of rows.
    This is the original Gaussian elimination helper, kept for external uses.
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

        # Swap pivot row up
        M[rank], M[pivot] = M[pivot], M[rank]
        inv = pow(M[rank][col], -1, p)

        # Eliminate below
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
# 7) Equivalence under shifts (general finite abelian group)
#    NOTE: signature unchanged — works with arbitrary prime powers
#          but assumes inputs are valid and product <= 256.
# ============================================================

def is_single_equivalence_class_under_shifts(
    Z_wt: int,
    X_wt: int,
    prime_powers: list[int],
    vectors: list[list[int]] | list[tuple[int, ...]],
) -> bool:
    """
    Given:
        Z_wt, X_wt : nonnegative integers with len(vectors) == Z_wt + X_wt
        prime_powers: [n1, ..., nr], each ni a prime power (primes may differ)
        vectors     : list of group elements, each an iterable of length r

    We consider G = ⊕_i Z/ni Z and an equivalence relation generated by:

      For each v ∈ G, form
          S(v) = { vectors[i] + v   for i in range(Z_wt) }
               ∪ { vectors[i] - v   for i in range(Z_wt, Z_wt + X_wt) }

      (operations done component-wise modulo ni).

    Returns True iff this equivalence relation has exactly one equivalence
    class (i.e. the whole group is identified), and False otherwise.

    Assumptions (for speed, no checks):
        - Each ni is indeed a prime power.
        - len(vectors) == Z_wt + X_wt.
        - product(prime_powers) <= 256.
    """
    r = len(prime_powers)

    # Normalise vectors modulo each modulus
    base_vectors = [
        tuple(v_i % n_i for v_i, n_i in zip(v, prime_powers))
        for v in vectors
    ]

    # Enumerate all group elements (N <= 256 by assumption)
    elems = list(product(*(range(n_i) for n_i in prime_powers)))
    index_of = {e: i for i, e in enumerate(elems)}
    N = len(elems)

    # Union–find (disjoint set union)
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
            union(base_idx, index_of[w])

    # Check if all elements lie in a single equivalence class
    root0 = find(0)
    for i in range(1, N):
        if find(i) != root0:
            return False

    return True
