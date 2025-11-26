from itertools import product, combinations

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


def group_type_from_prime_powers(prime_powers):
    """
    Given a list [p**λ1, ..., p**λr] (all same p, already sorted by λi),
    return (p, [λ1,...,λr]).
    """
    ps = []
    lambdas = []
    for n in prime_powers:
        p, k = factor_prime_power(n)
        ps.append(p)
        lambdas.append(k)
    if len(set(ps)) != 1:
        raise ValueError("All factors must be powers of the *same* prime.")
    return ps[0], lambdas


# ---------- 1) Lexicographically minimal vectors ----------

def lex_minimal_vectors(prime_powers):
    """
    Given a finite abelian p-group described as a list of prime powers:

        prime_powers = [p**λ1, p**λ2, ..., p**λr]

    with the same prime p, and sorted non-decreasing by λi, return all
    lexicographically minimal vectors (one per Aut(G)-orbit) as described
    in the previous discussion.

    Each vector is a tuple (x_1, ..., x_r) with 0 <= x_i < prime_powers[i].
    Lex order is the usual tuple order in Python.
    """
    p, lambdas = group_type_from_prime_powers(prime_powers)
    r = len(lambdas)

    # Always include the zero vector (it is the only element in its orbit).
    vectors = {tuple(0 for _ in range(r))}

    indices = list(range(r))

    # We use the characterization:
    # Choose indices i_1 < ... < i_t and exponents α_j with
    #      0 <= α_1 < ... < α_t < λ_{i_t}
    # and
    #      λ_{i_1} - α_1 < ... < λ_{i_t} - α_t
    # Then the corresponding vector has x_{i_j} = p**α_j, others = 0.
    for t in range(1, r + 1):
        for idxs in combinations(indices, t):

            def rec(pos, prev_alpha, prev_beta, alphas):
                """Recursively build α_1 < ... < α_t with β_j strictly increasing."""
                if pos == t:
                    # Build the actual group element
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
                    if prev_beta is not None and not (beta > prev_beta):
                        continue
                    rec(pos + 1, alpha, beta, alphas + [alpha])

            rec(0, None, None, [])

    # Return as a sorted list in lexicographic order
    return sorted(vectors)


# ---------- 2) Automorphisms fixing given vectors ----------

def endo_entry_options(p, lam_i, lam_j):
    """
    Allowed values for matrix entry a_ij of a group endomorphism:

    - We view the matrix over Z / p^{lam_i} Z in row i.
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
        # Find a pivot row with nonzero entry in this column
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


def automorphisms_fixing_vectors(prime_powers, fixed_vectors):
    """
    Enumerate all automorphisms of the abelian p-group described by `prime_powers`
    that fix each vector in `fixed_vectors`.

    Input:
        prime_powers : list of ints [p**λ1, ..., p**λr],
                       sorted non-decreasing by λi, all powers of the same prime p.
        fixed_vectors: non-empty list of vectors (tuples/lists of length r).
                       Entry v[i] is in Z / prime_powers[i] Z (we reduce mod).

    Output:
        A list of matrices. Each matrix is a list of `r` rows,
        each row is a list of integers.

    Interpretation:
        Treat column vectors v as length-r tuples. The image A*v has i-th coordinate

            (sum_j A[i][j] * v[j]) mod prime_powers[i].

        Each returned matrix A is an automorphism of the group and
        satisfies A*v = v for every v in `fixed_vectors`.
    """
    if not fixed_vectors:
        raise ValueError("fixed_vectors must be non-empty.")

    p, lambdas = group_type_from_prime_powers(prime_powers)
    r = len(lambdas)

    # Normalize fixed vectors modulo the appropriate coordinate moduli
    fixed = []
    for v in fixed_vectors:
        if len(v) != r:
            raise ValueError("Each fixed vector must have length equal to number of factors.")
        fixed.append(tuple(v_i % n_i for v_i, n_i in zip(v, prime_powers)))

    # Precompute all candidate rows for each i, imposing:
    #   - endomorphism entry constraints (endo_entry_options),
    #   - the condition that A(v) = v in coordinate i for each fixed vector v.
    row_candidates = []
    for i in range(r):
        n_i = prime_powers[i]
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

    automorphisms = []

    def backtrack(i, current_rows, current_rows_mod_p):
        """
        Build the matrix row by row.

        current_rows       : rows over Z / p^{λ_i} Z
        current_rows_mod_p : corresponding rows reduced mod p
        """
        if i == r:
            # All rows chosen. By construction, rows are linearly independent
            # mod p, so det(A) is not divisible by p and A is an automorphism.
            automorphisms.append([list(row) for row in current_rows])
            return

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)

            # A zero row modulo p cannot appear in an invertible matrix.
            if all(c == 0 for c in row_mod_p):
                continue

            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)

            # If rank does not increase, row is dependent; determinant mod p would be 0.
            if new_rank != prev_rank + 1:
                continue

            backtrack(i + 1,
                      current_rows + [row],
                      new_rows_mod_p)

    backtrack(0, [], [])
    return automorphisms


# ---------- (Optional) tiny example usage ----------

if __name__ == "__main__":
    # Example: G = C_2 ⊕ C_4 ⊕ C_8
    G = [2, 4, 8]

    print("Lex-minimal representatives:")
    for v in lex_minimal_vectors(G):
        print(v)

    # All automorphisms (only fixing 0)
    autos_all = automorphisms_fixing_vectors(G, [(0, 0, 0)])
    print(f"\nNumber of automorphisms of C2 × C4 × C8: {len(autos_all)}")

    # Automorphisms that fix a nonzero element, e.g. (0,0,1)
    autos_fix = automorphisms_fixing_vectors(G, [(0, 0, 1)])
    print(f"Automorphisms fixing (0,0,1): {len(autos_fix)}")


from itertools import product

# ---------- Small helpers for the pushing algorithm ----------

def _p_adic_valuation(x, p):
    """
    v_p(x) for x != 0, with x an integer.
    (We will only call this with x reduced modulo a power of p and x != 0.)
    """
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def element_order_p_group(prime_powers, v):
    """
    Compute the order of a vector v in the p-group given by prime_powers.

    prime_powers: [p**λ1, ..., p**λr]
    v           : iterable of length r, entries are integers (reduced mod n_i internally).

    Returns: p^k, where k is the maximal exponent among the coordinates,
             or 1 if v is the zero vector.
    """
    p, lambdas = group_type_from_prime_powers(prime_powers)
    if len(v) != len(lambdas):
        raise ValueError("Vector length must match number of factors.")

    v_norm = [v_i % n_i for v_i, n_i in zip(v, prime_powers)]

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
        return 1  # order of the zero element

    return p ** max_exp


def _automorphism_sending_vector(prime_powers, source, target):
    """
    Internal helper:
    Try to construct a single automorphism A of the group (given by prime_powers)
    such that A * source = target.

    Returns:
        - a matrix A (list of rows) if possible
        - None if no such automorphism exists
    """
    p, lambdas = group_type_from_prime_powers(prime_powers)
    r = len(lambdas)

    if len(source) != r or len(target) != r:
        raise ValueError("source and target must have same length as prime_powers.")

    src = tuple(source[i] % prime_powers[i] for i in range(r))
    tgt = tuple(target[i] % prime_powers[i] for i in range(r))

    # Easy impossibility check: nonzero -> zero cannot happen under an automorphism.
    if all(x == 0 for x in src) and any(x != 0 for x in tgt):
        return None

    # Zero -> zero: identity works.
    if all(x == 0 for x in src) and all(x == 0 for x in tgt):
        return [[int(i == j) for j in range(r)] for i in range(r)]

    # Build candidate rows, enforcing A*src = tgt row-by-row and endomorphism constraints.
    row_candidates = []
    for i in range(r):
        n_i = prime_powers[i]
        opts_per_j = [endo_entry_options(p, lambdas[i], lambdas[j]) for j in range(r)]
        candidates_i = []
        for entries in product(*opts_per_j):
            row = list(entries)
            dot = sum(row[j] * src[j] for j in range(r)) % n_i
            if dot == tgt[i]:
                candidates_i.append(tuple(row))
        row_candidates.append(candidates_i)

    # Backtrack to pick rows such that the matrix is invertible modulo p.
    def backtrack(i, current_rows, current_rows_mod_p):
        if i == r:
            # Full matrix with full rank mod p ⇒ automorphism.
            return [list(row) for row in current_rows]

        prev_rank = rank_mod_p(current_rows_mod_p, p)

        for row in row_candidates[i]:
            row_mod_p = tuple(c % p for c in row)
            new_rows_mod_p = current_rows_mod_p + [row_mod_p]
            new_rank = rank_mod_p(new_rows_mod_p, p)

            # Require rank to increase by 1 at each step (independence mod p).
            if new_rank != prev_rank + 1:
                continue

            result = backtrack(i + 1, current_rows + [row], new_rows_mod_p)
            if result is not None:
                return result

        return None

    return backtrack(0, [], [])


# ---------- Main function: push to lex-minimal ----------

def push_to_lex_minimal(prime_powers, v):
    """
    Given a p-group described by prime_powers = [p**λ1, ..., p**λr] and a vector v,
    find the lexicographically minimal element w in the Aut(G)-orbit of v and
    return a pair (w, A), where A is an automorphism matrix with A * v = w.

    - prime_powers: list of prime powers, same prime, sorted by exponent.
    - v           : iterable of ints, length r.

    Returns:
        list of rows (each a list of ints), the automorphism matrix.
    """
    p, lambdas = group_type_from_prime_powers(prime_powers)
    r = len(lambdas)

    if len(v) != r:
        raise ValueError("Vector length must match number of factors.")

    # Normalize v modulo the group.
    v_norm = tuple(v_i % n_i for v_i, n_i in zip(v, prime_powers))

    # Zero is fixed by every automorphism; it's already lex-minimal.
    if all(x == 0 for x in v_norm):
        return v_norm

    # Order is an invariant under automorphisms, so only consider candidates with same order.
    ord_v = element_order_p_group(prime_powers, v_norm)

    # All lex-minimal representatives (one per orbit) for this group.
    candidates = lex_minimal_vectors(prime_powers)

    # Iterate in lex order and find the first candidate reachable from v.
    for w in candidates:
        if element_order_p_group(prime_powers, w) != ord_v:
            continue

        A = _automorphism_sending_vector(prime_powers, v_norm, w)
        if A is not None:
            return w

    # Theoretically this should never happen if everything is coded correctly.
    raise RuntimeError("No automorphism found mapping v to a lex-minimal representative.")
