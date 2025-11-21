from itertools import product
from math import prod

def list_automorphisms(
    g,
    include_all_lifts=False,
    max_count=None,
):
    """
    Enumerate automorphisms (as integer matrices) of the finite abelian group
        G = Z_{g[0]} × Z_{g[1]} × ... × Z_{g[m-1]},
    assuming each g[i] is a prime power.

    Parameters
    ----------
    g : list[int]
        Moduli for each cyclic factor. Every entry must be a prime power.
    include_all_lifts : bool, default False
        If False (default), produce one canonical lift for each valid mod-p pattern
        (fastest and still returns valid automorphisms).
        If True, enumerate *all* p-adic lifts (i.e., literally all automorphisms).
    max_count : int | None
        If provided, stop after yielding this many matrices.

    Yields
    ------
    A : list[list[int]]
        An m×m matrix (m=len(g)). Applying A to a vector x of residues and
        reducing component-wise modulo g gives the image under the automorphism.

    Notes
    -----
    • The matrix is block-diagonal across different primes; cross-prime blocks are zero.
    • For a fixed prime p, order indices by nondecreasing exponent e (g[i]=p^e).
      The endomorphism ring consists of matrices (a_ij) such that p^{e_i} | a_ij * p^{e_j},
      i.e. a_ij ≡ 0 (mod p^{max(0, e_i-e_j)}). Mod p this forces a_ij ≡ 0 when e_i>e_j.
      Automorphisms are those whose mod-p matrix has invertible diagonal blocks
      for each group of equal exponents.

    Helper:
      • apply_iso(A, x, g) -> image vector
    """

    # ---------- small utilities ----------
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        f = 3
        while f * f <= n:
            if n % f == 0:
                return False
            f += 2
        return True

    def is_prime_power(n):
        """Return (p, e) with n = p**e, or raise ValueError."""
        if n < 2:
            raise ValueError(f"{n} is not a valid prime power ≥ 2.")
        # find smallest prime divisor
        p = None
        t = n
        d = 2
        while d * d <= t and p is None:
            if t % d == 0:
                p = d
            d += 1 if d == 2 else 2
        if p is None:
            # n itself prime
            return n, 1
        # verify power
        e = 0
        while t % p == 0:
            t //= p
            e += 1
        if t != 1 or not is_prime(p):
            raise ValueError(f"{n} is not a prime power.")
        return p, e

    def matmul_mod(A, x, mod_vec):
        """Compute (A @ x) modulo component-wise moduli in mod_vec."""
        m = len(mod_vec)
        y = [0] * m
        for i in range(m):
            s = 0
            row = A[i]
            for j, aij in enumerate(row):
                s += aij * x[j]
            y[i] = s % mod_vec[i]
        return y

    # Rank over F_p for independence tests while enumerating GL(n,p).
    def rank_mod_p(M, p):
        """Row rank over F_p via Gaussian elimination (in place on a copy)."""
        if not M:
            return 0
        r = [row[:] for row in M]
        nrows = len(r)
        ncols = len(r[0])
        row = 0
        for col in range(ncols):
            # find pivot
            pivot = None
            for i in range(row, nrows):
                if r[i][col] % p != 0:
                    pivot = i
                    break
            if pivot is None:
                continue
            r[row], r[pivot] = r[pivot], r[row]
            inv = pow(r[row][col] % p, -1, p)
            # normalize pivot row
            r[row] = [(val * inv) % p for val in r[row]]
            # eliminate others
            for i in range(nrows):
                if i != row and r[i][col] % p != 0:
                    factor = r[i][col] % p
                    r[i] = [(r[i][j] - factor * r[row][j]) % p for j in range(ncols)]
            row += 1
            if row == nrows:
                break
        return row

    def enumerate_GL(n, p):
        """
        Yield all n×n invertible matrices over F_p efficiently by building
        an ordered basis column-by-column (no brute-force over p^{n^2}).
        """
        if n == 0:
            yield []
            return

        # recursive builder of independent columns
        def extend(cols):
            k = len(cols)
            if k == n:
                # Assemble matrix with these columns
                # Represent as row-major lists (more convenient later)
                Mcols = cols
                # transpose columns -> rows
                rows = [[Mcols[c][r] for c in range(n)] for r in range(n)]
                yield rows
                return
            # precompute rank of current columns (as rows for rank function)
            if k == 0:
                base_rows = []
                base_rank = 0
            else:
                # columns -> rows
                base_rows = [[cols[c][r] for c in range(k)] for r in range(n)]
                base_rank = rank_mod_p(base_rows, p)
            for vec in product(range(p), repeat=n):
                if all(v == 0 for v in vec):
                    continue  # must not be zero
                # check if independent of existing columns
                # build augmented rows and check rank increased by 1
                aug_rows = [row[:] for row in base_rows]
                # append vec as an extra column
                aug_rows = [aug_rows[r] + [vec[r]] for r in range(n)]
                if rank_mod_p(aug_rows, p) == base_rank + 1:
                    yield from extend(cols + [list(vec)])

        yield from extend([])

    # Enumerate all s×t matrices over F_p (row-major list of lists).
    def enumerate_mat_mod_p(rows, cols, p):
        for flat in product(range(p), repeat=rows * cols):
            # turn into row-major
            yield [list(flat[r * cols:(r + 1) * cols]) for r in range(rows)]

    # Build all automorphism matrices for a single prime block.
    def prime_block_automorphisms(indices, exponents, p):
        """
        indices : list[int]  (global indices belonging to this prime p)
        exponents : list[int] (e_i with g[indices[i]] = p**e_i), ordered nondecreasing
        p : prime

        Yields m×m integer matrices for this block (m=len(indices)), with entries
        already lifted to Z_{p^{e_i}} row-wise (minimal lift if include_all_lifts=False).
        """
        m = len(indices)
        # Partition local indices by exponent value
        exp_vals = sorted(set(exponents))
        groups = []
        for e in exp_vals:
            grp = [i for i, ee in enumerate(exponents) if ee == e]
            groups.append(grp)
        sizes = [len(grp) for grp in groups]

        # Precompute invertible diagonal options over F_p for each equal-exponent block
        diag_options = [list(enumerate_GL(sz, p)) for sz in sizes]

        # Helper: place a small block B into big mod-p matrix Mbar at rows R, cols C
        def place(Mbar, R, C, B):
            for rr, i in enumerate(R):
                for cc, j in enumerate(C):
                    Mbar[i][j] = B[rr][cc]

        # Iterate over all choices:
        # 1) Choose each diagonal block (invertible over F_p).
        for diag_choice in product(*diag_options):
            # Start with mod-p matrix
            Mbar = [[0] * m for _ in range(m)]
            # Place diagonal invertible blocks
            for blk_idx, grp in enumerate(groups):
                place(Mbar, grp, grp, diag_choice[blk_idx])

            # 2) Choose super-diagonal blocks freely over F_p (forced zeros below diagonal mod p)
            #    For groups k<l (exp[k] <= exp[l]), fill any matrix; for k>l, keep zeros.
            def fill_super_blocks(k_start, cur_Mbar):
                if k_start == len(groups):
                    # Fully determined mod-p pattern; now lift to Z_{p^{e_i}}
                    if include_all_lifts:
                        # Enumerate all p-adic lifts satisfying divisibility a_ij ≡ 0 (mod p^{max(0,e_i-e_j)})
                        # and a_ij ≡ cur_Mbar[i][j] (mod p) when allowed.
                        # We'll recurse entry-by-entry.
                        A = [[0] * m for _ in range(m)]
                        def lift_entry(i, j):
                            if i == m:
                                # Emit a copy of A
                                yield [row[:] for row in A]
                                return
                            ni = exponents[i]
                            nj = exponents[j]
                            delta = max(0, ni - nj)
                            mod_row = p ** ni
                            if delta >= 1:
                                # Must be multiples of p^delta; mod-p value must be 0
                                if cur_Mbar[i][j] % p != 0:
                                    return  # incompatible (shouldn't happen by construction)
                                choices = [ (t * (p ** delta)) % mod_row
                                            for t in range(p ** (ni - delta)) ]
                            else:
                                # Free w.r.t. divisibility; must reduce to cur_Mbar[i][j] modulo p
                                base = cur_Mbar[i][j] % p
                                choices = [ (base + p * t) % mod_row
                                            for t in range(p ** (ni - 1)) ] if ni > 0 else [base % mod_row]
                            for val in choices:
                                A[i][j] = val
                                # advance indices
                                nj_next = j + 1
                                ni_next = i
                                if nj_next == m:
                                    nj_next = 0
                                    ni_next = i + 1
                                yield from lift_entry(ni_next, nj_next)

                        yield from lift_entry(0, 0)
                    else:
                        # Minimal lift: just satisfy divisibility and mod-p constraints with smallest reps
                        A = [[0] * m for _ in range(m)]
                        for i in range(m):
                            for j in range(m):
                                ni = exponents[i]
                                nj = exponents[j]
                                delta = max(0, ni - nj)
                                if delta >= 1:
                                    A[i][j] = 0
                                else:
                                    A[i][j] = cur_Mbar[i][j] % p
                        yield A
                    return

                # For fixed row-group k_start, iterate all super blocks (k_start, l) with l>k_start
                k = k_start
                rows = groups[k]
                # recurse over all choices for blocks (k, l) for l>k
                def fill_for_l(l, Macc):
                    if l == len(groups):
                        # move to next k
                        fill_super_blocks(k_start + 1, Macc)
                        return
                    if l <= k:
                        fill_for_l(l + 1, Macc)
                        return
                    cols = groups[l]
                    # any matrix over F_p
                    for block in enumerate_mat_mod_p(len(rows), len(cols), p):
                        # place and continue
                        Mnext = [row[:] for row in Macc]
                        place(Mnext, rows, cols, block)
                        fill_for_l(l + 1, Mnext)

                fill_for_l(0, [row[:] for row in Mbar])

            yield from fill_super_blocks(0, Mbar)

    # ---------- main body ----------
    m = len(g)
    if m == 0:
        # The trivial group has exactly one automorphism: the empty matrix.
        yield []
        return

    # Validate and collect prime/exponent for each component
    pe = [is_prime_power(n) for n in g]  # list of (p,e)
    primes = sorted(set(p for p, _ in pe))

    # Group global indices by prime
    prime_to_indices = {p: [] for p in primes}
    prime_to_exps   = {p: [] for p in primes}
    for idx, (p, e) in enumerate(pe):
        prime_to_indices[p].append(idx)
        prime_to_exps[p].append(e)

    # For each prime, we need indices ordered by nondecreasing exponent
    for p in primes:
        zipped = list(zip(prime_to_indices[p], prime_to_exps[p]))
        zipped.sort(key=lambda t: t[1])  # sort by exponent
        prime_to_indices[p] = [i for i, _ in zipped]
        prime_to_exps[p]    = [e for _, e in zipped]

    # Enumerate per-prime block automorphisms, then combine into one big matrix
    per_prime_lists = []
    for p in primes:
        per_prime_lists.append(list(prime_block_automorphisms(prime_to_indices[p],
                                                              prime_to_exps[p],
                                                              p)))

    # Now combine via block-diagonal embedding into the full m×m matrix.
    # For each tuple of blocks, embed them and yield.
    count = 0
    for blocks in product(*per_prime_lists):
        # start with all zeros
        A = [[0] * m for _ in range(m)]
        for p, Ablock in zip(primes, blocks):
            idxs = prime_to_indices[p]
            # place
            for li, gi in enumerate(idxs):
                for lj, gj in enumerate(idxs):
                    A[gi][gj] = Ablock[li][lj] % g[gi]
        yield A
        count += 1
        if max_count is not None and count >= max_count:
            return


# ---------- convenience helper to apply an automorphism ----------
def apply_iso(A, x, g):
    """Apply matrix A to vector x and reduce component-wise modulo g."""
    if len(A) != len(g) or any(len(row) != len(g) for row in A):
        raise ValueError("Matrix shape must be len(g) × len(g).")
    if len(x) != len(g):
        raise ValueError("Vector length must be len(g).")
    y = []
    for i, row in enumerate(A):
        s = 0
        for aij, xj in zip(row, x):
            s += aij * xj
        y.append(s % g[i])
    return y
