"""
`util.py`

A bunch of simple helper functions that process and convert between
stim, Pauli string, and symplectic representations of codes.
"""
import numpy as np
import stim
import os

# ---------- GF(2) linear algebra helpers ----------

def _gf2_rref(A: np.ndarray):
    """RREF of A over GF(2). Returns (R, pivot_cols)."""
    A = (A.copy() & 1).astype(np.uint8)
    n_rows, n_cols = A.shape
    pivots = []
    r = 0
    for c in range(n_cols):
        pivot = None
        for rr in range(r, n_rows):
            if A[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for rr in range(n_rows):
            if rr != r and A[rr, c]:
                A[rr] ^= A[r]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return A, pivots

def _gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """
    Basis for nullspace of A over GF(2) as a (k, n_cols) uint8 array.
    Solves A x = 0.
    """
    R, pivots = _gf2_rref(A)
    n_rows, n_cols = R.shape
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]

    basis = []
    for f in free_cols:
        v = np.zeros(n_cols, dtype=np.uint8)
        v[f] = 1
        # pivot var equals the coefficient in the pivot row (since RHS=0)
        for i, pc in enumerate(pivots):
            if R[i, f]:
                v[pc] = 1
        basis.append(v)

    return np.zeros((0, n_cols), dtype=np.uint8) if not basis else np.stack(basis, axis=0)

def code_connected(stabs):
    n = len(stabs)
    reduced, _ = _gf2_rref(np.array(stabs))
    sum_ = reduced[:, :n] + reduced[:, n:]
    found = np.array([False] * n)
    found[0] = True
    visited = set()
    while True:
        found_something = False
        for i in range(n):
            if found[i] and i not in visited:
                found_something = True
                visited.add(i)
                for j in range(n):
                    for k in range(n):
                        if sum_[j][i] > 0 and sum_[j][k] > 0:
                            found[k] = True
        if not found_something:
            break
    return (found).all()

def _gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2) for one solution x (free vars set to 0).
    Raises ValueError if inconsistent.
    """
    A = (A.copy() & 1).astype(np.uint8)
    b = (b.copy() & 1).astype(np.uint8)
    k, m = A.shape
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)

    pivots = []
    r = 0
    for c in range(m):
        pivot = None
        for rr in range(r, k):
            if aug[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]
        for rr in range(k):
            if rr != r and aug[rr, c]:
                aug[rr] ^= aug[r]
        pivots.append(c)
        r += 1
        if r == k:
            break

    # inconsistency: 0 = 1 row
    for rr in range(k):
        if aug[rr, :m].sum() == 0 and aug[rr, m]:
            raise ValueError("No GF(2) solution: sign constraints are inconsistent.")

    x = np.zeros(m, dtype=np.uint8)
    # in RREF with free vars = 0, pivot vars equal RHS
    for i, pc in enumerate(pivots):
        x[pc] = aug[i, m]
    return x


# ---------- Pauli phase computation (canonical Y convention) ----------

def _check_commuting(z: np.ndarray, x: np.ndarray) -> bool:
    # symplectic commutator: (z x^T + x z^T) mod 2 must be 0
    comm = (z @ x.T + x @ z.T) & 1
    return bool(np.all(comm == 0))


def _product_phase_for_subset(
    z: np.ndarray,
    x: np.ndarray,
    subset: np.ndarray,
    sign_bits: np.ndarray | None = None,
):
    """
    Multiply the chosen generators (subset[i]=1).

    Representation used:
      generator i corresponds to  i^{p_i} X^{x_i} Z^{z_i}
      with canonical p_i = (z_i · x_i) mod 4  (this makes overlaps into Y's),
      and an extra -1 sign flip adds +2 mod 4.

    Returns (z_acc, x_acc, p_acc mod 4).
    """
    m, n = z.shape
    subset = (subset & 1).astype(np.uint8)
    if sign_bits is None:
        sign_bits = np.zeros(m, dtype=np.uint8)
    sign_bits = (sign_bits & 1).astype(np.uint8)

    p_acc = 0
    z_acc = np.zeros(n, dtype=np.uint8)
    x_acc = np.zeros(n, dtype=np.uint8)

    for i in np.nonzero(subset)[0]:
        p_i = (int((z[i] & x[i]).sum()) % 4 + 2 * int(sign_bits[i])) % 4
        # multiplying current (X^x_acc Z^z_acc) by (X^x_i Z^z_i)
        # contributes (-1)^{z_acc · x_i} = i^{2*(z_acc·x_i mod2)} to the phase.
        comm_parity = int((z_acc & x[i]).sum()) & 1
        p_acc = (p_acc + p_i + 2 * comm_parity) % 4
        z_acc ^= z[i]
        x_acc ^= x[i]

    return z_acc, x_acc, p_acc


# ---------- Main function ----------

def sign_corrections_symplectic(H: np.ndarray) -> np.ndarray:
    """
    Input:
      H: (m x 2n) binary matrix in symplectic form [Z | X]
         (Z columns first, then X columns).

    Output:
      f: length-m binary vector.
         If f[i]=1, multiply stabilizer i by -1 (flip its sign) to eliminate any -I
         produced by linear relations among the rows (under the canonical Y convention).

    Notes:
      - Assumes the rows mutually commute (raises ValueError otherwise).
      - Assumes the current signs are all +1; if you have existing sign bits s,
        you would use: s_corrected = s XOR f.
    """
    H = (np.array(H, dtype=np.uint8) & 1)
    if H.ndim != 2 or (H.shape[1] % 2) != 0:
        raise ValueError("H must be an m x 2n binary matrix (even number of columns).")

    m, two_n = H.shape
    n = two_n // 2
    z = H[:, :n]
    x = H[:, n:]

    if not _check_commuting(z, x):
        raise ValueError("Rows do not mutually commute (not a stabilizer generator set).")

    # Relations among generators are lambda in GF(2)^m with H^T lambda = 0
    rel_basis = _gf2_nullspace(H.T)  # shape (k, m)
    k = rel_basis.shape[0]
    if k == 0:
        return np.zeros(m, dtype=np.uint8)

    # For each independent relation, detect whether the product is -I (need parity 1) or +I (parity 0)
    r = np.zeros(k, dtype=np.uint8)
    for j in range(k):
        lam = rel_basis[j]
        z_acc, x_acc, p_acc = _product_phase_for_subset(z, x, lam)
        if z_acc.any() or x_acc.any():
            raise RuntimeError("Internal error: relation did not cancel supports.")
        if p_acc == 2:
            r[j] = 1
        elif p_acc == 0:
            r[j] = 0
        else:
            raise ValueError("A cancelling relation produced ±iI (unexpected for commuting Hermitian Paulis).")

    # Choose sign flips f so that every relation product becomes +I:
    # for each relation lam:  lam · f = r   (mod 2)
    f = _gf2_solve(rel_basis, r)

    # (Optional) verify on the basis relations
    for j in range(k):
        _, _, p_acc = _product_phase_for_subset(z, x, rel_basis[j], sign_bits=f)
        if p_acc != 0:
            raise RuntimeError("Verification failed: sign correction did not fix all basis relations.")

    return f

def gap_bat():
    gap_bat = r"C:/Users/andsin/AppData/Local/GAP-4.15.1/gap.bat"
    gap_root = os.path.dirname(os.path.abspath(gap_bat))
    runtime_bin = os.path.join(gap_root, "runtime", "bin")
    bash_exe = os.path.join(runtime_bin, "bash.exe")
    if not os.path.exists(bash_exe):
        gap_bat = r"C:/Program Files/GAP-4.15.1/gap.bat"
    return gap_bat

def symp2Pauli(x, n, positive=True):
    """
    Return a sign-free Pauli string representation of the length 2`n` symplectic vector `x`.

    Input:
        * n (int): number of qubits.
        * x (numpy.ndarray): binary vector of length 2n, symplectically representing a n-qubit Pauli string.

    The convention for the symplectic vector is [Z | X] .
    
    Returns:
        * length-n string over {I, X, Y, Z} where the ith character is the Pauli on the ith qubit
    """
    vec = []
    for i in range(n):
        char = 'I'
        if x[i] == 0 and x[i+n] == 1:
            char = 'X'
        elif x[i] == 1 and x[i+n] == 1:
            char = 'Y'
        elif x[i] == 1 and x[i+n] == 0:
            char = 'Z'
        vec.append(char)
    return ('' if positive else '-') + ''.join(vec)


def stimify_stabs(stabs):
    """
    Convert a stabilizer tableau in Pauli string notation into stim convention.
    """
    return [stim.PauliString(x) for x in stabs]


def stimify_symplectic(stabs):
    """
    Convert a symplectic tableau into stim convention.
    """
    assert len(stabs[0]) % 2 == 0 and len(stabs[0]) > 0
    n = len(stabs[0]) // 2
    signs = sign_corrections_symplectic(stabs)
    stabs = [symp2Pauli(vec, n, positive=(sign == 0)) for vec, sign in zip(stabs, signs)]
    return stimify_stabs(stabs)


def is_CSS(stabs, n):
    """
    Test whether a stabilizer tableau is CSS. The code is defined to be CSS
    if every check is either all X's or all Z's.

    Params:
        * stabs (np.ndarray): r x 2n binary matrix giving the stabilizer tableau in symplectic representation.
        * n (int): number of qubits.
    
    Returns:
        * bool indicating if `stabs` represents a CSS code.
    """
    is_unmixed = lambda x: np.all(x[:n] == 0) or np.all(x[n:] == 0)
    return np.all([is_unmixed(vec) for vec in stabs])

def find_strides(group):
    """Finds index incrementing values for a given group.

    Params:
        * group (np.ndarray): A list containing the group.

    Returns:
        * Array with index incrementing values, the backwards cumulative product.
    """
    if len(group) == 1:
        return [1]
    return np.append(np.cumprod(group[::-1])[len(group) - 2::-1], 1)

def partitions(n, I=1):
    """
    Finds all partitions of n. I a lower bound on elements of the partitions,
    which is 1 by default. This function should be moved to util.py
    
    Input:
        * n (int): the number whose partitions we want to compute.
        * I (int, optional): the minimum entry in the partitions, 1 by default
    
    Returns:
        * Iterable of tuples containing the partitions of n with minimum I or more
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p

def index_to_array(group, index):
    """
    Compute an array representing a qubit in group, given an index from 0 to n - 1,
    the size of the group. This is no different than expressing a number "base
    group". Notably, this works for larger indices too, but will only consider the
    index mod n.

    Params:
        * group (np.ndarray): the group we are decomposing the index into
        * index (int): A number from 0 to n - 1 corresponding to a tuple mod group.
    
    Returns:
        * A number array with the same length as group, corresponding to the indexth
        array mod group.
    """
    result = []
    for g in group[::-1]:
        result += [int(index % g)]
        index //= g
    return result[::-1]

def binary_rank(A):
    """
    Finds the rank mod 2 of a matrix A without explicit row swapping,
    using XOR to absorb the pivot row.

    Params:
        * A (np.ndarray): the matrix whose rank we want to find

    Returns:
        * int containing the rank of A
    """
    M = np.array(A, dtype=np.uint8) & 1
    m, n = M.shape
    r = 0
    for c in range(n):
        for i in range(r, m):
            if M[i, c]:
                if i != r:
                    M[r] ^= M[i]
                break
        else:
            continue
        rows_to_eliminate = np.where(M[r+1:, c])[0] + (r + 1)
        M[rows_to_eliminate] ^= M[r]
        r += 1
        if r == m:
            break
    return r
