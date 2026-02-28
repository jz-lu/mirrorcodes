"""
`util.py`

A bunch of simple helper functions that process and convert between
stim, Pauli string, and symplectic representations of codes.
"""
import itertools as it
import math
import numpy as np
import stim
import os

def gap_bat():
    gap_bat = r"C:/Users/andsin/AppData/Local/GAP-4.15.1/gap.bat"
    gap_root = os.path.dirname(os.path.abspath(gap_bat))
    runtime_bin = os.path.join(gap_root, "runtime", "bin")
    bash_exe = os.path.join(runtime_bin, "bash.exe")
    if not os.path.exists(bash_exe):
        gap_bat = r"C:/Program Files/GAP-4.15.1/gap.bat"
    return gap_bat

def symp2Pauli(x, n):
    """
    Return a sign-free Pauli string representation of the length 2`n` symplectic vector `x`.

    Input:
        * n (int): number of qubits.
        * x (numpy.ndarray): binary vector of length 2n, symplectically representing a n-qubit Pauli string.

    The convention for the symplectic vector is [Z | X] .
    
    Returns:
        * length-n string over {I, X, Y, Z} where the ith character is the Pauli on the ith qubit
    """
    Y_count = 0
    vec = []
    for i in range(n):
        char = 'I'
        if x[i] == 0 and x[i+n] == 1:
            char = 'X'
        elif x[i] == 1 and x[i+n] == 1:
            char = 'Y'
            Y_count += 1
        elif x[i] == 1 and x[i+n] == 0:
            char = 'Z'
        vec.append(char)
    if (Y_count // 2) % 2 == 1:
        return '-' + ''.join(vec)
    return ''.join(vec)


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
    stabs = [symp2Pauli(vec, n) for vec in stabs]
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

