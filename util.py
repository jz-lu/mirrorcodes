"""
`util.py`

Code file with a bunch of simple helper functions that process and convert between
stim, Pauli string, and symplectic representations of codes.
"""
import stim
import numpy as np

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
    return ''.join(vec)


def stimify_stabs(stabs):
    """
    Convert a stabilizer tableau into stim convention.
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




