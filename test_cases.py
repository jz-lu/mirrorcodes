import stim
import argparse
import numpy as np
from distance import distance
import itertools as it

def find_stabilizers(group, z0, x0):
    n = int(np.prod(group))
    d = len(group)
    strides = np.zeros(d, dtype=np.int64)
    strides[:-1] = np.cumprod(group[::-1])[d - 2::-1]
    strides[-1] = 1
    stabilizers = np.zeros((n, 2 * n), dtype=np.uint8)
    for i, g in enumerate(it.product(*[range(a) for a in group])):
        g = np.array(g)
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1
    return stabilizers

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

def get_stabilizers(code):
    """
    Get the stim stabilizers of a given code
    """
    stabs = None
    if code == "5qubit":
        stabs = stimify_stabs(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])
    elif code == "repetition":
        stabs = stimify_stabs(['ZZI', 'ZIZ'])
    elif code == "cookie":
        group = (6, 12)
        z0 = [(0,0), (0,1), (-3,2)]
        x0 = [(-1,-1), (-2,-1), (-3,2)]
        stabs_symp = find_stabilizers(group, z0, x0)

        stabs = stimify_symplectic(stabs_symp)
    elif code == "BB":
        offsets = {1, -1, 1j, -1j, 3 + 6j, -6 + 3j}
        w = 24
        h = 12

        def wrap(c: complex) -> complex:
            return c.real % w + (c.imag % h)*1j

        def index_of(c: complex) -> int:
            c = wrap(c)
            return int(c.real + c.imag * w) // 2
        
        stabs: list[stim.PauliString] = []
        for x in range(w):
            for y in range(h):
                if x % 2 != y % 2:
                    continue  # This is a data qubit.
                m = x + 1j*y
                basis = 'XZ'[x % 2]
                sign = -1 if basis == 'Z' else +1
                stabilizer = stim.PauliString(w * h // 2)
                for offset in offsets:
                    stabilizer[index_of(m + offset * sign)] = basis
                stabs.append(stabilizer)
    else:
        assert False, f"Unrecognized code name {code}"
    return stabs

def main(args):
    code = args.code
    stabs = get_stabilizers(code)
    dist = distance(stabs)
    return dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the distance of some CSS and non-CSS codes"
    )

    parser.add_argument(
        "--code", "-c",
        type=str,
        default='BB',
    )
    args = parser.parse_args()
    main(args)