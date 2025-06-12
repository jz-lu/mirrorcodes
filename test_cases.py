import stim
import argparse
import numpy as np
from distance import distance
import itertools as it
from util import stimify_stabs, stimify_symplectic

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

def get_stabilizers(code):
    """
    Get the stim stabilizers of a given code, and whether or not they are CSS.

    Params:
        * code (str): description of the code. There are a few options.
    
    Returns:
        * stabs (list[stim.PauliString]): stabilizer tableau in stim form (list of stim.PauliString objects)
        * IS_CSS (bool): bit indicating if code is CSS.
    """
    stabs = None
    IS_CSS = False
    if code == "5qubit":
        stabs = stimify_stabs(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])
    elif code == "repetition":
        stabs = stimify_stabs(['ZZI', 'ZIZ'])
        IS_CSS = True
    elif code == "cookie":
        group = (6, 12)
        z0 = [(0,0), (0,1), (-3,2)]
        x0 = [(-1,-1), (-2,-1), (-3,2)]
        stabs_symp = find_stabilizers(group, z0, x0)

        stabs = stimify_symplectic(stabs_symp)
    elif code == "BB":
        IS_CSS = True
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
        raise ValueError(f"Unrecognized code name {code}")
    return stabs, IS_CSS

def main(args):
    code = args.code
    stabs, IS_CSS = get_stabilizers(code)
    dist = distance(stabs, IS_CSS=IS_CSS, verbose=True)
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
