"""
`helix.py`
Code file to construct a helix code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} x ... x Z_{aN}
    * Z0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to Z0
    * X0 [{u1, u2, ..., uQ}]: a set of N-tuples specifying elements of the group which belong to X0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the helix code.
The check weight is exactly |Z0| + |X0| - |Z0 ^ X0| <= |Z0| + |X0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form---there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
"""

import numpy as np
import itertools as it

def find_stabilizers(group, z0, x0):
    n = int(np.prod(group))
    d = len(group)
    strides = np.zeros(d, np.int64)
    strides[:-1] = np.cumprod(group[::-1])[d - 2::-1]
    strides[-1] = 1
    stabilizers = np.zeros((n, 2 * n), dtype = np.uint8)
    for i, g in enumerate(it.product(*[range(a) for a in group])):
        g = np.array(g)
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1
    return stabilizers
