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
from test_cases import symp2Pauli

def canonicalize_with_mult(group, z0, x0, mult):
    z0 = np.mod(z0 * mult, group)
    x0 = np.mod(x0 * mult, group)
    subtract = []
    for i, k in enumerate(group):
        if k % 2 == 0:
            subtract += [2 * (x0[0, i] // 2)]
        else:
            subtract += [x0[0, i]]
    x0 = np.mod(x0 - subtract, group)
    return np.vstack([z0, x0])

def canonicalize_with_order(group, z0, x0):
    z0 = np.array(z0)
    x0 = np.array(x0)
    x0 = np.mod(x0 - z0[0], group)
    z0 = np.mod(z0 - z0[0], group)
    assert max(z0[0]) == 0
    mults = [0] * len(group)
    for i, k in enumerate(group):
        candidates = set()
        for j in range(k):
            if np.gcd(j, k) == 1:
                candidates.add(j)
        for j in range(1, len(z0)):
            m = min(np.array(list(candidates)) * z0[j, i] % k)
            for l in candidates.copy():
                if z0[j, i] * l % k != m:
                    candidates.remove(l)
        mults[i] = list(candidates)
    sets = []
    for i in it.product(*mults):
        new = np.array([canonicalize_with_mult(group, z0, x0, i)])
        if len(sets) == 0:
            sets = new
            continue
        sets = np.append(sets, new, axis = 0)
    sets.sort(axis = 0)
    return sets[0]

def canonicalize_perms(group, z0, x0):
    results = []
    for i in it.permutations(z0):
        for j in it.permutations(x0):
            new = np.array([canonicalize_with_order(group, i, j)])
            if len(results) == 0:
                results = new
                continue
            results = np.append(results, new, axis = 0)
    return results[np.lexsort(results.reshape(len(results), -1).T[::-1])[0]]

def canonicalize(group, z0, x0):
    if len(z0) > len(x0):
        return canonicalize(group, x0, z0)
    if len(z0) == len(x0):
        options = np.array([canonicalize_perms(group, z0, x0),
                            canonicalize_perms(group, x0, z0)])
        options.sort(axis = 0)
        return options[0, :len(z0)], options[0, len(z0):]
    result = canonicalize_perms(group, z0, x0)
    return result[:len(z0)], result[len(z0):]


def build_set(group, a, b):
    s = []
    for i in a:
        for j in b:
            new = np.mod(j - i, group)
            if len(s) == 0:
                s = [new]
                continue
            s = np.vstack([s, new])
    return np.unique(s, axis = 0)

def css_flips(n, group, z0, x0, strides):
    zz = build_set(group, z0, z0)
    xx = build_set(group, x0, x0)
    same_diffs = np.unique(np.vstack([zz, xx]), axis = 0)
    zx = build_set(group, z0, x0)
    flips = [np.zeros(len(group), np.int64)]
    for g in same_diffs:
        gen_g = [g]
        cur = g
        while np.max(cur) > 0:
            cur = np.mod(cur + g, group)
            gen_g = np.vstack([gen_g, cur])
        cur_flips = flips.copy()
        for i in cur_flips:
            for j in gen_g:
                flips = np.vstack([flips, np.mod(i + j, group)])
        flips = np.unique(flips, axis = 0)
        if len(flips) == n:
            return False, set()
    group_times_two = []
    for g in it.product(*[range(a) for a in group]):
        new = np.mod(2 * np.array(g), group)
        if len(group_times_two) == 0:
            group_times_two = [new]
            continue
        group_times_two = np.vstack([group_times_two, new])
    group_times_two = np.unique(group_times_two, axis = 0)
    bad = set()
    for i in zx:
        for j in group_times_two:
            bad.add(np.mod(i + j, group) @ strides)
    for i in flips:
        if i @ strides in bad:
            return False, set()
    return True, flips


def find_stabilizers(group, z0, x0):
    group = np.array(group, np.int64)
    z0 = np.array(z0, np.int64)
    x0 = np.array(x0, np.int64)
    n = int(np.prod(group))
    d = len(group)
    strides = np.zeros(d, np.int64)
    strides[:-1] = np.cumprod(group[::-1])[d - 2::-1]
    strides[-1] = 1
    stabilizers = np.zeros((n, 2 * n), dtype = np.uint8)
    can_flip, flips = css_flips(n, group, z0, x0, strides)
    print(f"Your code is{'' if can_flip else ' NOT'} CSS")
    for i, g in enumerate(it.product(*[range(a) for a in group])):
        stabilizers[i, np.mod(z0 + g, group) @ strides] = 1
        stabilizers[i, np.mod(x0 - g, group) @ strides + n] = 1
    if can_flip:
        for g in flips:
            index = g @ strides
            stabilizers[:, [index, index + n]] = stabilizers[:, [index + n, index]] 
    return stabilizers

if __name__ == "__main__":
    """
    Run unit tests.
    """

    # Make some CSS codes and check if they are CSS
    CSS_group = (4, 7, 3)
    n = int(np.prod(CSS_group))
    X0 = ((0, 4, 1), (2, 3, 2))
    Z0 = ((1, 6, 2), (3, 1, 0), (1, 1, 1))
    print(canonicalize(CSS_group, X0, Z0))
    CSS_stabs = find_stabilizers(CSS_group, Z0, X0)
    print(f"Your CSS stabs are:")
    for stab in CSS_stabs:
        print(symp2Pauli(stab, n))

    # Make some non-CSS codes and check if they are CSS
