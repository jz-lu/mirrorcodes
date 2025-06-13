"""
`search.py`
Code file for conducting a search for good helix codes.
This consists of two steps:
    1) Generate a helix code according to some systematic protocol.
    2) Evaluate if the code is good. If so, keep it. If not, delete and continue.
We can start by searching through codes for which n ~ 100 +/- 100, and check weight <= 6.
But let n >> check weight so that the LDPCness kicks in, e.g. n >= 30.

The precise meaning of "good" is debatable, but we will adopt the following two-stage filtering method.
Stage 1 (distance-rate tradeoff):
    * Evaluate the rate R of the code. If R < 1/16, discard.
    * Evaluate the distance d of the code. If evaluation of the distance takes >3 min, keep the code.
    * If the distance is calculated successfully, discard if Rd < 1/2. Keep otherwise.

Stage 2 (practicality):
    * Evaluate the pseudo-threshold using BP-OSD. If it is above some TBD cutoff, keep.
    * Evaluate the circuit distance?
"""
import numpy as np
import itertools as it
from primefac import primefac
from helix import canonicalize, find_stabilizers
from distance import distance

def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def is_canonical(helix_code):
    """
    Check whether `helix_code` is in canonical form.
    We say that a helix code (group, Z0, X0) is in canonical form if TODO
    
    Input:
        * `helix_code` (tuple): Tuple of (group, Z0, X0) where group is a tuple and Z0 and X0 are sets
    
    Returns:
        * Binary indication of whether `helix_code` is in canonical form.
    """
    return helix_code == canonicalize(*helix_code)


def n_level_search(n, Z_wt, X_wt):
    """
    Search helix codes over `n` qubits with Z-weight `Z_wt` and X-weight `X_wt`.
    Save the ones which pass stage 1.

    Input:
        * n (int): number of qubits
        * Z_wt (int): weight of the Z's in each check
        * X_wt (int): weight of the X's in each check
    
    Output:
        * A .npy file for each code which passes the stage 1 test
    
    Returns:
        * None
    """
    pass

def num_indices(n, Z_wt, X_wt):
    result = 1
    for i in it.groupby(list(primefac(n))):
        result *= len(list(partitions(len(list(i[1])))))
    return result * n ** (Z_wt + X_wt - 1)

def index_to_tuple(group, index):
    result = []
    for g in group[::-1]:
        result.append(index % g)
        index //= g
    return (*result[::-1],)

def multi_index_to_tuples(group, Z_wt, X_wt, index):
    n = int(np.prod(group))
    Zs = []
    Xs = []
    Z_nums = []
    X_nums = []
    for _ in range(X_wt):
        Xs.append(index_to_tuple(group, index))
        X_nums.append(index % n)
        index //= n
    for _ in range(Z_wt - 1):
        Zs.append(index_to_tuple(group, index))
        Z_nums.append(index % n)
        index //= n
    Zs.append((0,) * len(group))
    Z_nums.append(0)
    return Zs[::-1], Xs[::-1], Z_nums[::-1], X_nums[::-1]

def n_partitions(n):
    primes = []
    powers = []
    for i in it.groupby(list(primefac(n))):
        primes.append(i[0])
        powers.append(len(list(i[1])))
    combos = [list(partitions(i)) for i in powers]
    result = []
    for i in it.product(*combos):
        group_sizes = []
        for j, p in zip(i, primes):
            for k in j:
                group_sizes.append(p ** k)
        result.append((*group_sizes,))
    return result

'''
index_end is exclusive
'''
def process_codes(n, Z_wt, X_wt, index_start = 0, index_end = None):
    """
    FILL IN THIS DOCUMENTATION

    Params:
        * Fill me in
    
    Returns:
        * list of helix codes in (group, Z_0, X_0) form.
    
    """
    if index_end == None:
        index_end = num_indices(n, Z_wt, X_wt)
    groups = n_partitions(n)
    index = index_start
    COUNTER = 0
    while index < index_end:
        quotient, remainder = divmod(index, n ** (Z_wt + X_wt - 1))
        group = groups[quotient]
        Zs, Xs, Z_nums, X_nums = multi_index_to_tuples(group, Z_wt, X_wt, remainder)
        jump =  -1
        for i in range(1, Z_wt + X_wt):
            if (any([j > 0 and g % j > 0 for j, g in zip(Zs[1], group)]) or
                (i < Z_wt and Z_nums[i] <= Z_nums[i - 1]) or
                (i > Z_wt and X_nums[i - Z_wt] <= X_nums[i - Z_wt - 1]) or
                (i == Z_wt and (max(Xs[0]) > 1 or
                    any([j > 0 and g % 2 > 0 for j, g in zip(Xs[0], group)])))):
                jump = n ** (Z_wt + X_wt - 1 - i)
                break
        group = np.array(group)
        Zs = np.array(Zs)
        Xs = np.array(Xs)
        if jump < 0:
            canon_Z, canon_X = canonicalize(group, Zs, Xs)
            if not (np.all(Zs == canon_Z) and np.all(Xs == canon_X)):
                jump = 1
        if jump >= 0:
            index = (index // jump + 1) * jump
            continue
        COUNTER += 1
        index += 1
    return COUNTER

def main():
    pass

if __name__ == "__main__":
    main()
