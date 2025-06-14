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
    """
    Finds all partitions of n. I a lower bound on elements of the partitions, which is 1 by default. This function should be moved to util.py
    
    Input:
        * n (int): the number whose partitions we want to compute.
        * I (int, optional): the minimum entry in the partitions, 1 by default
    
    Returns:
        * Iterable of tuples containing the partitions of n with minimum I or more
    """
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

def passes_stage_1(helix_code):
    """
    Evaluate whether a given code passes through stage 1 of tests.

    Input:
        * `helix_code` (tuple): Tuple of (group, Z0, X0) where group is a tuple and Z0 and X0 are sets
    
    Returns:
        * Binary indication of whether `helix_code` passes stage 1 of tests.
    """
    pass

def group_level_search(group, Z_wt, X_wt):
    """
    Search helix codes formed from `group` with Z-weight `Z_wt` and X-weight `X_wt`.
    Save the ones which pass stage 1.

    Input:
        * group (tuple): tuple of integers specifying the group
        * Z_wt (int): weight of the Z's in each check
        * X_wt (int): weight of the X's in each check
    
    Output:
        * A .npy file for each code which passes the stage 1 test
    
    Returns:
        * None
    """
    pass

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
    """
    Compute number of possible codes on n qubits with given weights. This does not assume any canonicalization whatsoever, other than setting Z0[0] to 0. This function exists to have nice ranges over which to index. This returns the number of partitions of the powers of n times n to the power of Z_wt + X_wt - 1.

    Params:
        * n (int): The number of qubits we want to search over
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
    
    Returns:
        * An int with the number of possible codes on n qubits. Calculates the number of partitions of the powers of n times number of ways to pick qubits in Z0 and X0, not counting Z0[0].
    """
    result = 1
    #find all the powers of primes in n, and count how many partitions they have
    for i in it.groupby(list(primefac(n))):
        result *= len(list(partitions(len(list(i[1])))))
    return result * n ** (Z_wt + X_wt - 1)

def index_to_tuple(group, index):
    """
    Compute a tuple representing a qubit in group, given an index from 0 to n - 1, the size of the group. This is no different than expressing a number "base group". Notably, this works for larger indices too, but will only consider the index mod n.

    Params:
        * group (np.ndarray): the group we are decomposing the index into
        * index (int): A number from 0 to n - 1 corresponding to a tuple mod group.
    
    Returns:
        * A tuple with the same length as group, corresponding to the indexth tuple mod group.
    """
    result = []
    for g in group[::-1]:
        result.append(index % g)
        index //= g
    return (*result[::-1],)

def multi_index_to_tuples(group, Z_wt, X_wt, index):
    """
    Compute several tuples representing some qubits in group, which correspond to the values in Z0 and X0. This is done given an index from 0 to n ** (Z_wt + X_wt - 1) - 1. This is no different than expressing several numbers "base group". Notably, this works for larger indices too, but will only consider the index mod n ** (Z_wt + X_wt). Regardless of what the index says, Z0[0] is force-set to the all 0's tuple.

    Params:
        * group (np.ndarray): the group we are decomposing the index into
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index (int): A number from 0 to n ** (Z_wt + X_wt - 1) - 1 corresponding to several tuples mod group.
    
    Returns:
        * Four lists. The first has length Z_wt and is a list of tuples containing the values of Z0 as given by index (after dividing by n ** X_wt) with a tuple of zeros prepended. The second has length X_wt and is a list of tuples containing the values of X0 as given by index (after modding by n ** X_wt). The third and fourth lists contain the indices used to construct the tuples of the first and second lists, respectively. Each entry in the third and fourth lists is from 0 to n - 1.
    """
    n = int(np.prod(group))
    Zs = []
    Xs = []
    Z_nums = []
    X_nums = []

    #build Xs list
    for _ in range(X_wt):
        Xs.append(index_to_tuple(group, index))
        X_nums.append(index % n)
        index //= n

    #build Zs list
    for _ in range(Z_wt - 1):
        Zs.append(index_to_tuple(group, index))
        Z_nums.append(index % n)
        index //= n

    #prepend Z0[0]
    Zs.append((0,) * len(group))
    Z_nums.append(0)

    return Zs[::-1], Xs[::-1], Z_nums[::-1], X_nums[::-1]

def n_partitions(n):
    """
    Find all non-isomorphic abelian groups of n qubits.

    Params:
        * n (int): The number of qubits
    
    Returns:
        * A list of tuples, each containing the sizes of various abelian groups, such that the product of all the groups is n. Each group must be a power of a prime.
    """
    primes = []
    powers = []

    #find prime factorization of n
    for i in it.groupby(list(primefac(n))):
        primes.append(i[0])
        powers.append(len(list(i[1])))

    #find all partitions of the powers in the prime factorization of n
    combos = [list(partitions(i)) for i in powers]

    #find all combinations of decompositions of the various prime powers according to the above partitions
    result = []
    for i in it.product(*combos):
        group_sizes = []
        for j, p in zip(i, primes):
            for k in j:
                group_sizes.append(p ** k)
        result.append((*group_sizes,))
    
    return result

def process_codes(n, Z_wt, X_wt, index_start = 0, index_end = None):
    """
    Wrapper for processing only codes which are canonical. Currently returns the number of valid codes.

    Params:
        * n (int): The number of physical qubits of the desired code
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index_start (int, optional): The index at which the loop should start counting codes. Indices can range from 0 to n ** (Z_wt + X_wt - 1) - 1. 0 by default. 
        * index_end (int, optional): The index at which the loop should stop counting codes. The code at index_end is not counted. Indices can range from 0 to n ** (Z_wt + X_wt - 1) - 1. None by default, which is then set to the max value. 
    
    Returns:
        * Number of canonical codes with n qubits.
    """
    #find max index
    if index_end == None:
        index_end = num_indices(n, Z_wt, X_wt)

    #find all group partitions
    groups = n_partitions(n)

    index = index_start
    COUNTER = 0
    #for each index...
    while index < index_end:
        quotient, remainder = divmod(index, n ** (Z_wt + X_wt - 1))
        #find the relevant group
        group = groups[quotient]
        #find the values of Z0 and X0 from the index
        Zs, Xs, Z_nums, X_nums = multi_index_to_tuples(group, Z_wt, X_wt, remainder)
        jump =  -1
        #for each value of Z0 and X0, other than Z0[0]...
        for i in range(1, Z_wt + X_wt):
            #see if there are any immediate mistakes
            #mistakes include not being in order, or Z0[1] not consisting of divisors of n, or X0[0] not consisting of 0's or 1's where appropriate
            if (any([j > 0 and g % j > 0 for j, g in zip(Zs[1], group)]) or
                (i < Z_wt and Z_nums[i] <= Z_nums[i - 1]) or
                (i > Z_wt and X_nums[i - Z_wt] <= X_nums[i - Z_wt - 1]) or
                (i == Z_wt and (max(Xs[0]) > 1 or
                    any([j > 0 and g % 2 > 0 for j, g in zip(Xs[0], group)])))):
                #compute by how much the index should be incremented based on the position of the mistake
                jump = n ** (Z_wt + X_wt - 1 - i)
                break
        group = np.array(group)
        Zs = np.array(Zs)
        Xs = np.array(Xs)
        #if this is a valid code candidate, check that it is canonical
        if jump < 0:
            canon_Z, canon_X = canonicalize(group, Zs, Xs)
            if not (np.all(Zs == canon_Z) and np.all(Xs == canon_X)):
                jump = 1
        #if code is not canonical, increment the index
        if jump >= 0:
            index = (index // jump + 1) * jump
            continue
        #if we have reached this point, the code is canonical and should be counted
        COUNTER += 1
        index += 1
    return COUNTER

def main():
    for i in range(16):
        print(i, process_codes(i, 3, 3))

if __name__ == "__main__":
    main()
