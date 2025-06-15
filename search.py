"""
`search.py`
Code file for conducting a search for good helix codes.
This consists of two steps:
    1) Generate a helix code according to some systematic protocol.
    2) Evaluate if the code is good. If so, keep it. If not, delete and continue.
We can start by searching through codes for which n ~ 100 +/- 100, and check
weight <= 6.
But let n >> check weight so that the LDPCness kicks in, e.g. n >= 30.

The precise meaning of "good" is debatable, but we will adopt the following
two-stage filtering method.
Stage 1 (distance-rate tradeoff):
    * Evaluate the rate R of the code. If R < 1/16, discard.
    * Evaluate the distance d of the code. If evaluation of the distance takes >3
      min, keep the code.
    * If the distance is calculated successfully, discard if Rd < 1/2. Keep
      otherwise.

Stage 2 (practicality):
    * Evaluate the pseudo-threshold using BP-OSD. If it is above some TBD cutoff,
      keep.
    * Evaluate the circuit distance?
"""
import itertools as it
from math import gcd
import numpy as np
from primefac import primefac

from constants import RATE_THRESHOLD
from distance import distance
from helix import canonicalize, is_X_canonical, is_Z_canonical
from util import compute_rank_from_tuples, find_isos, find_strides, \
                 index_to_tuple, partitions

def is_canonical(helix_code):
    """
    Check whether `helix_code` is in canonical form.
    We say that a helix code (group, Z0, X0) is in canonical form if TODO
    
    Input:
        * helix_code (tuple): Tuple of (group, Z0, X0) where group is a tuple
          and Z0 and X0 are sets
    
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
    """
    Compute number of possible codes on n qubits with given weights. This does not
    assume any canonicalization whatsoever, other than setting Z0[0] to 0. This
    function exists to have nice ranges over which to index. This returns the
    number of partitions of the powers of n times n to the power of Z_wt + X_wt - 1.

    Params:
        * n (int): The number of qubits we want to search over
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
    
    Returns:
        * An int with the number of possible codes on n qubits. Calculates the
        number of partitions of the powers of n times number of ways to pick
        qubits in Z0 and X0, not counting Z0[0].
    """
    result = 1
    #find all the powers of primes in n, and count how many partitions they have
    for i in it.groupby(list(primefac(n))):
        result *= len(list(partitions(len(list(i[1])))))
    return result * n ** (Z_wt + X_wt - 1)

def multi_index_to_tuples(group, Z_wt, X_wt, index):
    """
    Compute several tuples representing some qubits in group, which correspond to
    the values in Z0 and X0. This is done given an index
    from 0 to n ** (Z_wt + X_wt - 1) - 1. This is no different than expressing
    several numbers "base group". Notably, this works for larger indices too, but
    will only consider the index mod n ** (Z_wt + X_wt). Regardless of what the
    index says, Z0[0] is force-set to the all 0's tuple.

    Params:
        * group (np.ndarray): the group we are decomposing the index into
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index (int): A number from 0 to n ** (Z_wt + X_wt - 1) - 1 corresponding
          to several tuples mod group.
    
    Returns:
        * Four lists. The first has length Z_wt and is a list of tuples containing
          the values of Z0 as given by index (after dividing by n ** X_wt) with a
          tuple of zeros prepended. The second has length X_wt and is a list of
          tuples containing the values of X0 as given by index (after modding by
          n ** X_wt). The third and fourth lists contain the indices used to
          construct the tuples of the first and second lists, respectively. Each
          entry in the third and fourth lists is from 0 to n - 1.
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
        * A list of tuples, each containing the sizes of various abelian groups,
          such that the product of all the groups is n. Each group must be a power
          of a prime.
    """
    primes = []
    powers = []

    #find prime factorization of n
    for i in it.groupby(list(primefac(n))):
        primes.append(i[0])
        powers.append(len(list(i[1])))

    #find all partitions of the powers in the prime factorization of n
    combos = [list(partitions(i)) for i in powers]

    #find all combinations of decompositions of the various prime powers according
    #to the above partitions
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
    Wrapper for processing only codes which are canonical. Currently returns the
    number of valid codes.

    Params:
        * n (int): The number of physical qubits of the desired codes
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * index_start (int, optional): The index at which the loop should start
          counting codes. Indices can range from 0 to n ** (Z_wt + X_wt - 1) - 1. 0
          by default. 
        * index_end (int, optional): The index at which the loop should stop
          counting codes. The code at index_end is not counted. Indices can range
          from 0 to n ** (Z_wt + X_wt - 1) - 1. None by default, which is then set
          to the max value. 
    
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
            #mistakes include not being in order, or Z0[1] not consisting of
            #divisors of the group size, or X0[0] not consisting of 0's or 1's
            #where appropriate
            if (any([j > 0 and g % j > 0 for j, g in zip(Zs[1], group)]) or
                (i < Z_wt and Z_nums[i] <= Z_nums[i - 1]) or
                (i > Z_wt and X_nums[i - Z_wt] <= X_nums[i - Z_wt - 1]) or
                (i == Z_wt and (max(Xs[0]) > 1 or
                    any([j > 0 and g % 2 > 0 for j, g in zip(Xs[0], group)])))):
                #compute by how much the index should be incremented based on the
                #position of the mistake
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

def decomposed_Z0_candidates(Z_wt, size):
    """
    Finds possible elements of Z0[:, k] that are worth searching over for some k.

    Params:
        * Z_wt (int): The number of terms in Z0
        * size (int): The size of the cyclic group we are finding candidates for
    
    Returns:
        * List of tuples. Each tuple contains a list of entries of Z0[:, k] and
          the gcd mod size of the elements and the array size, needed for computing
          the list of isomorphisms that leaves them invariant.
    """
    #compute possible indices. first is always 0, second is always a divisor, etc
    candidates = [[0] for _ in range(Z_wt)]
    candidates[1] = [i for i in range(size) if i == 0 or size % i == 0]
    for i in range(2, Z_wt):
        candidates[i] = range(size)

    #compute isomorphisms that leave indices unchanged
    isos_by_gcd = [[] for _ in range(size)]
    for i in range(1, size):
        if size % i > 0:
            continue
        isos_by_gcd[i % size] = [j for j in range(1, size) 
                                 if (j - 1) % (size / i) == 0 and gcd(j, size) == 1]

    #check to make sure each candidate is minimal under isomorphisms and append
    result = []
    isos = find_isos(size)
    strides = find_strides([size] * Z_wt)
    for zs in it.product(*candidates):
        z_list = np.array(zs)
        if z_list @ strides == min(
                [np.mod(z_list * i, size) @ strides for i in isos]):
            result.append((z_list, gcd(*zs, size) % size))
    return result

def decomposed_X0_candidates(X_wt, size):
    """
    Finds possible elements of X0[:, k] that are worth searching over for some k.

    Params:
        * X_wt (int): The number of terms in X0
        * size (int): The size of the cyclic group we are finding candidates for
    
    Returns:
        * 3d array. The outer dimension is loops over all values from 0
          to size, inclusive. Only the entries which are factors of size have any
          elements. The second dimension is just listing candidates. The inner
          dimension loops over the terms of X0, and thus has length X_wt.
    """
    #compute possible indices. first is either 0 or 1 if size is even
    candidates = [[0] if size % 2 == 1 else [0, 1] for _ in range(X_wt)]
    for i in range(1, X_wt):
        candidates[i] = range(size)

    #compute isomorphisms that might leave Z indices unchanged
    isos_by_gcd = [[] for _ in range(size)]
    for i in range(1, size + 1):
        if size % i > 0:
            continue
        isos_by_gcd[i % size] = [j for j in range(1, size)
                                 if (j - 1) % (size / i) == 0 and gcd(j, size) == 1]

    #check to make sure each candidate is minimal under isomorphisms and append
    result = [[] for _ in range(size + 1)]
    strides = find_strides([size] * X_wt)
    for i in range(size):
        if len(isos_by_gcd[i]) == 0:
            continue
        for xs in it.product(*candidates):
            x_list = np.array(xs)
            if x_list @ strides == min(
                    [np.mod(x_list * j, size) @ strides for j in isos_by_gcd[i]]):
                result[i].append(x_list)
    return result

def build_Z0_candidates(Z_wt, group, rate_filter = True):
    """
    Finds possible tuples Z0 that are worth searching over.

    Params:
        * Z_wt (int): The number of terms in Z0
        * group (np.ndarray): The group whose codes we are finding
        * rate_filter (bool, optional): Whether codes should be filtered by rate
    
    Returns:
        * List of Z0 candidates with possible isomorphisms. Each candidate is a
          tuple containing (Z0, isomorphisms). Each Z0 is a sorted list/tuple of
          lists/tuples mod group. Each Z0[0] is always all zeros, and each Z0[1]
          only contains terms which divide the group size. Each candidate
          isomorphisms is a index of gcds for each dimension, communicating
          potential isomorphisms which canonicalize Z0 as much as possible.
    """
    n = np.prod(group)
    d = len(group)
    candidates = [decomposed_Z0_candidates(Z_wt, i) for i in group]
    lengths = [len(c) for c in candidates]
    #index incrementing amount for each index
    strides = find_strides(lengths)
    
    gcd_strides = find_strides(group)

    max_index = np.prod(lengths)
    index_val = 0
    result = []
    isos = find_isos(group)
    #for each valid combination from each subgroup
    while index_val < max_index:
        jump = -1
        index = index_to_tuple(lengths, index_val)
        zs = np.zeros((Z_wt, d), dtype = int)
        z_indices = None
        #find first instance of an index sorted out of order...
        for i in range(d):
            zs[:, i] = candidates[i][index[i]][0]
            z_indices = zs @ strides
            if np.any(z_indices[:-1] > z_indices[1:]):
                jump = strides[i]
                break
        #and increment there if it is not canonical
        if (jump > 0 or np.any(z_indices[:-1] >= z_indices[1:])
            or not is_Z_canonical(group, zs, isos)
            or (rate_filter and
                n - compute_rank_from_tuples(group, zs) < n * RATE_THRESHOLD)):
            jump = max(jump, 1)
            index_val = (index_val // jump + 1) * jump
            continue
        #otherwise, write down the code and the gcds
        result.append((zs, int(np.array([candidates[i][index[i]][1]
                                         for i in range(d)]) @ gcd_strides)))
        index_val += 1
    return result

def build_X0_candidates(X_wt, group, rate_filter = True):
    """
    Finds possible tuples X0 that are worth searching over. Split up by the gcd of
    the Zs, which defines isomorphisms under which the Xs are minimal.

    Params:
        * X_wt (int): The number of terms in X0
        * group (np.ndarray): The group whose codes we are finding
        * rate_filter (bool, optional): Whether codes should be filtered by rate
    
    Returns:
        * List of lists of X0 candidates. The outer list is indexed by the gcd's of
          the Zs which determine the isomorphisms under which the Zs are equivalent,
          meaning we must minimize the Xs over this set. Each X0 is a sorted
          list/tuple of lists/tuples mod group. Each X0[0] is always all 0 or 1.
    """
    n = np.prod(group)
    d = len(group)
    candidates = [decomposed_X0_candidates(X_wt, i) for i in group]
    result = [[] for i in range(n)]
    for i in range(n):
        gcds = index_to_tuple(group, i)
        lengths = [len(candidates[j][gcds[j]]) for j in range(d)]
        
        #index incrementing amount for each index
        strides = find_strides(lengths)
        max_index = np.prod(lengths)
        if max_index == 0:
            continue
        isos = [[k for k in range(1, g) if (k - 1) %
                 (1 if gcds[j] == 0 else g / gcds[j]) == 0 and gcd(k, g) == 1]
                for j, g in enumerate(group)]
        index_val = 0
        #for each valid combination from each subgroup
        while index_val < max_index:
            jump = -1
            index = index_to_tuple(lengths, index_val)
            xs = np.zeros((X_wt, d), dtype = int)
            #find first instance of an index sorted out of order...
            for j in range(d):
                xs[:, j] = candidates[j][gcds[j]][index[j]]
                x_indices = xs @ strides
                if np.any(x_indices[:-1] > x_indices[1:]):
                    jump = strides[j]
                    break
            #and increment there if it is not increasing
            if (jump > 0 or np.any(x_indices[:-1] >= x_indices[1:])
                or not is_X_canonical(group, xs, isos)
                or (rate_filter and
                    n - compute_rank_from_tuples(group, xs) < n * RATE_THRESHOLD)):
                jump = max(jump, 1)
                index_val = (index_val // jump + 1) * jump
                continue
            #otherwise, write down the code
            result[i].append(xs)
            index_val += 1
    return result

def find_all_codes_in_group(Z_wt, X_wt, group, rate_filter = True):
    """
    Finds all codes of weights Z_wt and X_wt for a given group.

    Params:
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * group (np.ndarray): The group whose codes we are finding
        * rate_filter (bool, optional): Whether codes should be filtered by rate
    
    Returns:
        * List of tuples of the form (Z0, X0) for valid codes. Each of Z0 and X0
        is a list/tuple of lists/tuples mod group.
    """
    zs = build_Z0_candidates(Z_wt, group)
    xs = build_X0_candidates(X_wt, group)
    total_codes = sum([len(xs[i[1]]) for i in zs])
    codes = [0] * total_codes
    position = 0
    for i in zs:
        for j in xs[i[1]]:
            codes[position] = (i[0], j)
            position += 1
    return codes

def find_all_codes(n, Z_wt, X_wt, rate_filter = True):
    """
    Finds all codes for a given number of qubits, n, of given weight.

    Params:
        * n (int): The number of physical qubits of the desired codes
        * Z_wt (int): The number of terms in Z0
        * X_wt (int): The number of terms in X0
        * rate_filter (bool, optional): Whether codes should be filtered by rate
    
    Returns:
        * List of tuples of the form (group, Z0, X0) for valid codes. Each of Z0
        and X0 is a list/tuple of lists/tuples mod group.
    """
    if n < 2:
        return []
    result = []
    for group in n_partitions(n):
        result += find_all_codes_in_group(Z_wt, X_wt, group)
    return result

def main():
    for i in range(16):
        print(i, process_codes(i, 3, 3))

if __name__ == "__main__":
    main()
