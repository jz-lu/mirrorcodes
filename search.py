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
from helix import find_stabilizers
from distance import distance

def is_canonical(helix_code):
    """
    Check whether `helix_code` is in canonical form.
    We say that a helix code (group, X0, Z0) is in canonical form if TODO
    
    Input:
        * `helix_code` (tuple): Tuple of (group, X0, Z0) where group is a tuple and X0 and Z0 are sets
    
    Returns:
        * Binary indication of whether `helix_code` is in canonical form.
    """
    pass

def passes_stage_1(helix_code):
    """
    Evaluate whether a given code passes through stage 1 of tests.

    Input:
        * `helix_code` (tuple): Tuple of (group, X0, Z0) where group is a tuple and X0 and Z0 are sets
    
    Returns:
        * Binary indication of whether `helix_code` is in canonical form.
    """
    pass

def group_level_search(group, X_wt, Z_wt):
    """
    Search helix codes formed from `group` with X-weight `X_wt` and Z-weight `Z_wt`.
    Save the ones which pass stage 1.

    Input:
        * group (tuple): tuple of integers specifying the group
        * X_wt (int): weight of the X's in each check
        * Z_wt (int): weight of the Z's in each check
    
    Output:
        * A .npy file for each code which passes the stage 1 test
    
    Returns:
        * None
    """
    pass

def n_level_search(n, X_wt, Z_wt):
    """
    Search helix codes over `n` qubits with X-weight `X_wt` and Z-weight `Z_wt`.
    Save the ones which pass stage 1.

    Input:
        * n (int): number of qubits
        * X_wt (int): weight of the X's in each check
        * Z_wt (int): weight of the Z's in each check
    
    Output:
        * A .npy file for each code which passes the stage 1 test
    
    Returns:
        * None
    """
    pass


def main():
    pass

if __name__ == "__main__":
    main()