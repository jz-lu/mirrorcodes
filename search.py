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
