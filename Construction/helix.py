"""
`helix.py`
Code file to construct a helix code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} \oplus ... \oplus Z_{aN}
    * X0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to X0
    * Z0 [{v1, v2, ..., vQ}]: a set of N-tuples specifying elements of the group which belong to Z0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the helix code.
The check weight is exactly |X0| + |Z0| - |X0 \cap Z0| <= |X0| + |Z0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form---there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
"""

