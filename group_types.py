import itertools
import numpy as np
import collections
import math

# ----------------------------
# Basic group infrastructure
# ----------------------------
class Group:
    def __init__(self, name, elems, mul, inv):
        self.name = name
        self.elems = list(elems)
        self.mul = mul
        self.inv = inv
        self.index = {g: i for i, g in enumerate(self.elems)}
        self.n = len(self.elems)

    def right(self, S, g):  # S g
        return frozenset(self.mul(s, g) for s in S)

    def left(self, g, S):   # g S
        return frozenset(self.mul(g, s) for s in S)

def C6():
    elems = list(range(6))
    return Group(
        "C6",
        elems,
        lambda a, b: (a + b) % 6,
        lambda a: (-a) % 6
    )

def S3():
    elems = list(itertools.permutations([0, 1, 2]))

    def mul(a, b):  # composition a ∘ b: apply b then a
        return tuple(a[i] for i in b)

    def inv(p):
        q = [0] * 3
        for i, v in enumerate(p):
            q[v] = i
        return tuple(q)

    return Group("S3", elems, mul, inv)

Gs = [C6(), S3()]

# ----------------------------
# Pretty printing for S3
# ----------------------------
def perm_to_cycles(p):
    # p is a tuple mapping i -> p[i]
    n = len(p)
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i] or p[i] == i:
            seen[i] = True
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j + 1)
            j = p[j]
        if len(cyc) > 1:
            cycles.append("(" + "".join(map(str, cyc)) + ")")
    return "e" if not cycles else "".join(cycles)

def elem_str(G, g):
    if G.name == "S3":
        return perm_to_cycles(g)
    return str(g)

def subset_str(G, S):
    return "{" + ", ".join(elem_str(G, g) for g in sorted(S, key=lambda x: str(x))) + "}"

# ----------------------------
# Mirror code generators (Type 1 / Type 2)
# Represent Paulis by symplectic vectors over F2: (z | x) in F2^(2n)
# ----------------------------
def gen_vec(G, A, B, g, typ):
    n = G.n
    z = np.zeros(n, dtype=np.uint8)
    x = np.zeros(n, dtype=np.uint8)

    if typ == 1:
        Zset = G.right(A, g)             # A g
        Xset = G.right(B, G.inv(g))      # B g^{-1}
    elif typ == 2:
        Zset = G.right(A, g)             # A g
        Xset = G.left(G.inv(g), B)       # g^{-1} B
    else:
        raise ValueError("typ must be 1 or 2")

    for q in Zset:
        z[G.index[q]] = 1
    for q in Xset:
        x[G.index[q]] = 1

    return np.concatenate([z, x])

def symp_comm(v, w, n):
    z1, x1 = v[:n], v[n:]
    z2, x2 = w[:n], w[n:]
    return int(((z1 & x2).sum() + (x1 & z2).sum()) % 2) == 0

def commutes_all(G, A, B, typ):
    gens = [gen_vec(G, A, B, g, typ) for g in G.elems]
    n = G.n
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            if not symp_comm(gens[i], gens[j], n):
                return False
    return True

# ----------------------------
# Stabilizer group rank + support-weight enumerator
# ----------------------------
def gf2_rref_rows(rows):
    """Return independent rows (RREF-like) and rank over F2."""
    if not rows:
        return [], 0
    M = [row.copy() for row in rows]
    m = len(M)
    N = len(M[0])
    r = 0
    col = 0
    while r < m and col < N:
        piv = None
        for i in range(r, m):
            if M[i][col]:
                piv = i
                break
        if piv is None:
            col += 1
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(m):
            if i != r and M[i][col]:
                M[i] ^= M[r]
        r += 1
        col += 1
    basis = [M[i] for i in range(r)]
    return basis, r

def weight(v, n):
    z, x = v[:n], v[n:]
    return int((z | x).sum())  # support size, counts Y once

def weight_enumerator_from_basis(basis, n):
    r = len(basis)
    counts = [0] * (n + 1)
    counts[0] = 1  # identity
    for mask in range(1, 1 << r):
        v = np.zeros(2 * n, dtype=np.uint8)
        mm = mask
        i = 0
        while mm:
            if mm & 1:
                v ^= basis[i]
            mm >>= 1
            i += 1
        counts[weight(v, n)] += 1
    return tuple(counts)

def invariants(G, A, B, typ):
    gens = [gen_vec(G, A, B, g, typ) for g in G.elems]
    basis, rank = gf2_rref_rows(gens)
    enum = weight_enumerator_from_basis(basis, G.n)
    # sanity: size should be 2^rank
    assert sum(enum) == (1 << rank)
    return (enum, rank)

# ----------------------------
# Enumerate all A,B subsets for a group
# ----------------------------
def subsets_by_mask(elems):
    elems = list(elems)
    n = len(elems)
    for mask in range(1 << n):
        S = frozenset(elems[i] for i in range(n) if (mask >> i) & 1)
        yield mask, S

def enumerate_type_invariants(typ):
    inv_set = set()
    example = {}  # invariant -> (Gname, Amask, Bmask)
    for G in Gs:
        subsets = list(subsets_by_mask(G.elems))
        for Am, A in subsets:
            for Bm, B in subsets:
                if commutes_all(G, A, B, typ):
                    inv = invariants(G, A, B, typ)
                    inv_set.add(inv)
                    if inv not in example:
                        example[inv] = (G.name, Am, Bm)
    return inv_set, example

def mask_to_subset(G, mask):
    return frozenset(G.elems[i] for i in range(G.n) if (mask >> i) & 1)

def decode_example(Gname, Am, Bm):
    G = next(g for g in Gs if g.name == Gname)
    A = mask_to_subset(G, Am)
    B = mask_to_subset(G, Bm)
    return G, A, B

# ----------------------------
# Main: compute sets and find witnesses
# ----------------------------
t1_set, t1_ex = enumerate_type_invariants(1)
t2_set, t2_ex = enumerate_type_invariants(2)

only1 = list(t1_set - t2_set)
only2 = list(t2_set - t1_set)

print("Distinct (enum,rank) invariants realized:")
print("  Type 1:", len(t1_set))
print("  Type 2:", len(t2_set))
print("  Type 1 but not Type 2:", len(only1))
print("  Type 2 but not Type 1:", len(only2))
print()

# Choose a Type-1 witness where Type2 does NOT commute for the same (A,B), to reduce confusion
w1 = None
for inv in only1:
    Gname, Am, Bm = t1_ex[inv]
    G, A, B = decode_example(Gname, Am, Bm)
    if commutes_all(G, A, B, 1) and not commutes_all(G, A, B, 2):
        w1 = (inv, G, A, B)
        break

# Choose a Type-2 witness where Type1 does NOT commute for the same (A,B), to reduce confusion
w2 = None
for inv in only2:
    Gname, Am, Bm = t2_ex[inv]
    G, A, B = decode_example(Gname, Am, Bm)
    if commutes_all(G, A, B, 2) and not commutes_all(G, A, B, 1):
        w2 = (inv, G, A, B)
        break

assert w1 is not None, "Did not find a clean Type-1 witness with Type-2 noncommuting."
assert w2 is not None, "Did not find a clean Type-2 witness with Type-1 noncommuting."

(inv1, G1, A1, B1) = w1
(inv2, G2, A2, B2) = w2

print("WITNESS 1 (Type 1 but not Type 2):")
print("  Group:", G1.name)
print("  A =", subset_str(G1, A1))
print("  B =", subset_str(G1, B1))
print("  Type1 commutes?", commutes_all(G1, A1, B1, 1))
print("  Type2 commutes?", commutes_all(G1, A1, B1, 2))
print("  Type1 invariant (enum,rank) =", inv1)
print("  Verified: inv1 not in Type2-set?", inv1 not in t2_set)
print()

print("WITNESS 2 (Type 2 but not Type 1):")
print("  Group:", G2.name)
print("  A =", subset_str(G2, A2))
print("  B =", subset_str(G2, B2))
print("  Type2 commutes?", commutes_all(G2, A2, B2, 2))
print("  Type1 commutes?", commutes_all(G2, A2, B2, 1))
print("  Type2 invariant (enum,rank) =", inv2)
print("  Verified: inv2 not in Type1-set?", inv2 not in t1_set)
print()

print("DONE: Incomparability proven by LC+permutation invariant mismatch.")
