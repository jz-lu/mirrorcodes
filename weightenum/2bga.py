"""
weightenum.2bga.py
Code file using the weight enumerator method to show that 2BGA codes are not contained in either type of mirror codes.
Written by ChatGPT under supervision by JZL.
"""
import itertools, random, time
import numpy as np
from dataclasses import dataclass

# -------------------------
# Group infrastructure
# -------------------------
@dataclass(frozen=True)
class Group:
    name: str
    elems: tuple
    mul_table: np.ndarray
    inv_table: np.ndarray
    conj_classes: tuple

def make_group(elems, mul_func, inv_func, name):
    elems = tuple(elems)
    idx = {e:i for i,e in enumerate(elems)}
    n=len(elems)
    mul_table = np.zeros((n,n), dtype=np.int16)
    for i,a in enumerate(elems):
        for j,b in enumerate(elems):
            mul_table[i,j]=idx[mul_func(a,b)]
    inv_table=np.zeros(n,dtype=np.int16)
    for i,a in enumerate(elems):
        inv_table[i]=idx[inv_func(a)]

    # conjugacy classes
    classes=[]
    unassigned=set(range(n))
    for i in range(n):
        if i not in unassigned: 
            continue
        orb=set()
        for g in range(n):
            gi=inv_table[g]
            x = mul_table[mul_table[gi,i], g]
            orb.add(int(x))
        changed=True
        while changed:
            changed=False
            for x in list(orb):
                for g in range(n):
                    gi=inv_table[g]
                    y=mul_table[mul_table[gi,x], g]
                    if int(y) not in orb:
                        orb.add(int(y)); changed=True
        for x in orb:
            unassigned.discard(x)
        classes.append(tuple(sorted(orb)))
    conj_classes=tuple(sorted(classes, key=lambda c:(len(c),c)))
    return Group(name, elems, mul_table, inv_table, conj_classes)

def cyclic(n):
    elems=list(range(n))
    def mul(a,b): return (a+b)%n
    def inv(a): return (-a)%n
    return make_group(elems,mul,inv,f"C{n}")

def direct_product(G:Group,H:Group,name):
    nG=len(G.elems); nH=len(H.elems)
    elems=[(i,j) for i in range(nG) for j in range(nH)]
    def mul(a,b):
        i1,j1=a; i2,j2=b
        return (int(G.mul_table[i1,i2]), int(H.mul_table[j1,j2]))
    def inv(a):
        i,j=a
        return (int(G.inv_table[i]), int(H.inv_table[j]))
    return make_group(elems,mul,inv,name)

def s3():
    elems=list(itertools.permutations([0,1,2]))
    def compose(p,q): 
        return tuple(p[i] for i in q)
    def inv(p):
        r=[0]*3
        for i,v in enumerate(p): r[v]=i
        return tuple(r)
    return make_group(elems,compose,inv,"S3")

def dihedral_order_12():
    n=6
    elems=[(k,b) for k in range(n) for b in (0,1)]
    def mul(x,y):
        k,b=x; l,c=y
        if b==0:
            return ((k+l)%n, c)
        else:
            return ((k-l)%n, b^c)
    def inv(x):
        k,b=x
        if b==0:
            return ((-k)%n,0)
        else:
            return (k,1)
    return make_group(elems,mul,inv,"D12")

def a4():
    elems=[]
    for p in itertools.permutations(range(4)):
        inv_count=0
        for i in range(4):
            for j in range(i+1,4):
                if p[i]>p[j]: inv_count+=1
        if inv_count%2==0:
            elems.append(p)
    def compose(p,q): 
        return tuple(p[i] for i in q)
    def inv(p):
        r=[0]*4
        for i,v in enumerate(p): r[v]=i
        return tuple(r)
    return make_group(elems,compose,inv,"A4")

def dic12():
    elems=[('a',k) for k in range(6)] + [('x',k) for k in range(6)]
    def mul(u,v):
        t1,i=u; t2,j=v
        if t1=='a' and t2=='a':
            return ('a',(i+j)%6)
        if t1=='a' and t2=='x':
            return ('x',(i+j)%6)
        if t1=='x' and t2=='a':
            return ('x',(i-j)%6)
        if t1=='x' and t2=='x':
            return ('a',(i-j+3)%6)
        raise ValueError
    def inv(u):
        t,i=u
        if t=='a':
            return ('a',(-i)%6)
        else:
            return ('x',(i-3)%6)
    return make_group(elems,mul,inv,"Dic12")

# -------------------------
# Translate tables
# -------------------------
def precompute_translate_tables(G:Group):
    n=len(G.elems)
    N=1<<n
    perm_r = np.zeros((n,n), dtype=np.int16)
    perm_l = np.zeros((n,n), dtype=np.int16)
    for g in range(n):
        perm_r[g]=G.mul_table[:,g]
        perm_l[g]=G.mul_table[g,:]
    rt = np.zeros((n,N), dtype=np.uint16)
    lt = np.zeros((n,N), dtype=np.uint16)
    for g in range(n):
        pr = perm_r[g]
        pl = perm_l[g]
        for mask in range(1,N):
            lsb = mask & -mask
            i = (lsb.bit_length()-1)
            prev = mask ^ lsb
            rt[g,mask] = rt[g,prev] ^ (1<<int(pr[i]))
            lt[g,mask] = lt[g,prev] ^ (1<<int(pl[i]))
    inv = G.inv_table.astype(np.int16)
    return rt, lt, inv

parity4096 = np.array([bin(i).count("1") & 1 for i in range(4096)], dtype=np.uint8)
popcount4096 = np.array([bin(i).count("1") for i in range(4096)], dtype=np.uint8)

# -------------------------
# GF(2) basis and weight enumerator
# -------------------------
def gf2_row_basis(rows):
    basis=[]
    for r in rows:
        x=int(r)
        for b in basis:
            lb = b.bit_length()-1
            if (x>>lb)&1:
                x ^= b
        if x==0:
            continue
        lb = x.bit_length()-1
        new=[]
        for b in basis:
            if (b>>lb)&1:
                b ^= x
            new.append(b)
        new.append(x)
        new.sort(reverse=True)
        basis=new
    return basis

def stabilizer_weight_enumerator(gens, n):
    rows=[(x | (z<<n)) for x,z in gens]
    basis = gf2_row_basis(rows)
    r=len(basis)
    counts = np.zeros(n+1, dtype=np.int32)
    bx=[b & ((1<<n)-1) for b in basis]
    bz=[(b>>n) & ((1<<n)-1) for b in basis]
    x=0; z=0
    counts[0]+=1
    prev_gray=0
    for i in range(1, 1<<r):
        gray = i ^ (i>>1)
        diff = gray ^ prev_gray
        j = (diff.bit_length()-1)
        x ^= bx[j]; z ^= bz[j]
        counts[(x|z).bit_count()] += 1
        prev_gray=gray
    return tuple(int(c) for c in counts), r

# -------------------------
# 2BGA generators
# -------------------------
def gens_2bga(G:Group, rt, lt, inv, A_mask, B_mask):
    m=len(G.elems); n=2*m
    Ainv_mask = 0
    Binv_mask = 0
    for i in range(m):
        if (A_mask>>i)&1:
            Ainv_mask |= 1<<int(inv[i])
        if (B_mask>>i)&1:
            Binv_mask |= 1<<int(inv[i])
    gens=[]
    for g in range(m):
        ZL = int(rt[g, A_mask])
        ZR = int(lt[g, B_mask])
        gens.append((0, ZL | (ZR<<m)))
    for g in range(m):
        XL = int(lt[g, Binv_mask])
        XR = int(rt[g, Ainv_mask])
        gens.append((XL | (XR<<m), 0))
    return gens, n

# -------------------------
# Mirror generators
# -------------------------
def gens_mirror_sym(G:Group, rt, inv, A_mask, B_mask):
    n=len(G.elems)
    gens=[]
    for g in range(n):
        Z = int(rt[g, A_mask])
        X = int(rt[int(inv[g]), B_mask])
        gens.append((X,Z))
    return gens, n

def gens_mirror_asym(G:Group, rt, lt, inv, A_mask, B_mask):
    n=len(G.elems)
    gens=[]
    for g in range(n):
        Z = int(rt[g, A_mask])
        X = int(lt[int(inv[g]), B_mask])
        gens.append((X,Z))
    return gens, n

# -------------------------
# Validity check: mirror commutation
# -------------------------
def valid_Bs_mirror(G:Group, rt, lt, inv, A_mask, mirror_type="sym"):
    n=len(G.elems)
    N=1<<n
    Zg = rt[:, A_mask].astype(np.uint16)
    if mirror_type=="sym":
        Xtab = rt[inv.astype(int), :]
    else:
        Xtab = lt[inv.astype(int), :]
    valid = np.ones(N, dtype=bool)
    for g in range(n):
        Zg_g = Zg[g]
        for h in range(g+1,n):
            left = parity4096[(Zg_g & Xtab[h]).astype(np.int32)]
            right = parity4096[(Zg[h] & Xtab[g]).astype(np.int32)]
            valid &= (left == right)
            if not valid.any():
                return valid
    return valid

# -------------------------
# Support signature (strong)
# -------------------------
def signature_supports_strong(supp_list, n):
    supp = np.array(supp_list, dtype=np.uint16)
    gen_w = tuple(sorted(int(popcount4096[s]) for s in supp))
    degs=[]
    for i in range(n):
        degs.append(int(((supp>>i)&1).sum()))
    degs=tuple(sorted(degs))
    inter=[]
    for i in range(n):
        si=int(supp[i])
        for j in range(i+1,n):
            inter.append(int(popcount4096[si & int(supp[j])]))
    inter=tuple(sorted(inter))
    return (gen_w, degs, inter)

# -------------------------
# Target search: mirror codes matching sig AND WE
# -------------------------
def find_mirror_matches_to_target(G:Group, mirror_type:str,
                                 target_sig, target_we, target_rank,
                                 assume_all_commute=False):
    n=len(G.elems)
    rt, lt, inv = precompute_translate_tables(G)
    N=1<<n
    if mirror_type=="sym":
        Xtab = rt[inv.astype(int), :]
    else:
        Xtab = lt[inv.astype(int), :]

    target_genw = np.array(target_sig[0], dtype=np.uint8)
    target_deg  = np.array(target_sig[1], dtype=np.uint8)
    target_inter= np.array(target_sig[2], dtype=np.uint8)

    # cheap hash on (genw,degs)
    p=np.uint64(1315423911)
    powp=np.array([pow(int(p),i,2**64) for i in range(24)], dtype=np.uint64)
    v_target = np.array(list(target_sig[0])+list(target_sig[1]), dtype=np.uint64)
    h_target = int((v_target*powp).sum(dtype=np.uint64))

    sols=[]
    for A in range(N):
        if not assume_all_commute:
            validB = valid_Bs_mirror(G, rt, lt, inv, A, mirror_type)
            if not validB.any():
                continue
            idxs = np.nonzero(validB)[0]
        else:
            idxs = None

        Zvec = rt[:,A].astype(np.uint16)[:,None]
        supp = (Zvec | Xtab).astype(np.uint16)

        if idxs is not None:
            supp_v = supp[:, idxs]
        else:
            supp_v = supp

        W = popcount4096[supp_v]
        W_sorted = np.sort(W, axis=0)

        degrees = np.empty((n, W_sorted.shape[1]), dtype=np.uint8)
        for q in range(n):
            degrees[q] = ((supp_v >> q) & 1).sum(axis=0)
        D_sorted = np.sort(degrees, axis=0)

        h = (W_sorted.astype(np.uint64) * powp[:12,None]).sum(axis=0, dtype=np.uint64) + \
            (D_sorted.astype(np.uint64) * powp[12:,None]).sum(axis=0, dtype=np.uint64)

        cand = np.nonzero(h == h_target)[0]
        if cand.size==0:
            continue

        mask = np.all(W_sorted[:,cand] == target_genw[:,None], axis=0) & \
               np.all(D_sorted[:,cand] == target_deg[:,None], axis=0)
        cand = cand[np.nonzero(mask)[0]]
        if cand.size==0:
            continue

        supp_c = supp_v[:, cand]
        inters = np.empty((n*(n-1)//2, cand.size), dtype=np.uint8)
        k=0
        for i in range(n):
            si = supp_c[i]
            for j in range(i+1,n):
                inters[k] = popcount4096[(si & supp_c[j]).astype(np.int32)]
                k+=1
        I_sorted = np.sort(inters, axis=0)

        mask2 = np.all(I_sorted == target_inter[:,None], axis=0)
        cand2 = cand[np.nonzero(mask2)[0]]

        for col in cand2:
            B = int(col if idxs is None else idxs[col])
            if mirror_type=="sym":
                gens,_ = gens_mirror_sym(G, rt, inv, A, B)
            else:
                gens,_ = gens_mirror_asym(G, rt, lt, inv, A, B)
            we, r = stabilizer_weight_enumerator(gens, n)
            if r == target_rank and we == target_we:
                sols.append((A,B))
    return sols

# -------------------------
# Groups of order 12 (up to isomorphism)
# -------------------------

def is_abelian(G: Group) -> bool:
    return bool(np.all(G.mul_table == G.mul_table.T))

def groups_order_12():
    """
    Returns representatives of all groups of order 12 (up to isomorphism):
      - abelian: C12, C6xC2, C3xC2xC2
      - nonabelian: D12, A4, Dic12
    """
    C12 = cyclic(12)
    C6xC2 = direct_product(cyclic(6), cyclic(2), "C6xC2")
    C3xC2xC2 = direct_product(cyclic(3), direct_product(cyclic(2), cyclic(2), "C2xC2"), "C3xC2xC2")
    D12 = dihedral_order_12()
    A4 = a4()
    Dic12 = dic12()
    return [C12, C6xC2, C3xC2xC2, D12, A4, Dic12]

def find_matches_all_order12_groups(target_sig, target_we, target_rank, mirror_type="sym"):
    """
    For a fixed target (sig,we,rank) on n=12 qubits, search over ALL (G',A',B')
    where |G'|=12, and mirror_type in {"sym","asym"}.
    Returns dict: group_name -> list of (A,B) solutions
    """
    results = {}
    for Gp in groups_order_12():
        assert len(Gp.elems) == 12, f"{Gp.name} not order 12?"
        assume_all = is_abelian(Gp)  # for abelian groups, mirror commutation is automatic
        t0 = time.time()
        sols = find_mirror_matches_to_target(
            Gp,
            "sym" if mirror_type == "sym" else "asym",
            target_sig,
            target_we,
            target_rank,
            assume_all_commute=assume_all
        )
        dt = time.time() - t0
        results[Gp.name] = sols
        print(f"[{mirror_type}] {Gp.name:10s} abelian={assume_all}  matches={len(sols)}  time={dt:.2f}s")
    return results


# -------------------------
# Example: build a target 2BGA on S3, then scan all order-12 groups
# -------------------------

if __name__ == "__main__":
    # Build target 2BGA on S3
    G_s3 = s3()
    rt6, lt6, inv6 = precompute_translate_tables(G_s3)

    # Choose your target (A_t, B_t) in [0..63] bitmasks over S3 (6 elements)
    A_t, B_t = 3, 7

    gens_t, n_t = gens_2bga(G_s3, rt6, lt6, inv6, A_t, B_t)
    assert n_t == 12

    supp_t = [x | z for (x, z) in gens_t]
    sig_t = signature_supports_strong(supp_t, n_t)
    we_t, rank_t = stabilizer_weight_enumerator(gens_t, n_t)

    print("Target 2BGA on S3:")
    print("  A_t =", A_t, "B_t =", B_t)
    print("  rank =", rank_t)
    print("  we (counts by weight 0..12) =", we_t)

    # Scan symmetric/type-1 mirror codes on all groups of order 12
    res_sym = find_matches_all_order12_groups(sig_t, we_t, rank_t, mirror_type="sym")

    # Scan asymmetric/type-2 mirror codes on all groups of order 12
    res_asym = find_matches_all_order12_groups(sig_t, we_t, rank_t, mirror_type="asym")

    # Print a brief summary of which groups had any solutions
    print("\nSummary (groups with >=1 match):")
    print("  sym :", [k for k,v in res_sym.items() if len(v) > 0])
    print("  asym:", [k for k,v in res_asym.items() if len(v) > 0])
