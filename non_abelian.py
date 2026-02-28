import os
import subprocess
import tempfile
import pathlib
import util
from mirror import MirrorCode, valid_non_abelian
import itertools as it

import ast
import textwrap
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class IndexedGroupOps:
    n: int
    i: int
    description: str
    inv_table: List[int]                # inv_table[a] = a^{-1} (0-based)
    mul_table: List[List[int]]          # mul_table[a][b] = a*b (0-based)
    gap_bat: str
    timeout: int = 120

    _center_cache: Optional[List[int]] = None
    _aut_cache: Optional[List[List[int]]] = None

    # ---- core ops ----
    def inv(self, a: int) -> int:
        if not (0 <= a < self.n):
            raise IndexError(f"index out of range: {a}")
        return self.inv_table[a]

    def mul(self, *idxs: int) -> int:
        if len(idxs) < 2:
            raise ValueError("mul() needs at least two indices")
        acc = idxs[0]
        if not (0 <= acc < self.n):
            raise IndexError(f"index out of range: {acc}")
        for x in idxs[1:]:
            if not (0 <= x < self.n):
                raise IndexError(f"index out of range: {x}")
            acc = self.mul_table[acc][x]
        return acc

    # ---- center (computed from mul_table; cached) ----
    def center(self) -> List[int]:
        if self._center_cache is not None:
            return self._center_cache

        # Z(G) = {a : ab = ba for all b}
        n = self.n
        mul = self.mul_table
        Z = []
        for a in range(n):
            row_a = mul[a]
            ok = True
            for b in range(n):
                if row_a[b] != mul[b][a]:
                    ok = False
                    break
            if ok:
                Z.append(a)

        self._center_cache = Z
        return Z

    # ---- automorphisms (computed via GAP once; cached) ----
    def automorphisms(self) -> List[List[int]]:
        """
        Returns all automorphisms as index maps [phi(0), phi(1), ..., phi(n-1)].

        WARNING: |Aut(G)| can be very large even for n <= 300. This will materialize
        every automorphism into memory.
        """
        if self._aut_cache is not None:
            return self._aut_cache

        n, i = self.n, self.i

        gap_code = textwrap.dedent(f"""
            if LoadPackage("smallgrp") = fail then
                Print("ERR\\tNO_SMALLGRP\\n");
            fi;

            G := SmallGroup({n}, {i});

            # Use a stable element order; then force identity to index 0.
            elts := AsSortedList(G);
            e := Identity(G);
            ord := Concatenation([e], Filtered(elts, x -> x <> e));
            len := Length(ord);

            A := AutomorphismGroup(G);
            auts := Elements(A);

            # Emit each automorphism as a 0-based index list of images
            for phi in auts do
                imgidx := List(ord, x -> Position(ord, Image(phi, x)) - 1);
                Print("AUT\\t", imgidx, "\\n");
            od;

        """)

        out, err = _run_gap_file_via_runtime_bash(gap_code, gap_bat=self.gap_bat, timeout=self.timeout)
        out = out.replace(" \n ", "")
        auts: List[List[int]] = []
        for line in out.splitlines():
            if line.startswith("ERR\t"):
                _, msg = line.split("\t", 1)
                raise RuntimeError(f"GAP error: {msg}")
            if line.startswith("AUT\t"):
                _, payload = line.split("\t", 1)
                phi = ast.literal_eval(payload)
                # basic sanity: must be a permutation of 0..n-1 and fix identity
                if len(phi) != n:
                    raise RuntimeError(f"Bad automorphism length {len(phi)} (expected {n}).")
                if phi[0] != 0:
                    raise RuntimeError("Automorphism does not fix identity index 0 (indexing mismatch).")
                auts.append(phi)

        if not auts:
            raise RuntimeError(
                "Received no AUT lines from GAP. "
                "If you see a GAP prompt/banner in output, GAP did not run headless."
            )

        self._aut_cache = auts
        return auts


def build_indexed_group_ops(group_dict, timeout: int = 120) -> IndexedGroupOps:
    """
    Build inv/mul tables using GAP, with indices 0..n-1 and identity at 0.
    """
    gap_bat = util.gap_bat()
    if "n" in group_dict:
        n = int(group_dict["n"])
        i = int(group_dict["i"])
        desc = str(group_dict.get("description", ""))
    else:
        n = group_dict[0]
        i = group_dict[1]
        desc = group_dict[2]

    gap_code = textwrap.dedent(f"""
        if LoadPackage("smallgrp") = fail then
            Print("ERR\\tNO_SMALLGRP\\n");
        fi;

        G := SmallGroup({n}, {i});

        # Stable order, then force identity to 0
        elts := AsSortedList(G);
        e := Identity(G);
        ord := Concatenation([e], Filtered(elts, x -> x <> e));
        len := Length(ord);

        invs := List([1..len], a -> Position(ord, ord[a]^-1) - 1);
        Print("INV\\t", invs, "\\n");

        for a in [1..len] do
            row := List([1..len], b -> Position(ord, ord[a]*ord[b]) - 1);
            Print("ROW\\t", a-1, "\\t", row, "\\n");
        od;

    """)

    out, err = _run_gap_file_via_runtime_bash(gap_code, gap_bat=gap_bat, timeout=timeout)

    inv_table = None
    mul_table: List[Optional[List[int]]] = [None] * n

    out = out.replace(" \n ", "")
    for line in out.splitlines():
        if line.startswith("ERR\t"):
            _, msg = line.split("\t", 1)
            raise RuntimeError(f"GAP error: {msg}")
        if line.startswith("INV\t"):
            _, payload = line.split("\t", 1)
            inv_table = ast.literal_eval(payload)
        elif line.startswith("ROW\t"):
            _, r_str, payload = line.split("\t", 2)
            r = int(r_str)
            mul_table[r] = ast.literal_eval(payload)

    if inv_table is None:
        raise RuntimeError(f"Did not receive INV table from GAP. Output was:\n{out}")

    if any(row is None for row in mul_table):
        missing = [k for k, row in enumerate(mul_table) if row is None][:10]
        raise RuntimeError(f"Missing multiplication rows from GAP (first missing rows: {missing}).")

    mul_table_final: List[List[int]] = mul_table  # type: ignore

    # sanity checks
    if len(inv_table) != n:
        raise RuntimeError(f"INV length {len(inv_table)} != n={n}")
    if inv_table[0] != 0:
        raise RuntimeError("Index 0 should be identity and self-inverse; indexing mismatch.")
    if mul_table_final[0][0] != 0:
        raise RuntimeError("0*0 should be 0; indexing mismatch.")
    for a in range(n):
        if mul_table_final[0][a] != a or mul_table_final[a][0] != a:
            raise RuntimeError("0 should act as identity on all elements; indexing mismatch.")

    return IndexedGroupOps(
        n=n,
        i=i,
        description=desc,
        inv_table=inv_table,
        mul_table=mul_table_final,
        gap_bat=gap_bat,
        timeout=timeout,
    )

CREATE_NO_WINDOW = 0x08000000

def win_to_cygdrive(path: str) -> str:
    p = os.path.abspath(path)
    drive = p[0].lower()
    rest = p[2:].replace("\\", "/")
    return f"/cygdrive/{drive}{rest}"

def _run_gap_file_via_runtime_bash(code: str, *, gap_bat: str, timeout=120):
    gap_root = os.path.dirname(os.path.abspath(gap_bat))
    runtime_bin = os.path.join(gap_root, "runtime", "bin")
    bash_exe = os.path.join(runtime_bin, "bash.exe")
    if not os.path.exists(bash_exe):
        raise FileNotFoundError(f"Could not find bash.exe at: {bash_exe}")


    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "script.g"
        script.write_text(code, encoding="utf-8")
        script_cyg = win_to_cygdrive(str(script))

        # IMPORTANT: bypass /run-gap.sh and call gap directly
        # bash --login -lc 'gap -q -T "<script>"'
        bash_cmd = f'gap -q -T "{script_cyg}"'
        cmd = [bash_exe, "--login", "-lc", bash_cmd]

        env = os.environ.copy()
        env["PATH"] = runtime_bin + os.pathsep + env.get("PATH", "")

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            creationflags=CREATE_NO_WINDOW,
        )

    out = p.stdout.decode("utf-8", errors="replace")
    err = p.stderr.decode("utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"GAP failed (exit {p.returncode}). STDERR:\n{err}\nSTDOUT:\n{out}")
    return out, err


def nonabelian_groups_of_order(n: int, timeout=120):
    gap_bat = util.gap_bat()
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    gap_code = textwrap.dedent(f"""
        Print("DBG:START\\n");

        lp := LoadPackage("smallgrp");
        Print("DBG:LoadPackage=", lp, "\\n");
        if lp = fail then
            Print("NO_SMALLGRP\\n");
        fi;

        av := SmallGroupsAvailable({n});
        Print("DBG:Available=", av, "\\n");
        if not av then
            Print("NOT_AVAILABLE\\n");
        fi;

        m := NumberSmallGroups({n});
        Print("DBG:NumberSmallGroups=", m, "\\n");

        cnt := 0;
        for i in [1..m] do
            G := SmallGroup({n}, i);
            if not IsAbelian(G) then
                cnt := cnt + 1;
                Print("HIT\\t", i, "\\t", StructureDescription(G), "\\n");
            fi;
        od;
        Print("DBG:CountNonAbelian=", cnt, "\\n");
        Print("DBG:END\\n");
    """)

    out, err = _run_gap_file_via_runtime_bash(gap_code, gap_bat=gap_bat, timeout=timeout)
    if out.strip() == "":
        raise RuntimeError(f"GAP stdout empty. STDERR was:\n{err}")


    if out == "NO_SMALLGRP":
        raise RuntimeError('GAP package "smallgrp" is not available to this GAP install.')
    if out == "NOT_AVAILABLE":
        raise ValueError(f"SmallGroups library does not provide a full list for order {n}.")
    if out == "":
        raise RuntimeError("GAP produced no output. This usually means the runner still detached or failed early.")

    res = []
    for line in out.splitlines():
        if line.startswith("HIT\t"):
            _, i_str, desc = line.split("\t", 2)
            res.append({"n": n, "i": int(i_str), "description": desc})
    return res

def _sort3(a, b, c):
    # small sorting network (faster than sorted([...]) for tiny fixed size)
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return (a, b, c)

def _norm0_pair(x, y):
    # canonical tuple for a 3-set known to contain identity 0: (0, min(x,y), max(x,y))
    return (0, x, y) if x < y else (0, y, x)

def find_all_non_abelian_codes(n, wz, wx, min_k=2):
    groups = nonabelian_groups_of_order(n)
    result = []

    # Precompute all B-combinations once (huge generator overhead otherwise)
    # Memory is OK for n~100, wx~3 (~161700 tuples).
    btuples_all = list(it.combinations(range(n), wx))

    # Localize globals used in the innermost loops
    MirrorCode_ = MirrorCode
    valid_non_abelian_ = valid_non_abelian

    for g in groups:
        group = build_indexed_group_ops(g)

        mul = group.mul_table
        inv = group.inv_table

        auts = group.automorphisms()
        Z = group.center()
        Zinv = [inv[c] for c in Z]

        # Specialize for the common case wz=3, wx=3 (your stated regime)
        wz3 = (wz == 3)
        wx3 = (wx == 3)
        eqw = (wz == wx)

        # A-sets always contain 0 in your code
        for restofa in it.combinations(range(1, n), wz - 1):
            if wz3:
                a1, a2 = restofa
                aset = (0, a1, a2)  # already sorted
            else:
                aset = tuple(sorted((0, *restofa)))

            symm_viable = True
            asym_viable = True

            # For the B-loop, we can precompute stabilizers that keep A fixed
            # under the candidate1a == aset condition.
            symm_stabs = []  # list of (phi, cinv, u)
            asym_stabs = []  # list of (phi, c, vinv)  (note: vinv per your code)

            # When wz==wx we still need the "swap-order" candidate2 checks;
            # precompute per-(phi,c) data that depends only on A.
            symm_phi_c_data = []  # (phi, cinv, ac0, ac1, ac2) if wz3 else (phi, cinv, ac_list)
            asym_phi_c_data = []  # (phi, c, cpa0, cpa1, cpa2) if wz3 else (phi, c, cpa_list)

            # ---- First phase: viability of A and build stabilizers ----
            for phi in auts:
                if not symm_viable and not asym_viable:
                    break

                if wz3:
                    p1 = phi[a1]
                    p2 = phi[a2]
                else:
                    # only need phi[a] for a in aset
                    pA = [phi[a] for a in aset]

                for c, cinv in zip(Z, Zinv):
                    if not symm_viable and not asym_viable:
                        break

                    # ---- symmetric viability (and stabilizers) ----
                    if symm_viable:
                        if wz3:
                            # phi_a_c = [phi[0]*c, phi[a1]*c, phi[a2]*c] = [c, p1*c, p2*c]
                            ac0 = c
                            ac1 = mul[p1][c]
                            ac2 = mul[p2][c]

                            if eqw:
                                symm_phi_c_data.append((phi, cinv, ac0, ac1, ac2))

                            # normalize by each base element (3 options)
                            # base = ac0
                            u = inv[ac0]
                            x = mul[u][ac1]; y = mul[u][ac2]
                            cand = _norm0_pair(x, y)
                            if cand < aset:
                                symm_viable = False
                            elif cand == aset:
                                symm_stabs.append((phi, cinv, u))

                            if symm_viable:
                                # base = ac1
                                u = inv[ac1]
                                x = mul[u][ac0]; y = mul[u][ac2]
                                cand = _norm0_pair(x, y)
                                if cand < aset:
                                    symm_viable = False
                                elif cand == aset:
                                    symm_stabs.append((phi, cinv, u))

                            if symm_viable:
                                # base = ac2
                                u = inv[ac2]
                                x = mul[u][ac0]; y = mul[u][ac1]
                                cand = _norm0_pair(x, y)
                                if cand < aset:
                                    symm_viable = False
                                elif cand == aset:
                                    symm_stabs.append((phi, cinv, u))

                        else:
                            # generic small-w fallback
                            ac = [mul[p][c] for p in pA]
                            if eqw:
                                symm_phi_c_data.append((phi, cinv, ac))
                            for base in ac:
                                u = inv[base]
                                cand = tuple(sorted(mul[u][x] for x in ac))
                                if cand < aset:
                                    symm_viable = False
                                    break
                                if cand == aset:
                                    symm_stabs.append((phi, cinv, u))
                            # end generic

                    # ---- asymmetric viability (and stabilizers) ----
                    if asym_viable:
                        if wz3:
                            # c_phi_a = [c*phi[0], c*phi[a1], c*phi[a2]] = [c, c*p1, c*p2]
                            cpa0 = c
                            cpa1 = mul[c][p1]
                            cpa2 = mul[c][p2]

                            if eqw:
                                asym_phi_c_data.append((phi, c, cpa0, cpa1, cpa2))

                            vinv = cpa0
                            v = inv[vinv]
                            x = mul[cpa1][v]; y = mul[cpa2][v]
                            cand = _norm0_pair(x, y)
                            if cand < aset:
                                asym_viable = False
                            elif cand == aset:
                                asym_stabs.append((phi, c, vinv))

                            if asym_viable:
                                vinv = cpa1
                                v = inv[vinv]
                                x = mul[cpa0][v]; y = mul[cpa2][v]
                                cand = _norm0_pair(x, y)
                                if cand < aset:
                                    asym_viable = False
                                elif cand == aset:
                                    asym_stabs.append((phi, c, vinv))

                            if asym_viable:
                                vinv = cpa2
                                v = inv[vinv]
                                x = mul[cpa0][v]; y = mul[cpa1][v]
                                cand = _norm0_pair(x, y)
                                if cand < aset:
                                    asym_viable = False
                                elif cand == aset:
                                    asym_stabs.append((phi, c, vinv))

                        else:
                            cpa = [mul[c][p] for p in pA]
                            if eqw:
                                asym_phi_c_data.append((phi, c, cpa))
                            for vinv in cpa:
                                v = inv[vinv]
                                cand = tuple(sorted(mul[x][v] for x in cpa))
                                if cand < aset:
                                    asym_viable = False
                                    break
                                if cand == aset:
                                    asym_stabs.append((phi, c, vinv))
                            # end generic

            if not symm_viable and not asym_viable:
                continue

            # ---- Second phase: iterate B-sets, using stabilizers ----
            for btuple in btuples_all:
                # btuple is already sorted from combinations()
                bset = btuple

                bsym = symm_viable
                basm = asym_viable

                # --- symmetric B test: candidate1 checks using stabilizer list only ---
                if bsym:
                    if wx3:
                        b0, b1, b2 = bset
                        for phi, cinv, u in symm_stabs:
                            t0 = mul[u][mul[phi[b0]][cinv]]
                            t1 = mul[u][mul[phi[b1]][cinv]]
                            t2 = mul[u][mul[phi[b2]][cinv]]
                            candb = _sort3(t0, t1, t2)
                            if candb < bset:
                                bsym = False
                                break
                    else:
                        for phi, cinv, u in symm_stabs:
                            candb = tuple(sorted(mul[u][mul[phi[b]][cinv]] for b in bset))
                            if candb < bset:
                                bsym = False
                                break

                # --- asymmetric B test: candidate1 checks using stabilizer list only ---
                if basm:
                    if wx3:
                        b0, b1, b2 = bset
                        for phi, c, vinv in asym_stabs:
                            t0 = mul[mul[c][phi[b0]]][vinv]
                            t1 = mul[mul[c][phi[b1]]][vinv]
                            t2 = mul[mul[c][phi[b2]]][vinv]
                            candb = _sort3(t0, t1, t2)
                            if candb < bset:
                                basm = False
                                break
                    else:
                        for phi, c, vinv in asym_stabs:
                            candb = tuple(sorted(mul[mul[c][phi[b]]][vinv] for b in bset))
                            if candb < bset:
                                basm = False
                                break

                if not bsym and not basm:
                    continue

                # --- wz == wx “swap-order” checks (candidate2 blocks) ---
                # These are still expensive, but we avoid recomputing A-dependent parts.
                if eqw and (bsym or basm):
                    if bsym:
                        if wz3 and wx3:
                            b0, b1, b2 = bset
                            for phi, cinv, ac0, ac1, ac2 in symm_phi_c_data:
                                tb0 = mul[phi[b0]][cinv]
                                tb1 = mul[phi[b1]][cinv]
                                tb2 = mul[phi[b2]][cinv]

                                # base = tb0
                                u = inv[tb0]
                                cb = _norm0_pair(mul[u][tb1], mul[u][tb2])
                                if cb < aset:
                                    bsym = False; break
                                if cb == aset:
                                    ca = _sort3(mul[u][ac0], mul[u][ac1], mul[u][ac2])
                                    if ca < bset:
                                        bsym = False; break

                                # base = tb1
                                u = inv[tb1]
                                cb = _norm0_pair(mul[u][tb0], mul[u][tb2])
                                if cb < aset:
                                    bsym = False; break
                                if cb == aset:
                                    ca = _sort3(mul[u][ac0], mul[u][ac1], mul[u][ac2])
                                    if ca < bset:
                                        bsym = False; break

                                # base = tb2
                                u = inv[tb2]
                                cb = _norm0_pair(mul[u][tb0], mul[u][tb1])
                                if cb < aset:
                                    bsym = False; break
                                if cb == aset:
                                    ca = _sort3(mul[u][ac0], mul[u][ac1], mul[u][ac2])
                                    if ca < bset:
                                        bsym = False; break
                            # end for symm_phi_c_data
                        else:
                            # generic fallback
                            for phi, cinv, ac in symm_phi_c_data:
                                tb = [mul[phi[b]][cinv] for b in bset]
                                for base in tb:
                                    u = inv[base]
                                    cb = tuple(sorted(mul[u][x] for x in tb))
                                    if cb < aset:
                                        bsym = False; break
                                    if cb == aset:
                                        ca = tuple(sorted(mul[u][x] for x in ac))
                                        if ca < bset:
                                            bsym = False; break
                                if not bsym:
                                    break

                    if basm:
                        if wz3 and wx3:
                            b0, b1, b2 = bset
                            for phi, c, cpa0, cpa1, cpa2 in asym_phi_c_data:
                                tb0 = mul[c][phi[b0]]
                                tb1 = mul[c][phi[b1]]
                                tb2 = mul[c][phi[b2]]

                                # base = tb0
                                vinv = inv[tb0]
                                cb = _norm0_pair(mul[tb1][vinv], mul[tb2][vinv])
                                if cb < aset:
                                    basm = False; break
                                if cb == aset:
                                    ca = _sort3(mul[cpa0][tb0], mul[cpa1][tb0], mul[cpa2][tb0])
                                    if ca < bset:
                                        basm = False; break

                                # base = tb1
                                vinv = inv[tb1]
                                cb = _norm0_pair(mul[tb0][vinv], mul[tb2][vinv])
                                if cb < aset:
                                    basm = False; break
                                if cb == aset:
                                    ca = _sort3(mul[cpa0][tb1], mul[cpa1][tb1], mul[cpa2][tb1])
                                    if ca < bset:
                                        basm = False; break

                                # base = tb2
                                vinv = inv[tb2]
                                cb = _norm0_pair(mul[tb0][vinv], mul[tb1][vinv])
                                if cb < aset:
                                    basm = False; break
                                if cb == aset:
                                    ca = _sort3(mul[cpa0][tb2], mul[cpa1][tb2], mul[cpa2][tb2])
                                    if ca < bset:
                                        basm = False; break
                            # end for asym_phi_c_data
                        else:
                            for phi, c, cpa in asym_phi_c_data:
                                tb = [mul[c][phi[b]] for b in bset]
                                for base in tb:
                                    vinv = inv[base]
                                    cb = tuple(sorted(mul[x][vinv] for x in tb))
                                    if cb < aset:
                                        basm = False; break
                                    if cb == aset:
                                        ca = tuple(sorted(mul[x][base] for x in cpa))
                                        if ca < bset:
                                            basm = False; break
                                if not basm:
                                    break

                if bsym:
                    code = MirrorCode_(group, list(aset), list(bset), abelian=False, symmetric=True)
                    if valid_non_abelian_(code) and code.get_k() >= min_k:
                        result.append(code)

                if basm:
                    code = MirrorCode_(group, list(aset), list(bset), abelian=False, symmetric=False)
                    if valid_non_abelian_(code) and code.get_k() >= min_k:
                        result.append(code)

    return result

if __name__ == "__main__":
    # for i in range(1, 25):
    #     print(i, len(find_all_non_abelian_codes(i, 3, 3)))
    code = find_all_non_abelian_codes(18, 3, 3)[17]
    print(code.z0, code.x0, code.get_n(), code.get_k(), code.get_d())