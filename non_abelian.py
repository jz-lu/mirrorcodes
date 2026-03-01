import os
import subprocess
import tempfile
import pathlib
import util
import shutil
import sys

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

        out, err = _run_gap(gap_code, gap_bat=self.gap_bat, timeout=self.timeout)
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


def build_indexed_group_ops(group_dict, timeout: int = 120):
    """
    Build inv/mul tables using GAP, with indices 0..n-1 and identity at 0.
    """
    gap_bat = _safe_gap_bat_path()
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

    out, err = _run_gap(gap_code, gap_bat=gap_bat, timeout=timeout)

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

def _safe_gap_bat_path() -> str:
    try:
        p = util.gap_bat()
        return p if isinstance(p, str) else ""
    except Exception:
        return ""

def _is_windows_gap(gap_bat: str) -> bool:
    return bool(gap_bat) and os.path.exists(gap_bat)

def win_to_cygdrive(path: str) -> str:
    p = os.path.abspath(path)
    drive = p[0].lower()
    rest = p[2:].replace("\\", "/")
    return f"/cygdrive/{drive}{rest}"

def _run_gap_windows(code: str, *, gap_bat: str, timeout=120):
    # Your existing Windows GAP bundle uses Cygwin runtime\bin\bash.exe
    gap_root = os.path.dirname(os.path.abspath(gap_bat))
    runtime_bin = os.path.join(gap_root, "runtime", "bin")
    bash_exe = os.path.join(runtime_bin, "bash.exe")
    if not os.path.exists(bash_exe):
        raise FileNotFoundError(f"Could not find bash.exe at: {bash_exe}")

    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "script.g"
        script.write_text(code, encoding="utf-8")
        script_cyg = win_to_cygdrive(str(script))

        # Use -q (quiet), -b (no banner), -T (no break loop on errors)
        bash_cmd = f'gap -q -b -T "{script_cyg}"'
        cmd = [bash_exe, "--login", "-lc", bash_cmd]

        env = os.environ.copy()
        env["PATH"] = runtime_bin + os.pathsep + env.get("PATH", "")

        # Force non-interactive behavior
        p = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
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

def _run_gap_linux(code: str, *, timeout=120):
    # Prefer GAP_EXE if provided, else find gap on PATH
    gap_exe = os.environ.get("GAP_EXE") or shutil.which("gap")
    if not gap_exe:
        raise FileNotFoundError(
            "Could not find GAP executable. On a cluster, load a module (e.g. `module load gap`), "
            "or set env var GAP_EXE=/full/path/to/gap."
        )

    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "script.g"
        script.write_text(code, encoding="utf-8")

        cmd = [gap_exe, "-q", "-b", "-T", str(script)]

        p = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,   # critical to avoid dropping into an interactive prompt
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    out = p.stdout.decode("utf-8", errors="replace")
    err = p.stderr.decode("utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"GAP failed (exit {p.returncode}). STDERR:\n{err}\nSTDOUT:\n{out}")
    return out, err

def _run_gap(code: str, *, gap_bat: str, timeout=120):
    if _is_windows_gap(gap_bat):
        return _run_gap_windows(code, gap_bat=gap_bat, timeout=timeout)
    else:
        return _run_gap_linux(code, timeout=timeout)

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

    out, err = _run_gap(gap_code, gap_bat=gap_bat, timeout=timeout)
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

if __name__ == "__main__":
    for i in [48, 64, 96, 144, 162, 168, 196]:
        print(f"There are {len(nonabelian_groups_of_order(i))} non-abelian groups of size {i}")
