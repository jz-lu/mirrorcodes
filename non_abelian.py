import os
import subprocess
import tempfile
import pathlib
import textwrap
import ast
import textwrap
from dataclasses import dataclass

@dataclass(frozen=True)
class IndexedGroupOps:
    n: int
    i: int
    description: str
    inv_table: list[int]        # inv_table[a] = a^{-1} (0-based)
    mul_table: list[list[int]]  # mul_table[a][b] = a*b (0-based)

    def inv(self, a: int) -> int:
        if not (0 <= a < self.n):
            raise IndexError(f"element index out of range: {a}")
        return self.inv_table[a]

    def mul(self, *idxs: int) -> int:
        if len(idxs) < 2:
            raise ValueError("mul() needs at least two indices")
        for x in idxs:
            if not (0 <= x < self.n):
                raise IndexError(f"element index out of range: {x}")
        acc = idxs[0]
        for x in idxs[1:]:
            acc = self.mul_table[acc][x]
        return acc


def build_indexed_group_ops(group_dict: dict, *, gap_bat: str, timeout=120) -> IndexedGroupOps:
    """
    group_dict: {"n": n, "i": i, "description": "..."} from your enumeration.
    Returns an IndexedGroupOps object with 0-based indexing and identity = 0.

    Requires your working _run_gap_file_via_runtime_bash(code, gap_bat=..., timeout=...) -> stdout (str).
    """
    n = int(group_dict["n"])
    i = int(group_dict["i"])
    desc = str(group_dict.get("description", ""))

    gap_code = textwrap.dedent(f"""
        # Build 0-based indexing with identity forced to position 0.
        if LoadPackage("smallgrp") = fail then
            Print("ERR\\tNO_SMALLGRP\\n");
            QUIT;
        fi;

        G := SmallGroup({n}, {i});
        elts := Elements(G);
        e := Identity(G);

        # ord[1] will be identity; ord[2..] are remaining elements in Elements(G) order
        ord := Concatenation([e], Filtered(elts, x -> x <> e));
        len := Length(ord);

        # Inverse table: inv[a] = index of ord[a]^-1, with 0-based output
        invs := List([1..len], a -> Position(ord, ord[a]^-1) - 1);

        Print("INV\\t", invs, "\\n");

        # Multiplication table rows: ROW <r> <list>
        for a in [1..len] do
            row := List([1..len], b -> Position(ord, ord[a]*ord[b]) - 1);
            Print("ROW\\t", a-1, "\\t", row, "\\n");
        od;

        QUIT;
    """)

    out = _run_gap_file_via_runtime_bash(gap_code, gap_bat=gap_bat, timeout=timeout)

    inv_table = None
    mul_table = [None] * n

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

    # Basic sanity checks
    if len(inv_table) != n:
        raise RuntimeError(f"INV length {len(inv_table)} != n={n}")
    if any(len(row) != n for row in mul_table):
        raise RuntimeError("One or more MUL rows have the wrong length.")
    if inv_table[0] != 0:
        raise RuntimeError("Index 0 is not self-inverse; identity mapping likely failed.")

    return IndexedGroupOps(n=n, i=i, description=desc, inv_table=inv_table, mul_table=mul_table)

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

    if "QUIT;" not in code:
        code += "\nQUIT;\n"

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


def nonabelian_groups_of_order(n: int, *, gap_bat: str, timeout=120):
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

if __name__ == "__main__":
    gap_bat = r"C:/Users/andsin/AppData/Local/GAP-4.15.1/gap.bat"
    print(nonabelian_groups_of_order(24, gap_bat=gap_bat))