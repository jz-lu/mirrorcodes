import os
import subprocess
import tempfile
import pathlib
import textwrap

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

    print(out)
    print(err)
    return out, err
    res = []
    for line in out.splitlines():
        i_str, desc = line.split("\t", 1)
        res.append({"n": n, "i": int(i_str), "description": desc})
    return res

if __name__ == "__main__":
    gap_bat = r"C:/Users/andsin/AppData/Local/GAP-4.15.1/gap.bat"
    print(nonabelian_groups_of_order(20, gap_bat=gap_bat))