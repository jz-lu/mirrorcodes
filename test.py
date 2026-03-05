import math
from functools import reduce
import operator
import numpy as np


def _is_integer_like(x, tol: float = 1e-9) -> bool:
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return abs(float(x) - round(float(x))) <= tol
    return False


def _compute_d(L):
    """
    L is a length-3 tuple.
    Returns: (d_str, d_key_int, is_exact, errors)
    """
    Lp = [x for x in L if float(x) >= 0]  # remove negatives
    errors = []
    if not Lp:
        errors.append("L' is empty after removing negatives.")
        return ("?", None, False, errors)

    ints = [int(round(float(x))) for x in Lp if _is_integer_like(x)]
    distinct_ints = sorted(set(ints))
    if len(distinct_ints) > 1:
        errors.append(f"More than one distinct integer in L': {distinct_ints}")

    m = min(float(x) for x in Lp)
    cm = math.ceil(m)

    bad = [v for v in distinct_ints if v != cm]
    if bad:
        errors.append(f"L' contains integer(s) not equal to ceil(min(L'))={cm}: {bad}")

    if ints:
        return (str(cm), cm, True, errors)
    return (r"\leq " + str(cm), cm, False, errors)


def _latex_G(G):
    parts = [rf"\mathbb{{Z}}_{{{int(q)}}}" for q in G]
    return "$" + r"\times\allowbreak ".join(parts) + "$"


def _latex_tuple(t):
    return "$(" + ",".join(str(int(x)) for x in t) + ")$"


def _latex_tuple_list_inline(tuples_list):
    return r",\allowbreak\ ".join(_latex_tuple(t) for t in tuples_list)


def data_to_ltablex_table(
    data,
    tabcolsep_pt=3,
    arraystretch=1.08,
    fontsize_cmd=r"\footnotesize",
    label=r"tab:blank",
):
    """
    OUTPUTS a multipage tabularx table intended for the ltablex package.

    Required LaTeX preamble:
      \\usepackage{array}
      \\usepackage{ltablex}
      \\keepXColumns

    IMPORTANT:
      - Put the generated table directly in the document body (NOT inside \\begin{table}...).
      - Compile twice (longtable behavior).

    This output is wrapped in \\begingroup ... \\endgroup so font/spacing changes do not
    affect text before/after the table.
    """
    rows = []
    errors = []

    for idx, item in enumerate(data):
        (G, A, B, CSS, k), L = item
        n = int(reduce(operator.mul, [int(q) for q in G], 1))

        A_list = [tuple(map(int, r)) for r in np.array(A).tolist()]
        B_list = [tuple(map(int, r)) for r in np.array(B).tolist()]
        wX, wZ = len(A_list), len(B_list)

        d_str, d_key, d_exact, d_errors = _compute_d(L)
        if d_errors:
            errors.append({"index": idx, "G": G, "k": k, "L": L, "errors": d_errors})

        rows.append(
            {
                "n": n,
                "k": int(k),
                "d": d_str,
                "d_key": d_key,
                "d_exact": d_exact,
                "CSS": bool(CSS),
                "wX": wX,
                "wZ": wZ,
                "G": tuple(int(q) for q in G),
                "A": A_list,
                "B": B_list,
            }
        )

    rows.sort(
        key=lambda r: (
            r["wX"],
            r["wZ"],
            r["n"],
            r["k"],
            r["d_key"],
            0 if r["d_exact"] else 1,
        )
    )

    def latex_bool(b):
        return r"\texttt{True}" if b else r"\texttt{False}"

    def latex_d(d):
        return "$" + d + "$"

    # Numeric columns fixed width with vertical centering (m{...}),
    # text-heavy columns are X (auto-fit). X is also vertical-centered by redefining tabularxcolumn.
    colspec = (
        r"|>{\centering\arraybackslash}m{0.07\textwidth}"
        r"|>{\centering\arraybackslash}m{0.05\textwidth}"
        r"|>{\centering\arraybackslash}m{0.06\textwidth}"
        r"|>{\centering\arraybackslash}m{0.08\textwidth}"
        r"|>{\centering\arraybackslash}m{0.05\textwidth}"
        r"|>{\centering\arraybackslash}m{0.05\textwidth}"
        r"|X|X|X|"
    )

    out = []
    out.append(r"\begingroup")

    # Local (scoped) style adjustments:
    out.append(r"\renewcommand{\tabularxcolumn}[1]{>{\centering\arraybackslash}m{#1}}")
    out.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep_pt}pt}}")
    out.append(rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}")
    out.append(r"\setlength{\LTpre}{0pt}")
    out.append(r"\setlength{\LTpost}{0pt}")
    out.append(r"\setlength{\LTleft}{0pt}")
    out.append(r"\setlength{\LTright}{0pt}")
    out.append(fontsize_cmd)

    out.append(rf"\begin{{tabularx}}{{\textwidth}}{{{colspec}}}")
    out.append(r"\hline")
    out.append(r"$n$ & $k$ & $d$ & CSS? & $w_X$ & $w_Z$ & $G$ & $A$ & $B$ \\")
    out.append(r"\hline\hline")
    out.append(r"\endfirsthead")

    out.append(r"\hline")
    out.append(r"$n$ & $k$ & $d$ & CSS? & $w_X$ & $w_Z$ & $G$ & $A$ & $B$ \\")
    out.append(r"\hline\hline")
    out.append(r"\endhead")

    out.append(r"\hline")
    out.append(r"\multicolumn{9}{r}{\emph{Continued on next page}} \\")
    out.append(r"\endfoot")

    out.append(r"\endlastfoot")

    for i, r in enumerate(rows):
        A_cell = _latex_tuple_list_inline(r["A"])
        B_cell = _latex_tuple_list_inline(r["B"])

        out.append(
            " & ".join(
                [
                    str(r["n"]),
                    str(r["k"]),
                    latex_d(r["d"]),
                    latex_bool(r["CSS"]),
                    str(r["wX"]),
                    str(r["wZ"]),
                    _latex_G(r["G"]),
                    A_cell,
                    B_cell,
                ]
            )
            + r" \\"
        )

        if i < len(rows) - 1:
            grp = (r["wX"], r["wZ"])
            nxt = (rows[i + 1]["wX"], rows[i + 1]["wZ"])
            out.append(r"\hline" if grp == nxt else r"\hline\hline")

    # Bottom rule BEFORE caption, and no rule after caption:
    out.append(r"\hline")
    out.append(rf"\caption{{}}\label{{{label}}}\\")
    out.append(r"\end{tabularx}")

    out.append(r"\endgroup")

    return "\n".join(out), errors

# ---------------------------
# Example usage:
#   latex, errs = latex_table_from_data(DATA)
#   print("\n".join(errs))   # if any
#   print(latex)
# ---------------------------

DATA = [(((2, 3, 5), np.array([[0, 0, 0],
       [0, 0, 1]], dtype=int), np.array([[1, 0, 0],
       [1, 1, 0],
       [1, 2, 2]], dtype=int), True, 4), (5, 5, -1)), (((4, 3, 3), np.array([[0, 0, 0],
       [2, 0, 1]], dtype=int), np.array([[1, 0, 0],
       [1, 1, 0],
       [1, 2, 1]], dtype=int), True, 4), (6, 6, -1)), (((16,), np.array([[0],
       [4]], dtype=int), np.array([[ 1],
       [ 3],
       [ 5],
       [11]], dtype=int), True, 4), (4, 4, -1)), (((8, 3), np.array([[0, 0],
       [2, 0]], dtype=int), np.array([[1, 0],
       [1, 1],
       [3, 0],
       [5, 1]], dtype=int), True, 6), (4, 4, -1)), (((2, 3, 7), np.array([[0, 0, 0],
       [0, 0, 1]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 2],
       [1, 1, 0],
       [1, 1, 3]], dtype=int), True, 6), (6, 6, -1)), (((8, 7), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[1, 0],
       [1, 2],
       [3, 0],
       [3, 3]], dtype=int), True, 8), (6, 6, -1)), (((2, 5, 7), np.array([[0, 0, 0],
       [0, 0, 1]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 2],
       [1, 1, 0],
       [1, 1, 3]], dtype=int), True, 10), (6, 6, -1)), (((4, 9), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[0, 0],
       [1, 1],
       [2, 3],
       [3, 2]], dtype=int), False, 3), (7, -1, 6.5)), (((4, 11), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[1, 0],
       [1, 2],
       [3, 0],
       [3, 6]], dtype=int), True, 4), (7, 7, -1)), (((2, 27), np.array([[0, 0],
       [0, 3]], dtype=int), np.array([[ 1,  0],
       [ 1,  1],
       [ 1,  6],
       [ 1, 16]], dtype=int), True, 6), (7, 7, -1)), (((8, 9), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[1, 0],
       [1, 2],
       [3, 0],
       [3, 4]], dtype=int), True, 8), (7, 7, -1)), (((2, 9, 5), np.array([[0, 0, 0],
       [0, 1, 0]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 1],
       [1, 2, 0],
       [1, 4, 1]], dtype=int), True, 10), (7, 7, -1)), (((2, 9, 7), np.array([[0, 0, 0],
       [0, 1, 0]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 1],
       [1, 2, 0],
       [1, 4, 1]], dtype=int), True, 14), (7, 7, -1)), (((16, 3), np.array([[0, 0],
       [4, 1]], dtype=int), np.array([[1, 0],
       [3, 0],
       [7, 2],
       [9, 1]], dtype=int), True, 4), (8, 8, -1)), (((8, 9), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[1, 0],
       [3, 0],
       [5, 1],
       [7, 6]], dtype=int), True, 6), (9, 9, -1)), (((2, 8, 5), np.array([[0, 0, 0],
       [1, 0, 1]], dtype=int), np.array([[0, 1, 0],
       [0, 3, 0],
       [1, 5, 1],
       [1, 7, 2]], dtype=int), True, 6), (9.5, -1, -1)), (((4, 3, 11), np.array([[0, 0, 0],
       [0, 0, 1]], dtype=int), np.array([[1, 0, 0],
       [1, 1, 1],
       [3, 0, 2],
       [3, 1, 5]], dtype=int), True, 8), (10.5, -1, -1)), (((7, 13), np.array([[0, 0],
       [0, 1]], dtype=int), np.array([[0, 0],
       [1, 0],
       [2, 3],
       [5, 4]], dtype=int), False, 4), (12.5, -1, -1)), (((16, 9), np.array([[0, 0],
       [8, 1]], dtype=int), np.array([[ 1,  0],
       [ 3,  0],
       [ 5,  1],
       [15,  7]], dtype=int), True, 6), (13.5, -1, -1)), (((2, 81), np.array([[0, 0],
       [0, 3]], dtype=int), np.array([[ 1,  0],
       [ 1,  4],
       [ 1, 16],
       [ 1, 51]], dtype=int), True, 6), (14.5, -1, -1)), (((2, 3, 29), np.array([[0, 0, 0],
       [0, 0, 1]], dtype=int), np.array([[ 1,  0,  0],
       [ 1,  0,  4],
       [ 1,  1,  7],
       [ 1,  1, 18]], dtype=int), True, 6), (15.5, -1, -1)), (((2, 9), np.array([[0, 0],
       [0, 1],
       [0, 2]]), np.array([[1, 0],
       [1, 1],
       [1, 5]]), True, 4), (4, 4, -1)), (((2, 3, 5), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 3]]), np.array([[1, 0, 0],
       [1, 0, 2],
       [1, 1, 1]]), True, 8), (4, 4, -1)), (((2, 3, 5), np.array([[0, 0, 0],
       [0, 1, 0],
       [0, 2, 1]]), np.array([[1, 0, 0],
       [1, 1, 1],
       [1, 2, 3]]), True, 4), (6, 6, -1)), (((2, 2, 3, 3), np.array([[0, 0, 0, 0],
       [0, 1, 0, 1],
       [1, 0, 0, 2]]), np.array([[0, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 1, 2, 0]]), False, 6), (6, 6, -1)), (((2, 2, 2, 2, 3), np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 0, 2]]), np.array([[0, 0, 1, 1, 0],
       [0, 1, 0, 0, 1],
       [1, 0, 0, 0, 2]]), False, 8), (6, 6, -1)), (((2, 4, 3, 3), np.array([[0, 0, 0, 0],
       [0, 2, 0, 1],
       [1, 0, 0, 2]]), np.array([[0, 1, 0, 0],
       [0, 3, 1, 0],
       [1, 1, 2, 0]]), True, 12), (6, 6, -1)), (((2, 2, 2, 4, 3), np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 2, 1],
       [0, 0, 1, 0, 2]]), np.array([[0, 0, 0, 1, 0],
       [0, 1, 0, 1, 1],
       [1, 0, 0, 1, 2]]), True, 16), (6, 6, -1)), (((16, 3), np.array([[0, 0],
       [4, 1]], dtype=int), np.array([[1, 0],
       [3, 0],
       [7, 2],
       [9, 1]], dtype=int), True, 4), (8, 8, -1)), (((8, 7), np.array([[0, 0],
       [0, 1],
       [2, 3]]), np.array([[1, 0],
       [1, 3],
       [5, 2]]), True, 6), (8, 8, -1)), (((8, 3, 3), np.array([[0, 0, 0],
       [0, 0, 1],
       [2, 0, 2]]), np.array([[1, 0, 0],
       [1, 1, 0],
       [3, 2, 0]]), True, 8), (8, 8, -1)), (((5, 17), np.array([[0, 0],
       [0, 1],
       [1, 9]]), np.array([[0, 0],
       [0, 4],
       [1, 2]]), False, 8), (9, -1, 8.5)), (((2, 2, 3, 5), np.array([[0, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 2, 1]]), np.array([[0, 1, 0, 0],
       [1, 0, 1, 1],
       [1, 1, 2, 2]]), False, 4), (10, -1, -1)), (((4, 3, 7), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 3]]), np.array([[1, 0, 0],
       [1, 1, 5],
       [3, 0, 1]]), True, 6), (10, -1, -1)), (((2, 3, 3, 5), np.array([[0, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 2, 1]]), np.array([[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 2, 0, 2]]), True, 8), (10, -1, -1)), (((2, 9, 7), np.array([[0, 0, 0],
       [0, 1, 0],
       [0, 5, 1]]), np.array([[1, 0, 0],
       [1, 1, 1],
       [1, 6, 6]]), True, 12), (10, -1, -1)), (((8, 3, 5), np.array([[0, 0, 0],
       [0, 0, 1],
       [2, 1, 3]]), np.array([[1, 0, 0],
       [1, 1, 1],
       [7, 0, 2]]), True, 8), (11.5, -1, -1)), (((2, 8, 3, 3), np.array([[0, 0, 0, 0],
       [0, 2, 0, 1],
       [1, 0, 0, 2]]), np.array([[0, 1, 0, 0],
       [0, 3, 1, 0],
       [1, 5, 2, 0]]), True, 12), (12, -1, -1)), (((2, 4, 3, 3, 3), np.array([[0, 0, 0, 0, 0],
       [0, 2, 0, 0, 1],
       [1, 0, 0, 1, 0]], dtype=int), np.array([[0, 1, 0, 0, 0],
       [0, 3, 1, 0, 0],
       [1, 1, 2, 1, 1]], dtype=int), True, 12), (14, -1, -1)), (((2, 16, 3, 3), np.array([[0, 0, 0, 0],
       [0, 2, 0, 1],
       [1, 0, 0, 2]], dtype=int), np.array([[0, 1, 0, 0],
       [0, 7, 1, 0],
       [1, 3, 2, 0]], dtype=int), True, 12), (17.5, -1, -1)), (((16, 3, 5), np.array([[0, 0, 0],
       [0, 0, 1],
       [2, 1, 3]], dtype=int), np.array([[ 1,  0,  0],
       [ 5,  0,  2],
       [11,  1,  1]], dtype=int), True, 8), (19.5, -1, -1)), (((2, 4, 3), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 2, 2]], dtype=int), np.array([[0, 0, 0],
       [0, 1, 0],
       [1, 0, 0],
       [1, 3, 0]], dtype=int), False, 6), (4, 4, -1)), (((2, 3, 7), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 0, 3]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 1],
       [1, 1, 5],
       [1, 2, 5]], dtype=int), True, 12), (4, 4, -1)), (((2, 3, 7), np.array([[0, 0, 0],
       [0, 1, 1],
       [0, 2, 3]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 1],
       [1, 1, 3],
       [1, 1, 6]], dtype=int), True, 10), (5, 5, -1)), (((3, 3, 5), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 3]], dtype=int), np.array([[0, 0, 0],
       [0, 0, 2],
       [1, 0, 1],
       [2, 2, 1]], dtype=int), False, 8), (6, 6, -1)), (((4, 4, 3), np.array([[0, 0, 0],
       [0, 2, 1],
       [2, 0, 2]], dtype=int), np.array([[0, 0, 0],
       [0, 1, 0],
       [1, 0, 0],
       [3, 3, 0]], dtype=int), False, 10), (6, 6, -1)), (((2, 3, 7), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 3]], dtype=int), np.array([[1, 0, 0],
       [1, 0, 2],
       [1, 1, 1],
       [1, 2, 4]], dtype=int), True, 6), (7, 7, -1)), (((2, 31), np.array([[ 0,  0],
       [ 0,  1],
       [ 0, 12]], dtype=int), np.array([[1, 0],
       [1, 1],
       [1, 2],
       [1, 6]], dtype=int), True, 10), (7, 7, -1)), (((2, 2, 9), np.array([[0, 0, 0],
       [0, 1, 1],
       [1, 0, 2]], dtype=int), np.array([[0, 0, 0],
       [0, 1, 4],
       [1, 1, 1],
       [1, 1, 3]], dtype=int), False, 4), (7.5, -1, 7.5)), (((2, 2, 3, 5), np.array([[0, 0, 0, 0],
       [0, 1, 0, 1],
       [1, 0, 1, 3]], dtype=int), np.array([[0, 0, 0, 0],
       [0, 1, 1, 1],
       [1, 0, 0, 3],
       [1, 1, 2, 0]], dtype=int), False, 8), (8.5, -1, 8.5)), (((4, 3, 5), np.array([[0, 0, 0],
       [0, 1, 0],
       [0, 2, 1]], dtype=int), np.array([[0, 0, 0],
       [1, 0, 1],
       [2, 0, 4],
       [3, 0, 3]], dtype=int), False, 6), (9.5, -1, 9.5)), (((9, 7), np.array([[0, 0],
       [0, 1],
       [3, 3]], dtype=int), np.array([[0, 0],
       [0, 3],
       [1, 2],
       [8, 2]], dtype=int), False, 6), (10.5, -1, 10.5)), (((3, 31), np.array([[ 0,  0],
       [ 0,  1],
       [ 0, 12]], dtype=int), np.array([[ 0,  0],
       [ 0,  4],
       [ 1, 18],
       [ 2, 18]], dtype=int), False, 10), (10.5, -1, 10.5)), (((2, 3, 9), np.array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 0]], dtype=int), np.array([[0, 0, 0],
       [0, 1, 4],
       [1, 0, 2],
       [1, 0, 8]], dtype=int), False, 4), (11.5, -1, 11.5)), (((5, 17), np.array([[0, 0],
       [0, 1],
       [1, 9]], dtype=int), np.array([[0, 0],
       [0, 3],
       [0, 8],
       [2, 2]], dtype=int), False, 8), (11.5, -1, 11.5)), (((4, 3, 5), np.array([[0, 0, 0],
       [0, 1, 1],
       [2, 0, 2]], dtype=int), np.array([[0, 0, 0],
       [1, 1, 1],
       [1, 2, 4],
       [2, 0, 4]], dtype=int), False, 4), (12.5, -1, -1)), (((2, 31), np.array([[ 0,  0],
       [ 0,  1],
       [ 1, 14]], dtype=int), np.array([[ 0,  0],
       [ 0,  6],
       [ 0, 20],
       [ 1, 11]], dtype=int), False, 5), (12.5, -1, 12.5)), (((2, 3, 11), np.array([[0, 0, 0],
       [0, 1, 0],
       [0, 2, 1]], dtype=int), np.array([[0, 0, 0],
       [0, 0, 4],
       [1, 1, 1],
       [1, 1, 7]], dtype=int), False, 4), (13.5, -1, -1)), (((3, 31), np.array([[ 0,  0],
       [ 1,  1],
       [ 2, 12]], dtype=int), np.array([[ 0,  0],
       [ 0,  4],
       [ 2, 10],
       [ 2, 15]], dtype=int), False, 7), (15.5, -1, 15.5)), (((3, 25), np.array([[0, 0],
       [0, 1],
       [1, 8]], dtype=int), np.array([[ 0,  0],
       [ 0,  2],
       [ 0, 16],
       [ 2, 11]], dtype=int), False, 4), (16.5, -1, -1)), (((9, 11), np.array([[0, 0],
       [3, 1],
       [6, 3]], dtype=int), np.array([[0, 0],
       [0, 2],
       [1, 1],
       [1, 4]], dtype=int), False, 6), (18.5, -1, -1)), (((3, 31), np.array([[ 0,  0],
       [ 0,  1],
       [ 1, 12]], dtype=int), np.array([[ 0,  0],
       [ 0,  4],
       [ 0, 23],
       [ 1, 26]], dtype=int), False, 5), (20.5, -1, -1)), (((3, 3, 11), np.array([[0, 0, 0],
       [0, 1, 1],
       [0, 2, 3]], dtype=int), np.array([[0, 0, 0],
       [0, 1, 5],
       [1, 0, 4],
       [2, 1, 8]], dtype=int), False, 4), (22.5, -1, -1))]

latex, errs = data_to_ltablex_table(DATA)
print("\n".join(errs))   # if any
print(latex)