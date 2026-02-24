from pathlib import Path
import pickle
from numpy import prod

codes = []
for i in range(130):
    if Path(f"data/STAGE3_n{i}.pkl").is_file():
        with open(f"data/STAGE3_n{i}.pkl", "rb") as f:
            codes += pickle.load(f)
for code in codes:
    if code[3]:
        continue
    for code2 in codes:
        n, k, d = prod(code[0]), code[4], code[5]
        n2, k2, d2 = prod(code2[0]), code2[4], code2[5]
        if n2 == 4 * n and k2 == 2 * k and d2 == 2 * d:
            print(code, code2)
