from pathlib import Path
import glob
import pickle
import os

codes = set()
for i in range(201):
    if Path(f"data/STAGE3_n{i}.pkl").is_file():
        with open(f"data/STAGE3_n{i}.pkl", "rb") as f:
            data = pickle.load(f)
        for code in data:
            codes.add((i, code[4], code[5]))
sortedCodes = [[[] for j in range(101)] for i in range(51)]
for i in codes:
    if i[2] >= 0:
        sortedCodes[i[2]][i[1]] += [i[0]]
filtered = []
for d in range(51):
    for k in range(101):
        sortedCodes[d][k].sort()
        for code in sortedCodes[d][k]:
            best = True
            for code2 in codes:
                if (code2[0] <= code and code2[1] * code >= k * code2[0]
                    and code2[2] >= d and (code2[0] < code
                    or code2[1] * code > k * code2[0] or code2[2] > d)):
                    best = False
                    break
            if best:
                filtered += [(code, k, d)]
print(filtered)
