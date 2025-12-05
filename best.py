from pathlib import Path
import pickle

codes = set()
for i in range(301):
    if Path(f"data/STAGE3_n{i}.pkl").is_file():
        with open(f"data/STAGE3_n{i}.pkl", "rb") as f:
            data = pickle.load(f)
        for code in data:
            codes.add((i, code[4], code[5], code[3]))
sortedCodes = [[[] for j in range(101)] for i in range(51)]
CSScodes = [[[] for j in range(101)] for i in range(51)]
for i in codes:
    if i[2] >= 0:
        sortedCodes[int(2 * i[2])][i[1]] += [i[0]]
        if i[3]:
            CSScodes[int(2 * i[2])][i[1]] += [i[0]]
filtered = []
for d in range(51):
    for k in range(101):
        sortedCodes[d][k].sort()
        for code in set(sortedCodes[d][k]):
            best = True
            for code2 in codes:
                n = code
                n2, k2, d2 = code2[0], code2[1], 2 * code2[2]
                if (n2 <= n and k2 * n >= k * n2 and d2 >= d and
                    (n2 < n or k2 * n > k * n2 or d2 > d)):
                    if k2 * n == k * n2 and d2 == d and 4 * k2 >= k:
                        continue
                    best = False
                    break
            if best:
                filtered += [(n, k, d // 2 + (0 if d % 2 == 0 else 0.5), n in CSScodes[d][k])]
print(filtered)
