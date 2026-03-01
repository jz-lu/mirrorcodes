from pathlib import Path
import pickle
import numpy as np

result = []
#pair_list = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]
pair_list = [(3, 3)]
for pair in pair_list:
    z, x = pair
    codes = set()
    counts = [[[0 for k in range(500)] for j in range(101)] for i in range(301)]
    for i in range(101):
        if Path(f"NAdata{z}{x}/NA_STAGE3_n{i}.pkl").is_file():
            with open(f"NAdata{z}{x}/NA_STAGE3_n{i}.pkl", "rb") as f:
                data = pickle.load(f)
            for code in data:
                n, k, d = code[0][0], code[4], code[5]
                if d > -1:
                    counts[int(2 * d)][k][i] += 1
                codes.add((i, k, d, code[3]))
    sortedCodes = [[[] for j in range(101)] for i in range(201)]
    SYMcodes = [[[] for j in range(101)] for i in range(201)]
    total = 0
    for i in codes:
        if i[2] >= 0:
            sortedCodes[int(2 * i[2])][i[1]] += [i[0]]
            counts[int(2 * i[2])][i[1]][i[0]] += 1
            if i[3]:
                SYMcodes[int(2 * i[2])][i[1]] += [i[0]]
    filtered = []
    for d in range(201):
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
                    total += counts[d][k][n]
                    filtered += [(n, k, d // 2 + (0 if d % 2 == 0 else 0.5), n in SYMcodes[d][k])]
    print()
    print(f'Z: {z}, X:{x}')
    print(filtered)
    print(f'Total = {total}')
    print()
    #print(result)
    for g in filtered:
        if Path(f"NAdata{z}{x}/NA_STAGE3_n{g[0]}.pkl").is_file():
            with open(f"NAdata{z}{x}/NA_STAGE3_n{g[0]}.pkl", "rb") as f:
                data = pickle.load(f)
            for code in data:
                if code[4] == g[1] and code[5] == g[2]:
                    result += [code]
#with open('goodcodes.pkl', 'wb') as f:
#    pickle.dump(result, f)
