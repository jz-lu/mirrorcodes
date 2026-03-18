from pathlib import Path
import pickle
from filter import stage3
import numpy as np

l1 = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]
l1 = [(3, 3)]
np.random.shuffle(l1)
for pair in l1:
    z, x = pair
    l2 = list(range(301))
    l2 = [240]
    np.random.shuffle(l2)
    for i in l2:
        cur = 0
        while Path(f"data{z}{x}/STAGE3_n{i}_part{cur}.pkl").is_file():
            with open(f"data{z}{x}/STAGE3_n{i}_part{cur}.pkl", "rb") as f:
                data = pickle.load(f)
            new_data = []
            save = False
            for code in data:
                n, k, d = np.prod(code[0]), code[4], code[5]
                if d == -1:
                    print(code)
                    result = stage3(n, [code[:5]], 36000, True, True)
                    if len(result) == 0:
                        save = True
                        new = None
                    elif (code[5] > -1 and result[0][5] + 0.5 >= code[5]) or result[0][5] == -1:
                        new = code
                    else:
                        save = True
                        new = result[0]
                else:
                    new = code
                if new is not None:
                    new_data += [new]
            if save:
                with open(f"data{z}{x}/STAGE3_n{i}_part{cur}.pkl", "wb") as f:
                    pickle.dump(new_data, f)
            cur += 1
        if Path(f"data{z}{x}/STAGE3_n{i}.pkl").is_file():
            with open(f"data{z}{x}/STAGE3_n{i}.pkl", "rb") as f:
                data = pickle.load(f)
            new_data = []
            save = False
            for code in data:
                n, k, d = np.prod(code[0]), code[4], code[5]
                if d == -1:
                    print(code)
                    result = stage3(n, codes = [code[:5]], t = 36000, verbose = True, estimate = True)
                    if len(result) == 0:
                        save = True
                        new = None
                    elif (code[5] > -1 and result[0][5] + 0.5 >= code[5]) or result[0][5] == -1:
                        new = code
                    else:
                        save = True
                        new = result[0]
                else:
                    new = code
                if new is not None:
                    new_data += [new]
            if save:
                with open(f"data{z}{x}/STAGE3_n{i}.pkl", "wb") as f:
                    pickle.dump(new_data, f)
