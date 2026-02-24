from pathlib import Path
import pickle
from filter import stage3
import numpy as np

for pair in [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]:
    z, x = pair
    for i in range(301):
        cur = 0
    #    while Path(f"data/STAGE3_n{i}_part{cur}.pkl").is_file():
    #        with open(f"data/STAGE3_n{i}_part{cur}.pkl", "rb") as f:
    #            data = pickle.load(f)
    #        new_data = []
    #        save = False
    #        for code in data:
    #            if code[5] >= 18 or code[5] == -1:
    #                print(code)
    #                save = True
    #                result = stage3(np.prod(code[0]), [code[:5]], 36000, True, True)
    #                if len(result) == 0:
    #                    new = None
    #                else:
    #                    new = result[0]
    #            else:
    #                new = code
    #            if new is not None:
    #                new_data += [new]
    #        if save:
    #            with open(f"data/STAGE3_n{i}_part{cur}.pkl", "wb") as f:
    #                pickle.dump(new_data, f)
    #        cur += 1
        if Path(f"data{z}{x}/STAGE3_n{i}.pkl").is_file():
            with open(f"data{z}{x}/STAGE3_n{i}.pkl", "rb") as f:
                data = pickle.load(f)
            new_data = []
            save = False
            for code in data:
                n, k, d = np.prod(code[0]), code[4], code[5]
                if (d >= 18 or d == -1 or z == 2) and not ((n == 240 and d <= 20) or (n == 288 and d <= 18)):
                    print(code)
                    save = True
                    result = stage3(np.prod(code[0]), [code[:5]], 36000, True, True)
                    if len(result) == 0:
                        new = None
                    elif result[0][5] + 0.5 >= code[5] or result[0][5] == -1:
                        new = code
                    else:
                        new = result[0]
                else:
                    new = code
                if new is not None:
                    new_data += [new]
            if save:
                with open(f"data{z}{x}/STAGE3_n{i}.pkl", "wb") as f:
                    pickle.dump(new_data, f)
