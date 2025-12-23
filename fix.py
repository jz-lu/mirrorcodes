from pathlib import Path
import pickle
from filter import stage3
import numpy as np

for i in range(301):
    if Path(f"data/STAGE3_n{i}.pkl").is_file():
        with open(f"data/STAGE3_n{i}.pkl", "rb") as f:
            data = pickle.load(f)
        new_data = []
        save = False
        for code in data:
            if code[5] > 16 or code[5] == -1:
                print(code)
                save = True
                result = stage3(np.prod(code[0]), [code[:5]], 36000, True, True)
                if len(result) == 0:
                    new = None
                else:
                    new = result[0]
            else:
                new = code
            if new is not None:
                new_data += [new]
        if save:
            with open(f"data/STAGE3_n{i}.pkl", "wb") as f:
                pickle.dump(new_data, f)
    cur = 0
    while Path(f"data/STAGE3_n{i}_part{cur}.pkl").is_file():
        with open(f"data/STAGE3_n{i}_part{cur}.pkl", "rb") as f:
            data = pickle.load(f)
        new_data = []
        save = False
        for code in data:
            if code[5] > 16 or code[5] == -1:
                print(code)
                save = True
                result = stage3(np.prod(code[0]), [code[:5]], 36000, True, True)
                if len(result) == 0:
                    new = None
                else:
                    new = result[0]
            else:
                new = code
            if new is not None:
                new_data += [new]
        if save:
            with open(f"data/STAGE3_n{i}_part{cur}.pkl", "wb") as f:
                pickle.dump(new_data, f)
        cur += 1
