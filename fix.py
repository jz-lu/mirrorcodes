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
            if code[5] == -1:
                print(code)
                save = True
                new = stage3(np.prod(code[0]), [code[:5]], 36000, True, True)[0]
            else:
                new = code
            new_data += [code]
        if save:
            with open(f"data/STAGE3_n{i}.pkl", "wb") as f:
                pickle.dump(new_data, f)
