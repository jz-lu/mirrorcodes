from pathlib import Path
import glob
import pickle
import os

for pair in [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]:
    results = [[]]
    z, x = pair
    print(f'Z = {z}, X = {x}')
    for s in range(1, 4):
        missing = []
        for i in range(301 if z == 3 and x == 3 else 201):
            if (Path(f"data{z}{x}/STAGE{s}_n{i}_part0.pkl").is_file()
                and not Path(f"data{z}{x}/STAGE{s}_n{i}.pkl").is_file()):
                result = []
                cur = 0
                while Path(f"data{z}{x}/STAGE{s}_n{i}_part{cur}.pkl").is_file():
                    with open(f"data{z}{x}/STAGE{s}_n{i}_part{cur}.pkl", "rb") as f:
                        result += pickle.load(f)
                    cur += 1
                if cur == len(glob.glob(f'data{z}{x}/STAGE{s}_n{i}_part*.pkl')):
                    with open(f"data{z}{x}/STAGE{s}_n{i}.pkl", "wb") as f:
                        pickle.dump(result, f)
                    continue
                elif Path(f"data{z}{x}/STAGE{s}_n{i}.pkl").is_file():
                    os.remove(f"data{z}{x}/STAGE{s}_n{i}.pkl")
            if Path(f"data{z}{x}/STAGE{s}_n{i}.pkl").is_file():
                if s != 3:
                    continue
                with open(f"data{z}{x}/STAGE3_n{i}.pkl", "rb") as f:
                    data = pickle.load(f)
                timeouts = 0
                for code in data:
                    if code[5] == -1:
                        timeouts += 1
                with open(f"data{z}{x}/STAGE2_n{i}.pkl", "rb") as f:
                    l = pickle.load(f)
                if timeouts > 0:
                    missing += [f"{i} ({timeouts} timeouts of {len(l)} codes)"]
                continue
            if s == 3 and Path(f"data{z}{x}/STAGE2_n{i}.pkl").is_file():
                with open(f"data{z}{x}/STAGE2_n{i}.pkl", "rb") as f:
                    missing += [f"{i} ({len(pickle.load(f))} codes)"]
            else:
                add = True
                for t in range(1, s):
                    if i in results[t]:
                        add = False
                if add:
                    missing += [i]
        results += [missing]
        print(f"Missing {missing} for stage {s}")
    print()
