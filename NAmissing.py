from pathlib import Path
import glob
import pickle
import os
from non_abelian import nonabelian_groups_of_order

#pair_list = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]
pair_list = [(3, 3)]
for pair in pair_list:
    z, x = pair
    print(f'Z = {z}, X = {x}')
    results = [[], [], [], [], []]
    for s in range(1, 4):
        missing = []
        for i in range(151):
            if Path(f"NAdata{z}{x}/NA_STAGE{s}_n{i}_part0.pkl").is_file():
                result = []
                cur = 0
                while Path(f"NAdata{z}{x}/NA_STAGE{s}_n{i}_part{cur}.pkl").is_file():
                    with open(f"NAdata{z}{x}/NA_STAGE{s}_n{i}_part{cur}.pkl", "rb") as f:
                        result += pickle.load(f)
                    cur += 1
                if cur == len(glob.glob(f"NAdata{z}{x}/NA_STAGE{s}_n{i}_part*.pkl")):
                    with open(f"NAdata{z}{x}/NA_STAGE{s}_n{i}.pkl", "wb") as f:
                        pickle.dump(result, f)
                    continue
                else:
                    print(f"Missing n = {i}, Stage {s}, Part {cur}")
                    if Path(f"NAdata{z}{x}/NA_STAGE{s}_n{i}.pkl").is_file():
                        os.remove(f"NAdata{z}{x}/NA_STAGE{s}_n{i}.pkl")
            if Path(f"NAdata{z}{x}/NA_STAGE{s}_n{i}.pkl").is_file():
                if s != 3:
                    continue
                with open(f"NAdata{z}{x}/NA_STAGE3_n{i}.pkl", "rb") as f:
                    data = pickle.load(f)
                timeouts = 0
                for code in data:
                    if code[5] == -1:
                        timeouts += 1
                with open(f"NAdata{z}{x}/NA_STAGE2_n{i}.pkl", "rb") as f:
                    l = pickle.load(f)
                if timeouts > 0:
                    missing += [f"{i} ({timeouts} timeouts of {len(l)} codes)"]
                    results[s] += [i]
                continue
            if s == 3 and Path(f"NAdata{z}{x}/NA_STAGE2_n{i}.pkl").is_file():
                with open(f"NAdata{z}{x}/NA_STAGE2_n{i}.pkl", "rb") as f:
                    missing += [f"{i} ({len(pickle.load(f))} codes)"]
                    results[s] += [i]
                continue
            add = True
            for t in range(1, s):
                if i in results[t]:
                    add = False
            if add:
                missing += [i]
                results[s] += [i]
        if s == 1:
            new_missing = []
            for i in missing:
                if i in [128, 192, 256]:
                    count = 'oo'
                else:
                    count = str(len(nonabelian_groups_of_order(i)))
                new_missing += [f"{i} (" + count + " non-abelian groups)"]
            missing = new_missing
        print(f"Missing {missing} for stage {s}")
    print()
