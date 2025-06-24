from pathlib import Path
import pickle

for s in range(1, 4):
    missing = []
    for i in range(201):
        if Path(f"data/STAGE{s}_n{i}.pkl").is_file():
            continue
        if Path(f"data/STAGE{s}_n{i}_part0.pkl").is_file():
            result = []
            cur = 0
            while Path(f"data/STAGE{s}_n{i}_part{cur}.pkl").is_file():
                with open(f"data/STAGE{s}_n{i}_part{cur}.pkl", "rb") as f:
                    result += pickle.load(f)
                cur += 1
            with open(f"data/STAGE{s}_n{i}.pkl", "wb") as f:
                pickle.dump(result, f)
            continue
        if s == 3 and Path(f"data/STAGE2_n{i}.pkl").is_file():
            with open(f"data/STAGE2_n{i}.pkl", "rb") as f:
                missing += [f"{i} ({len(pickle.load(f))} codes)"]
        else:
            missing += [i]
    print(f"Missing {missing} for stage {s}")
