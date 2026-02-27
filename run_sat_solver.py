import pickle
import numpy as np
import argparse
from mirror import MirrorCode
from benchmark import make_noise_model
import time

parser = argparse.ArgumentParser(
        description="Run SAT solver to schedule circuits on some mirror codes"
    )

parser.add_argument(
    "--idx", "-n",
    type=int,
    required=True,
    help="index"
)

args = parser.parse_args()
idx = args.idx

# 30, 8, 4
G = (2, 3, 5)
A = [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 3]]
B = [[1, 0, 0],
     [1, 0, 2],
     [1, 1, 1]]
CODE_30_8_4 = (G, A, B)


# 48, 8, 6
G = (2, 2, 2, 2, 3)
A = [[0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1],
     [0, 0, 1, 0, 2]]
B = [[0, 0, 1, 1, 0],
     [0, 1, 0, 0, 1],
     [1, 0, 0, 0, 2]]
CODE_48_8_6 = (G, A, B)


# 48, 4, 8
G = (16, 3)
A = [[0, 0],
     [0, 1],
     [2, 2]]
B = [[1, 0],
     [3, 1],
     [13, 2]]
CODE_48_4_8 = (G, A, B)


# 72, 12, 6
G = (2, 4, 3, 3)
A = [[0, 0, 0, 0],
     [0, 2, 0, 1],
     [1, 0, 0, 2]]
B = [[0, 1, 0, 0],
     [0, 3, 1, 0],
     [1, 1, 2, 0]]
CODE_72_12_6 = (G, A, B)


# 72, 8, 8
G = (2, 4, 3, 3)
A = [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 1, 0, 2]]
B = [[1, 0, 0, 0],
     [1, 0, 1, 0],
     [1, 1, 2, 0]]
CODE_72_8_8 = (G, A, B)


# 85, 8, 9
G = (5, 17)
A = [[0, 0],
     [0, 1],
     [1, 9]]
B = [[0, 0],
     [0, 4],
     [1, 2]]
CODE_85_8_9 = (G, A, B)


# 90, 8, 10
G = (2, 3, 3, 5)
A = [[0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 2, 1]]
B = [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 2, 0, 2]]
CODE_90_8_10 = (G, A, B)


# 144, 12, 12
G = (2, 2, 4, 3, 3)
A = [[0, 0, 0, 0, 0],
     [0, 0, 1, 0, 1],
     [0, 1, 0, 0, 2]]
B = [[1, 0, 0, 0, 0],
     [1, 0, 1, 1, 0],
     [1, 1, 3, 2, 0]]
CODE_144_12_12 = (G, A, B)

CODES = [CODE_30_8_4, 
         CODE_48_8_6, 
         CODE_48_4_8, 
         CODE_72_12_6, 
         CODE_72_8_8, 
         CODE_85_8_9, 
         CODE_90_8_10, 
         CODE_30_8_4]

i = idx
param = CODES[i]

G, A, B = param
code = MirrorCode(group = G, z0 = A, x0 = B)

print(f"[{i}] Bare...")
start = time.time()
code.bare_ancilla_sec(noise=make_noise_model(0.01, "SI1000"), num_rounds=1)
end = time.time()
mins, secs = divmod(end - start, 60)
print(f"Elapsed time: {int(mins)}m {int(secs)}s")

print(f"[{i}] Loop...")
code.loop_flag_sec(noise=make_noise_model(0.01, "SI1000"), num_rounds=1)

print(f"[{i}] FT-CSS...")
code.ft_for_w6_css_sec(noise=make_noise_model(0.01, "SI1000"), num_rounds=1)

print(f"[{i}] FT...")
code.ft_for_w6_sec(noise=make_noise_model(0.01, "SI1000"), num_rounds=1)

print(f"[{i}] Superdense...")
code.superdense_sec(noise=make_noise_model(0.01, "SI1000"), num_rounds=1)

print(f"[{i}] Phenomenological...")
code.phenomenological_sec(0.01, 0.01, 1)

