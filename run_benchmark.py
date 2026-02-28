import argparse
import numpy as np
import time
import sinter

from benchmark import benchmark, StabilizerCode, make_noise_model
from mirror import MirrorCode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="benchmark codes under various noise models"
        )

    parser.add_argument(
        "--idx", "-n",
        type=int,
        required=True,
        help="index"
    )

    parser.add_argument(
        "--phenom", "-p",
        action="store_true",
        help="use a phenomenological noise model"
    )

    parser.add_argument(
        "--circuit", "-c",
        type=str,
        default='bare',
        help="index",
        choices=['bare', 'loop', 'css', 'ft', 'superdense']
    )

    args = parser.parse_args()
    idx = args.idx
    PHENOM = args.phenom
    CIRCUIT = args.circuit

    """
    Here are the codes that we will analyze.
    There are currently 9 of them.
    """

    # 30, 8, 4
    G = (2, 3, 5)
    A = [[0, 0, 0],
        [0, 0, 1],
        [0, 1, 3]]
    B = [[1, 0, 0],
        [1, 0, 2],
        [1, 1, 1]]
    CODE_30_8_4 = (G, A, B)

    # 36, 6, 6
    G = [2, 2, 3, 3]
    A = [[0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 2]]
    B = [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 2, 0]]
    CODE_36_6_6 = (G, A, B)


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
            CODE_36_6_6,
            CODE_48_8_6, 
            CODE_48_4_8, 
            CODE_72_12_6, 
            CODE_72_8_8, 
            CODE_85_8_9, 
            CODE_90_8_10, 
            CODE_144_12_12]
    NAMES = ['30_8_4',
            '36_6_6',
            '48_8_6',
            '48_4_8',
            '72_12_6',
            '72_8_8',
            '85_8_9',
            '90_8_10',
            '144_12_12']

    code_param = CODES[idx]
    code = MirrorCode(*code_param)
    CODE_NAME = NAMES[idx]
    NOISE_MODEL_NAME = 'phenom' if PHENOM else 'SI1000'
    n, k, d = [int(x) for x in CODE_NAME.split('_')]
    CODE_NAME = f'[[{n}, {k}, {d}]]'

    print("Finding stabilizers...")
    stabilizers = code.get_stim_tableau()
    print(f"Parameters: [[{code.get_n()}, {code.get_k()}]]")
    print("Done.")

    benchmarker = StabilizerCode(stabilizers, verbose=False, name=CODE_NAME)

    T_LOW = 6 # min error rate is 10^-T_LOW
    T_HIGH = 2 # max error rate is 10^-T_HIGH
    NUM_PROBS = 8
    NUM_SHOTS = 100_000 * round(n / 30)

    if PHENOM:
        T_LOW = 2
        T_HIGH = 0.8
        NUM_PROBS = 20
        NUM_SHOTS = 10_000

    IDENTIFIER = f"{idx}_{'phenom' if PHENOM else CIRCUIT}_{NAMES[idx]}_{NUM_SHOTS}s"

    # Define the main parameters of the benchmarking
    ROUND_CHOICES = list(set([1, 2, d-3, d, d+3]))
    PS = np.logspace(-T_LOW, -T_HIGH, NUM_PROBS)

    print("Making syndrome extraction circuits...")
    SECS = None
    if PHENOM:
        print("Syndrome extraction: phenomenological")
        SECS = [code.phenomenological_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                    num_rounds=nrd
                                    )
                                    for nrd in ROUND_CHOICES
                                    for i in range(len(PS))
                ]
    elif CIRCUIT == 'bare':
        print("Syndrome extraction: bare circuit")
        SECS = [code.bare_ancilla_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                    num_rounds=nrd
                                    )
                                    for nrd in ROUND_CHOICES
                                    for i in range(len(PS))
                ]
    elif CIRCUIT == 'loop':
        print("Syndrome extraction: loop circuit")
        SECS = [code.loop_flag_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                    num_rounds=nrd
                                    )
                                    for nrd in ROUND_CHOICES
                                    for i in range(len(PS))
                ]
    elif CIRCUIT == 'css':
        print("Syndrome extraction: CSS fault-tolerant circuit")
        SECS = [code.ft_for_w6_css_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                    num_rounds=nrd
                                    )
                                    for nrd in ROUND_CHOICES
                                    for i in range(len(PS))
                ]
    elif CIRCUIT == 'ft':
        print("Syndrome extraction: general fault-tolerant circuit")
        SECS = [code.ft_for_w6_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                    num_rounds=nrd
                                    )
                                    for nrd in ROUND_CHOICES
                                    for i in range(len(PS))
                ]

    elif CIRCUIT == 'superdense':
        print("Syndrome extraction: superdense circuit")
        SECS = [code.superdense_sec(noise=make_noise_model(PS[i], NOISE_MODEL_NAME),
                                        num_rounds=nrd
                                        )
                                        for nrd in ROUND_CHOICES
                                        for i in range(len(PS))
                    ]
    print("Done")

    print("Benchmarking...")
    sinter_stats = benchmarker.sinter_benchmark(ps=PS,
                                secs=SECS,
                                rounds_choices=ROUND_CHOICES,
                                num_shots=NUM_SHOTS,
                                plot=True,
                                verbose=True,
                                save_as=f"plot_{IDENTIFIER}.jpg",
                                phenom=PHENOM
                                )
    print("Done")


    with open(f"data_{IDENTIFIER}.csv", "w") as f:
        print(sinter.CSV_HEADER)
        for s in sinter_stats:
            print(s.to_csv_line(), file=f)



