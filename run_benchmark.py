"""
Commandline-level script to benchmark some highlighted mirror codes.
There are a few different types of comparisons we can make:
    1) Rounds: fix a single code and circuit, vary the number of rounds from 1 to d+3, where d is the distance.
    2) Circuits: fix a single code, vary the type of syndrome extraction circuit used, fix rounds at 1, d.
    3) Codes: fix a circuit, fix rounds at 1, d, vary the code being used.

On the command line, --type <rounds, circuits, codes> specifies which type.
--idx / -n specifies the index of which code to use.
--circuit / -c specifies which circuit to use.
You may use only one or both of these flags at once, depending on which type you want.

If using --type rounds, then you must specify both flags.
If using --type circuits, then you need only specify which code to use, -n.
If using --type codes, then you need only specify which circuit to use, -c.

Warning: benchmarking may take several hours to run on a good laptop with 12+ cores.
"""
import argparse
import numpy as np
import time
import sinter
import stim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tesseract_decoder
import tesseract_decoder.tesseract as tesseract
from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder

decoder_dict = make_tesseract_sinter_decoders_dict()
decoder_dict['custom-tesseract-decoder'] = TesseractSinterDecoder(
    det_beam=10,
    beam_climbing=True,
    no_revisit_dets=True,
    merge_errors=True,
    pqlimit=1_000,
    num_det_orders=5,
    det_order_method=tesseract_decoder.utils.DetOrder.DetIndex,
    seed=2384753,
)

from benchmark import benchmark, StabilizerCode, make_noise_model
from mirror import MirrorCode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Benchmark codes under various noise models"
        )

    parser.add_argument(
        "--idx", "-n",
        type=int,
        default=0,
        help="Index of code to be benchmarked"
    )

    parser.add_argument(
        "--type", "-t",
        type=str,
        default='codes',
        help="Benchmarking variant",
        choices=['codes', 'rounds', 'circuits']
    )

    parser.add_argument(
        "--circuit", "-c",
        type=str,
        default='bare',
        help="Syndrome extraction circuit type",
        choices=['bare', 'loop', 'css', 'ft', 'superdense', 'phenom']
    )

    args = parser.parse_args()
    idx = args.idx
    CIRCUIT = args.circuit
    TYPE = args.type

    """
    Here are the codes that we will analyze.
    There are currently 9 of them.
    """

    # 6x6 Toric code
    G = (2, 6, 6)
    A = [[0, 0, 0],
         [0, 0, 1]]
    B = [[1, 0, 0],
         [1, 1, 0]]
    CODE_TORIC = (G, A, B)

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

    CODES = [CODE_TORIC,
            CODE_30_8_4, 
            CODE_36_6_6,
            # CODE_48_8_6, 
            CODE_48_4_8, 
            CODE_72_12_6, 
            CODE_72_8_8, 
            CODE_85_8_9, 
            # CODE_90_8_10, 
            CODE_144_12_12]
    NAMES = ['72_2_6',
            '30_8_4',
            '36_6_6',
            # '48_8_6',
            '48_4_8',
            '72_12_6',
            '72_8_8',
            '85_8_9',
            # '90_8_10',
            '144_12_12']

    T_LOW = 3 # min error rate is 10^-T_LOW
    T_HIGH = 1.5 # max error rate is 10^-T_HIGH
    NUM_PROBS = 5
    NUM_SHOTS = 15_000

    if CIRCUIT == "phenom":
        T_LOW = 2
        T_HIGH = 0.8
        NUM_PROBS = 8
        NUM_SHOTS = 20_000
    
    PS = np.logspace(-T_LOW, -T_HIGH, NUM_PROBS)

    """
    From here on out, the behavior will change based on what type of benchmarking is being done.
    """

    if TYPE == "rounds":
        """
        Fix a code given by `CODES[idx]`, fix a circuit given by `CIRCUIT`.
        Iterate over different choices of rounds.
        """
        print("=== TYPE: rounds ===")
        code_param = CODES[idx]
        code = MirrorCode(*code_param)
        CODE_NAME = NAMES[idx]
        NOISE_MODEL_NAME = 'phenom' if CIRCUIT == 'phenom' else 'SI1000'
        n, k, d = [int(x) for x in CODE_NAME.split('_')]
        CODE_NAME = f'[[{n}, {k}, {d}]]'

        print("Finding stabilizers...")
        stabilizers = code.get_stim_tableau()
        print(f"Parameters: [[{code.get_n()}, {code.get_k()}]]")
        print("Done.")

        benchmarker = StabilizerCode(stabilizers, verbose=False, name=CODE_NAME)

        IDENTIFIER = f"{TYPE}_{idx}_{'phenom' if CIRCUIT == 'phenom' else CIRCUIT}_{NAMES[idx]}_{NUM_SHOTS}s"

        # Define the main parameters of the benchmarking
        ROUND_CHOICES = list(set([d-3, d, d+3]))

        print("Making syndrome extraction circuits...")
        circ_func = None
        if CIRCUIT == 'phenom':
            print("Syndrome extraction: phenomenological")
            circ_func = code.phenomenological_sec
        elif CIRCUIT == 'bare':
            print("Syndrome extraction: bare circuit")
            circ_func = code.bare_ancilla_sec
        elif CIRCUIT == 'loop':
            print("Syndrome extraction: loop circuit")
            circ_func = code.loop_flag_sec
        elif CIRCUIT == 'css':
            print("Syndrome extraction: CSS fault-tolerant circuit")
            circ_func = code.ft_for_w6_css_sec
        elif CIRCUIT == 'ft':
            print("Syndrome extraction: general fault-tolerant circuit")
            circ_func = code.ft_for_w6_sec
        elif CIRCUIT == 'superdense':
            print("Syndrome extraction: superdense circuit")
            circ_func = code.superdense_sec
        else:
            raise SyntaxError(f"Invalid circuit type {CIRCUIT}")
        
        SECS = [circ_func(noise=make_noise_model(NOISE_MODEL_NAME, PS[i]),
                                            num_rounds=nrd
                                            )
                                            for nrd in ROUND_CHOICES
                                            for i in range(len(PS))
                        ]
        print("Done.")

        print("Benchmarking...")
        sinter_stats = benchmarker.sinter_benchmark(
                                    ps=PS,
                                    secs=SECS,
                                    num_logicals=k,
                                    rounds_choices=ROUND_CHOICES,
                                    num_shots=NUM_SHOTS,
                                    plot=True,
                                    verbose=True,
                                    save_as=f"plot_{IDENTIFIER}.pdf",
                                    phenom=(CIRCUIT == "phenom")
                                    )
        print("Done.")


        with open(f"data_{IDENTIFIER}.csv", "w") as f:
            print(sinter.CSV_HEADER, file=f)
            for s in sinter_stats:
                print(s.to_csv_line(), file=f)
    
    elif TYPE == "circuits":
        """
        Fix a code given by `CODES[idx]`, fix only 1 round and d rounds of syndrome extraction.
        Iterate over different choices of syndrome extraction circuits.
        """
        print("=== TYPE: circuits ===")
        code_param = CODES[idx]
        code = MirrorCode(*code_param)
        CODE_NAME = NAMES[idx]
        NOISE_MODEL_NAME = 'phenom' if CIRCUIT == 'phenom' else 'SI1000'
        n, k, d = [int(x) for x in CODE_NAME.split('_')]
        CODE_NAME = f'[[{n}, {k}, {d}]]'

        print("Finding stabilizers...")
        stabilizers = code.get_stim_tableau()
        print(f"Parameters: [[{code.get_n()}, {code.get_k()}]]")
        print("Done.")

        benchmarker = StabilizerCode(stabilizers, verbose=False, name=CODE_NAME)

        IDENTIFIER = f"{TYPE}_{idx}_{NAMES[idx]}_{NUM_SHOTS}s"

        # Define the main parameters of the benchmarking
        ROUND_CHOICES = list(set([d]))
        CIRCUIT_NAMES = ['bare', 'loop', 'superdense', 'css', 'ft']
        FANCY_CIRCUIT_NAMES = ['Bare', 'Loop', 'SD', 'CSS-FT', 'FT']

        tasks = []
        for circ_idx, circ_name in enumerate(CIRCUIT_NAMES):
            print(f"Making {circ_name} syndrome extraction circuits...")
            circ_func = None
            if circ_name == 'bare':
                circ_func = code.bare_ancilla_sec
            elif circ_name == 'loop':
                circ_func = code.loop_flag_sec
            elif circ_name == 'css':
                circ_func = code.ft_for_w6_css_sec
            elif circ_name == 'ft':
                circ_func = code.ft_for_w6_sec
            elif circ_name == 'superdense':
                circ_func = code.superdense_sec
            else:
                raise SyntaxError(f"Invalid circuit type {circ_name}")
            
            # Add on to the list of benchmarking tasks
            for nrd in ROUND_CHOICES:
                for i in range(len(PS)):
                    circuit = circ_func(
                                    noise=make_noise_model(NOISE_MODEL_NAME, PS[i]),
                                    num_rounds=nrd
                                )

                    tasks.append(sinter.Task(
                        circuit=circuit,
                        decoder='tesseract',
                        json_metadata={"p": PS[i], 
                                       "decoder": 'tesseract', 
                                       "rounds": nrd, 
                                       'circuit': FANCY_CIRCUIT_NAMES[circ_idx], 
                                       'circidx': circ_idx
                                       },
                    ))

            print("Done.")
            
        print("Collecting...")
        results = sinter.collect(
            num_workers=12,
            tasks=tasks,
            max_shots=NUM_SHOTS,
            decoders=['tesseract'],
            custom_decoders=decoder_dict,
            print_progress=True,
        )
        print("Done.")

        colors = plt.get_cmap('tab10').colors[:len(CIRCUIT_NAMES)]

        # Add to the plot
        print("Plotting...")
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })
        plt.rc("font", size=12)
        fig, ax = plt.subplots(1, 1)
        sinter.plot_error_rate(
            ax=ax,
            stats=results,
            x_func=lambda stat: stat.json_metadata['p'],
            group_func=lambda stat: {'color': colors[stat.json_metadata['circidx']], 
                                     'linestyle': ':' if stat.json_metadata['rounds'] == 1 else '-',
                                     'label': "_nolegend_" if stat.json_metadata['rounds'] == 1 else stat.json_metadata['circuit']},
            failure_units_per_shot_func=lambda stat: stat.json_metadata['rounds'],
            failure_values_func=lambda _: 2*k
        )
        ax.loglog(PS, PS, color='gray', linestyle='--')
        # ax.set_ylim(5e-3, 5e-2)
        ax.set_xlim(PS[0], PS[-1])
        ax.loglog()
        ax.set_title(f"{'' if CODE_NAME is None else ' ' + CODE_NAME} with Circuit Noise")
        ax.set_xlabel("Physical error rate")
        ax.set_ylabel("Logical error rate per round per logical")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'plot_{IDENTIFIER}.pdf')

        with open(f"data_{IDENTIFIER}.csv", "w") as f:
            print(sinter.CSV_HEADER, file=f)
            for s in results:
                print(s.to_csv_line(), file=f)

    elif TYPE == "codes":
        """
        Fix rounds at [1, d], fix a circuit given by `CIRCUIT`.
        Iterate over different choices of codes.
        """
        print("=== TYPE: codes ===")
        IDENTIFIER = f"{TYPE}_{CIRCUIT}_{NUM_SHOTS}s"

        tasks = []
        for code_idx, code_param in enumerate(CODES):
            code = MirrorCode(*code_param)
            CODE_NAME = NAMES[code_idx]
            NOISE_MODEL_NAME = 'phenom' if CIRCUIT == 'phenom' else 'SI1000'
            n, k, d = [int(x) for x in CODE_NAME.split('_')]
            CODE_NAME = f'[[{n}, {k}, {d}]]'
            ROUND_CHOICES = list(set([d]))

            stabilizers = code.get_stim_tableau()
            benchmarker = StabilizerCode(stabilizers, verbose=False, name=CODE_NAME)
            print(f"Working on [[{code.get_n()}, {code.get_k()}]] code...")

            print(f"Making {CIRCUIT} syndrome extraction circuits...")
            circ_func = None
            if CIRCUIT == 'bare':
                print("Bare syndrome extraction...")
                circ_func = code.bare_ancilla_sec
            elif CIRCUIT == 'loop':
                print("Loop syndrome extraction...")
                circ_func = code.loop_flag_sec
            elif CIRCUIT == 'css':
                print("CSS-FT syndrome extraction...")
                circ_func = code.ft_for_w6_css_sec
            elif CIRCUIT == 'ft':
                print("FT syndrome extraction...")
                circ_func = code.ft_for_w6_sec
            elif CIRCUIT == 'superdense':
                print("Superdense syndrome extraction...")
                circ_func = code.superdense_sec
            else:
                raise SyntaxError(f"Invalid circuit type {CIRCUIT}")

            for nrd in ROUND_CHOICES:
                for i in range(len(PS)):
                    circuit = circ_func(
                                    noise=make_noise_model(NOISE_MODEL_NAME, PS[i]),
                                    num_rounds=nrd
                                )

                    tasks.append(sinter.Task(
                        circuit=circuit,
                        decoder='tesseract',
                        json_metadata={"p": PS[i], 
                                       "decoder": 'tesseract', 
                                       "rounds": nrd, 
                                       "code": fr'$[[{n},{k},{d}]]$',
                                       "codeidx": code_idx
                                       },
                    ))
            
        print("Collecting...")
        results = sinter.collect(
            num_workers=12,
            tasks=tasks,
            max_shots=NUM_SHOTS,
            decoders=['tesseract'],
            custom_decoders=decoder_dict,
            print_progress=True,
        )
        print("Done.")

        colors = plt.get_cmap('tab10').colors[:len(CODES)]

        # Add to the plot
        print("Plotting...")
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })
        plt.rc("font", size=12)
        fig, ax = plt.subplots(1, 1)
        sinter.plot_error_rate(
            ax=ax,
            stats=results,
            x_func=lambda stat: stat.json_metadata['p'],
            group_func=lambda stat: {'color': colors[stat.json_metadata['codeidx']], 
                                    'linestyle': ':' if stat.json_metadata['rounds'] == 1 else '-',
                                    'label': "_nolegend_" if stat.json_metadata['rounds'] == 1 else stat.json_metadata['code']},
            failure_units_per_shot_func=lambda stat: stat.json_metadata['rounds'],
            failure_values_func=lambda _: 2*k
        )
        ax.loglog(PS, PS, color='gray', linestyle='--')
        # ax.set_ylim(5e-3, 5e-2)
        ax.set_xlim(PS[0], PS[-1])
        ax.loglog()
        ax.set_title(f"Mirror Codes with Circuit Noise")
        ax.set_xlabel("Physical error rate")
        ax.set_ylabel("Logical error rate per round per logical")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'plot_{IDENTIFIER}.pdf')

        with open(f"data_{IDENTIFIER}.csv", "w") as f:
            print(sinter.CSV_HEADER, file=f)
            for s in results:
                print(s.to_csv_line(), file=f)

    else:
        raise SyntaxError(f"Invalid benchmarking type {TYPE}")



