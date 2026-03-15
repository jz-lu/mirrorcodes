"""
Plot benchmarking results on some mirror codes.
This requires that you provide a CSV file that comes from sinter output, which will be loaded for the plot.

If using --type rounds, then you must specify both flags.
If using --type circuits, then you need only specify which code to use, -n.
If using --type codes, then you need only specify which circuit to use, -c.
"""
import argparse
import numpy as np
import sinter
import matplotlib.pyplot as plt
from benchmark import StabilizerCode, make_noise_model
from mirror import MirrorCode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Benchmark codes under various noise models"
        )

    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="sinter CSV filename, of the form outputted by run_benchmark.py",
    )

    args = parser.parse_args()
    INFO = args.file.split('/')[-1].split('_')[1:]
    FILENAME = args.file
    TYPE = INFO[0]

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

    CODES = [CODE_30_8_4, 
            CODE_36_6_6,
            CODE_48_8_6, 
            CODE_48_4_8, 
            CODE_72_12_6, 
            CODE_72_8_8, 
            CODE_85_8_9, 
            # CODE_90_8_10, 
            CODE_144_12_12]
    NAMES = ['30_8_4',
            '36_6_6',
            '48_8_6',
            '48_4_8',
            '72_12_6',
            '72_8_8',
            '85_8_9',
            # '90_8_10',
            '144_12_12']

    T_LOW = 3.096 # min error rate is 10^-T_LOW
    T_HIGH = 1.69 # max error rate is 10^-T_HIGH
    NUM_PROBS = 5
    NUM_SHOTS = 10_000
    
    PS = np.logspace(-T_LOW, -T_HIGH, NUM_PROBS)

    """
    From here on out, the behavior will change based on what type of benchmarking is being done.
    """

    if TYPE == "rounds":
        """
        Fix a code given by `CODES[idx]`, fix a circuit given by `CIRCUIT`.
        Iterate over different choices of rounds.
        """
        idx = INFO[1]
        CIRCUIT = INFO[2]
        n, k, d = int(INFO[3]), int(INFO[4]), int(INFO[5])
        CODE_NAME_NOBRACK = f"{n}_{k}_{d}"
        CODE_NAME = f"[[{CODE_NAME_NOBRACK}]]"

        if CIRCUIT == "phenom":
            T_LOW = 2
            T_HIGH = 0.8
            NUM_PROBS = 8
            NUM_SHOTS = 20_000
        print("=== TYPE: rounds ===")
        
        # TODO
    
    elif TYPE == "circuits":
        """
        Fix a code given by `CODES[idx]`, fix only 1 round and d rounds of syndrome extraction.
        Iterate over different choices of syndrome extraction circuits.
        """
        print("=== TYPE: circuits ===")

        idx = INFO[1]
        n, k, d = int(INFO[2]), int(INFO[3]), int(INFO[4])
        CODE_NAME_NOBRACK = f"{n}_{k}_{d}"
        CODE_NAME = f"[[{n}, {k}, {d}]]"
        IDENTIFIER = f"{TYPE}_{idx}_{CODE_NAME_NOBRACK}_{NUM_SHOTS}s"

        CIRCUIT_NAMES = ['bare', 'loop', 'superdense', 'css', 'ft']
        FANCY_CIRCUIT_NAMES = ['Bare', 'Loop', 'SD', r'CSS-FT$_6$', r'FT$_6$']
            
        results = sinter.read_stats_from_csv_files(FILENAME)

        colors = plt.get_cmap('tab10').colors[:len(CIRCUIT_NAMES)]
        # Add to the plot
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })
        plt.rc("font", size=12)
        fig, ax = plt.subplots(1, 1)

        print("Plotting...")
        sinter.plot_error_rate(
            ax=ax,
            stats=results,
            x_func=lambda stat: stat.json_metadata['p'],
            group_func=lambda stat: {'color': colors[stat.json_metadata['circidx']], 
                                     'linestyle': ':' if stat.json_metadata['rounds'] == 1 else '-',
                                     'label': "_nolegend_" if stat.json_metadata['rounds'] == 1 else FANCY_CIRCUIT_NAMES[stat.json_metadata['circidx']]},
            failure_units_per_shot_func=lambda stat: stat.json_metadata['rounds'],
            failure_values_func=lambda _: 2*k
        )
        ax.loglog(PS, PS, color='gray', linestyle='--', label=f'$p_L = p$')
        # ax.set_ylim(5e-3, 5e-2)
        ax.set_xlim(PS[0], PS[-1])
        ax.loglog()
        ax.set_title(f"{'' if CODE_NAME is None else ' ' + CODE_NAME} Code")
        ax.set_xlabel(r"Physical error rate ($p$)")
        ax.set_ylabel(r"Logical error rate ($p_L$)")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'customplot_{IDENTIFIER}.pdf')

    elif TYPE == "codes":
        """
        Fix rounds at [1, d], fix a circuit given by `CIRCUIT`.
        Iterate over different choices of codes.
        """
        print("=== TYPE: codes ===")
        CIRCUIT = INFO[1]
        if CIRCUIT == "phenom":
            T_LOW = 2
            T_HIGH = 0.8
            NUM_PROBS = 8
            NUM_SHOTS = 20_000
        IDENTIFIER = f"{TYPE}_{CIRCUIT}_{NUM_SHOTS}s"
        assert NUM_SHOTS == int(INFO[-1].split('.')[0][:-1]), f"Number of shots {int(INFO[-1].split('.')[0][:-1])} doesn't match expected {NUM_SHOTS}"

        results = sinter.read_stats_from_csv_files(FILENAME)

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
            failure_values_func=lambda stat: 2*int(stat.json_metadata['code'][3:-3].split(',')[1]) # 2*k
        )
        ax.loglog(PS, PS, color='gray', linestyle='--', label=f'$p_L = p$')
        # ax.set_ylim(5e-3, 5e-2)
        ax.set_xlim(PS[0], PS[-1])
        ax.loglog()
        # ax.set_title(f"Mirror Codes with Circuit Noise")
        ax.set_xlabel(r"Physical error rate ($p$)")
        ax.set_ylabel(r"Logical error rate ($p_L$)")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'customplot_{IDENTIFIER}.pdf')

    else:
        raise SyntaxError(f"Invalid benchmarking type {TYPE}")



