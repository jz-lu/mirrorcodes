"""
`benchmark.py`
Numerically study the performance of a general stabilizer code.
The code does not have to be CSS.
This has nothing specifically to do with mirror codes!
"""
from util import binary_rank, symp2Pauli, stimify_symplectic
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import stim
import multiprocessing
import time
import sinter


# Models are based on the specs provided by
# https://arxiv.org/pdf/2108.10457
def make_noise_model(p = 0.001, name = 'SD'):
    if name == 'SD': # "SD6" model
        return {
            'p2': p,
            'p1': p,
            'p_init': p,
            'p_meas': p,
            'p_idle': p,
            'p_res_idle': p,
        }
    elif name == 'phenom': # phenomenological model
        return {
            'p_depol': p,
            'p_meas': p/4
        }
    elif name == 'SI1000': # "SI1000" model
        return {
            'p2': p,
            'p1': p / 10,
            'p_init': 2 * p,
            'p_meas': 5 * p,
            'p_idle': p / 10,
            'p_res_idle': 2 * p,
        }
    else:
        raise SyntaxError(f"Invalid name '{name}'")


def benchmark(sec, num_shots = 1000, verbose=False):
        """
        Benchmark the decoding performance of a code under a single set of parameters.

        Params:
            * sec (stim.Circuit): syndrome extraction circuit (SEC) for the stabilizer code.
              The noise model should be pre-incorporated into the SEC.
            * num_shots (int): number of trials to conduct.
        
        Returns:
            * results (dict): 'num_errors': number of shots with errors
                              'num_shots': number of shots total
                              'time_seconds': time it took to decode, in seconds
        """
        import tesseract_decoder
        import tesseract_decoder.tesseract as tesseract
        from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder
        # Prepare tesseract decoder
        dem = sec.detector_error_model()
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=10000,
            no_revisit_dets=True,
            verbose=False,
            det_orders=tesseract_decoder.utils.build_det_orders(
                dem, num_det_orders=1,
                method=tesseract_decoder.utils.DetOrder.DetIndex,
                seed=137),
        )
        tesseract_dec = tesseract_config.compile_decoder()

        # Sample noise
        sampler = sec.compile_detector_sampler()
        dets, obs = sampler.sample(num_shots, separate_observables=True)
        
        # Run decoder on the noise
        num_errors = 0
        start_time = time.time()
        obs_predicted = tesseract_dec.decode_batch(dets)
        end_time = time.time()
        num_errors = np.sum(np.any(obs_predicted != obs, axis=1))

        results = {
            'num_errors': num_errors,
            'num_shots': len(dets),
            'time_seconds': end_time - start_time,
        }

        if verbose:
            print("Tesseract Decoder Stats:")
            print(f"   Number of Errors / num_shots: {results['num_errors']} / {results['num_shots']}")
            print(f"   Time: {results['time_seconds']:.4f} s")

        return results


def estimate_y_eq_x_crossing(xs, ys, *, return_all=False, atol=1e-12, rtol=1e-8):
    """
    Estimate where a curve (xs, ys) crosses the y = x line on a log-log plot.

    Assumes log-log interpolation: treat ly = log(y) as linear between points in lx = log(x).
    Crossing condition y = x  <=>  log(y) - log(x) = 0.

    Returns:
      - (x_cross, y_cross) with y_cross == x_cross
      - or a list of crossings if return_all=True
      - or None if no bracketing sign change is found
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # Need positive, finite points for logs
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    xs, ys = xs[m], ys[m]
    if xs.size < 2:
        raise ValueError("Need at least two positive, finite (x,y) points.")

    # Sort by x (important for bracketing/interpolation)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    lx = np.log(xs)
    ly = np.log(ys)
    f = ly - lx  # zero when y=x

    # Exact/near-exact hits
    hits = np.where(np.isclose(f, 0.0, atol=atol, rtol=rtol))[0]
    crossings = []
    for i in hits:
        crossings.append((xs[i], xs[i]))  # y=x at crossing

    # Bracketed sign changes between consecutive points
    idx = np.where(f[:-1] * f[1:] < 0)[0]
    for i in idx:
        # Linear interpolation in log-space for f
        t = -f[i] / (f[i + 1] - f[i])  # fraction between i and i+1 where f=0
        lx_cross = lx[i] + t * (lx[i + 1] - lx[i])
        x_cross = float(np.exp(lx_cross))
        crossings.append((x_cross, x_cross))

    if not crossings:
        return None

    # Sort by x and return
    crossings.sort(key=lambda p: p[0])
    return crossings if return_all else crossings[0]


class StabilizerCode():
    """
    Class structure for a generic stabilizer code, specified by its stabilizer tableau.
    Uses stim and tesseract_decoder to benchmark the code.
    """
    def __init__(self, stabilizers, verbose=False, name=None):
        """
        Initialization of StabilizerCode by specification of the stabilizers.
        Let the code have n physical qubits and a tableau of r stabilizers.

        Params:
            * stabilizers (list[stim.PauliString]): list of stabilizers in stim Pauli string form.
            * name (str, Optional): name of the code
            * verbose (bool, Optional): if True, prints log of benchmarking
        
        Returns:
            None
        """
        self.stabilizers = stabilizers
        completed_tableau = stim.Tableau.from_stabilizers(
            stabilizers,
            allow_redundant=True,
            allow_underconstrained=True,
        )
        obs_indices = [
            k
            for k in range(len(completed_tableau))
            if completed_tableau.z_output(k) not in stabilizers
        ]
        self.logical_zs = [
            completed_tableau.z_output(k)
            for k in obs_indices
        ]
        self.logical_xs = [
            completed_tableau.x_output(k)
            for k in obs_indices
        ]
        self.logicals = self.logical_zs + self.logical_xs
        self.verbose = verbose
        self.name = name
        self.num_logicals = len(self.logical_zs)

        if self.verbose:
            print(f"Initialized StabilizerCode. Stabilizer tableau:")
            for stab in stabilizers:
                print(stab)
        return
    

    def parallel_benchmark(self, ps, secs, 
                           rounds_choices = [3, 5, 7], num_shots = 1000,
                           plot=False, save_as="./result.jpeg"):
        """
        Benchmark a bunch of error probabilities in parallel, using a syndrome extraction circuit (SEC) with circuit-level noise.
        On input a list of data, 1-qubit, and 2-qubit error probabilities (must be the same size), runs benchmarking on each triple
        of error probabilities in parallel and returns their empirical logical error rates.
        This is done using the sinter package, which splits each benchmarking task into a bunch of individual worker processes and 
        runs these asynchronously in parallel.
        
        Params:
            * secs (list:stim.Circuit): list of syndrome extraction circuits.
            * rounds_choices (list:int): list of choices for number of rounds of syndrome extraction.
            * num_shots (float): number of trials.
            * phenomenological (bool): if True, the SEC will only put data errors (i.e. p_data before each round) and measurement errors.
        
        Returns:
            * physical_error_rates (list:float (num_noise,)) = physical error rate per syndrome extraction cycle
            * logical_error_rates (list:float (num_rounds, num_noise)) = logical error rate per syndrome extraction cycle
        """
        num_round_choices = len(rounds_choices)
        num_noise = len(secs) // num_round_choices

        args = [(sec, num_shots) for sec in secs]
            
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(benchmark, args)
        
        # Currently, `results` is a flattened array. Unflatten before returning.
        # Also, compute the logical error rate per cycle
        logical_error_rates = np.zeros((num_round_choices, num_noise))
        for i, num_rounds in enumerate(rounds_choices):
            for j in range(num_noise):
                logical_error_rates[i, j] = 1 - (1 - results[i*num_noise + j]['num_errors'] / num_shots)**(1/num_rounds)

        if plot:
            # Compute the physical error rate
            physical_error_rates = 1 - (1 - ps)**self.num_logicals

            # Make a plot of the logical versus the physical error rate
            colors = plt.cm.tab10(np.linspace(0, 1, num_round_choices))
            for i, nrd in enumerate(rounds_choices):
                plt.loglog(ps, logical_error_rates[i], color=colors[i], label=nrd, 
                         marker='o', markersize=6, markeredgewidth=2,
                         linestyle='-')
            plt.loglog(ps, physical_error_rates, color='gray', linestyle='--')
            plt.grid()
            plt.grid(which="minor", color="0.9")
            plt.legend(title='Rounds')
            
            plt.xlabel("Physical error rate")
            plt.ylabel("Logical error rate")
            plt.title(f"Logical error rate scaling of the{'' if self.name is None else ' ' + self.name} code ({num_shots} shots)")
            plt.tight_layout()
            plt.savefig(save_as)
        
        return physical_error_rates, logical_error_rates
    
    def sinter_benchmark(self, ps, secs, 
                           rounds_choices = [3, 5, 7], num_shots = 1000,
                           plot=False, save_as="./result.jpeg", verbose=False,
                           phenom=False):
        """
        Benchmark in parallel using the sinter package instead of an in-house solution.
        It probably is wiser to use sinter....
        
        Params:
            * secs (list:stim.Circuit): list of syndrome extraction circuits.
            * rounds_choices (list:int): list of choices for number of rounds of syndrome extraction.
            * num_shots (float): number of trials.
            * plot (bool): produce a pseudothreshold plot.
            * save_as (str): where to save the plot.
            * verbose (bool): whether to print progress/results.
        
        Returns:
            * results (object): sinter object that contains all the results of the simulation.
        """
        import tesseract_decoder
        import tesseract_decoder.tesseract as tesseract
        from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder
        tasks = []

        # decoders = ['tesseract', 'tesseract-long-beam', 'tesseract-short-beam']
        decoders = ['tesseract']
        decoder_dict = make_tesseract_sinter_decoders_dict()
        # # You can also make your own custom Tesseract Decoder to-be-used with Sinter.
        # decoders.append('custom-tesseract-decoder')
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

        for decoder in decoders:
            for i in range(len(ps)):
                circuit = secs[i]

                for nrd in rounds_choices:
                    tasks.append(sinter.Task(
                        circuit=circuit,
                        decoder=decoder,
                        json_metadata={"p": ps[i], "decoder": decoder, "rounds": nrd},
                    ))

        print("Collecting...")
        
        results = sinter.collect(
            num_workers=8,
            tasks=tasks,
            max_shots=num_shots,
            decoders=decoders,
            custom_decoders=decoder_dict,
            print_progress=verbose,
        )

        print("Done")
        
        if plot:
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
                group_func=lambda stat: stat.json_metadata['rounds'],
                failure_units_per_shot_func=lambda stat: stat.json_metadata['rounds'],
            )
            # physical_error_rates = 1 - (1 - ps)**self.num_logicals
            ax.loglog(ps, ps, color='gray', linestyle='--')
            # ax.set_ylim(5e-3, 5e-2)
            ax.set_xlim(ps[0], ps[-1])
            ax.loglog()
            noise_type = "Phenomenological" if phenom else "Circuit" 
            ax.set_title(f"{'' if self.name is None else ' ' + self.name} with {noise_type} Noise")
            ax.set_xlabel("Physical error rate")
            ax.set_ylabel("Logical error rate per round")
            ax.grid(which='major')
            ax.grid(which='minor')
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_as)
        
        if verbose:
            # Print samples as CSV data.
            print(sinter.CSV_HEADER)
            for result in results:
                print(result.to_csv_line())
        
        return results

    def pseudothreshold(self, rounds_choices, physical_error_rates, logical_error_rates):
        """
        Compute the pseudothreshold of the code, using circuit-level noise, for each number of rounds in `round_choices`.
        
        Params:
            * rounds_choices (list:int): list of choices for number of rounds of syndrome extraction.
            * physical_error_rates (list:float (num_noise,)) = physical error rate per syndrome extraction cycle
            * logical_error_rates (list:float (num_rounds, num_noise)) = logical error rate per syndrome extraction cycle
        Returns:
            * pseudothresholds: (list:float): computed pseudothreshold of the code, one for each number of rounds in `round_choices`.
                Returns -1 in each element for which no threshold is found.
        """
        pthrs = np.zeros(len(rounds_choices))
        for i, num_rounds in enumerate(rounds_choices):
            pthr = estimate_y_eq_x_crossing(physical_error_rates, logical_error_rates[i])
            if pthr == None:
                print(f"No pseudothreshold when using {num_rounds} rounds of syndrome extraction.")
                pthrs[i] = -1
            else:
                pthrs[i] = pthr
        return pthrs


