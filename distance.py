"""
`distance.py`
Code file to calculate the distance of a LDPC stabilizer code.
The code need not be CSS. If it is CSS, we use `CSS_make_circuit()`
to make it go much faster. If it is not CSS, we use `stab_make_circuit()`
which is significantly slower but works.

We define a code to be CSS if every check is either pure X or pure Z.
"""
import time
import stim
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import numpy as np

def make_code(stabilizers):
    """
    Takes a list of Paulis and outputs the stabilizers and logical operators.

    Params:
        * stabilizers (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
    
    Returns:
        * (stabilizers, observable_xs, observable_zs). The first is just the input. The other two
          are the k logical X's and k logical Z's of the code, where k is the logical dimension of the code.


    This function adapted from a StackExchange answer by Craig Gidney at
    https://quantumcomputing.stackexchange.com/questions/37289/compute-the-exact-minimum-distance-of-a-qecc-with-integer-linear-programming-met
    """
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
    observable_xs: list[stim.PauliString] = [
        completed_tableau.x_output(k)
        for k in obs_indices
    ]
    observable_zs: list[stim.PauliString] = [
        completed_tableau.z_output(k)
        for k in obs_indices
    ]

    return stabilizers, observable_xs, observable_zs

def CSS_make_circuit(stabilizers, logical_paulis, obs_type):
    """
    Make a distance computation circuit for a CSS code.
    This circuit alone does not solve the distance finding problem. 
    You'll have to run it twice --- once for Z-type stabs/logicals and once for X-type.

    Input in a CSS stabilizer code in stim representation, a set of logical Paulis which are 
    all 'X'-type ('Z'-type) and set `obs_type` to 'X' ('Z').
    
    Params:
        * stabilizers (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
        * logical_paulis (list[stim.PauliString]): logical paulis list in stim form.
        * obs_type (str): either 'Z' or 'X'. Specifies whether the logicals are X-type or Z-type.
    
    Returns:
        * stim.Circuit representing an error simulation and detection circuit for the code.
    
    Adapted from code written by Craig Gidney on StackExchange.
    https://quantumcomputing.stackexchange.com/questions/37289/compute-the-exact-minimum-distance-of-a-qecc-with-integer-linear-programming-met
    """
    assert obs_type in ['Z', 'X'], f"obs_type should be 'Z' or 'X', but is {obs_type}"
    num_qubits = len(stabilizers[0])
    circuit = stim.Circuit()
    stab_record_step = 1

    for stabilizer in stabilizers:
        if stabilizer.pauli_indices(obs_type):
            circuit.append("MPP", stim.target_combined_paulis(stabilizer))
            stab_record_step += 1

    for k, observable in enumerate(logical_paulis):
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = []
        for i, (x, z) in enumerate(zip(X_part, Z_part)):
            if x and z:
                stim_obs.append(stim.target_y(i))
            elif x:
                stim_obs.append(stim.target_x(i))
            elif z:
                stim_obs.append(stim.target_z(i))
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k)

    ERROR_TYPE = 'X' if obs_type == 'Z' else 'Z'

    circuit.append(f"{ERROR_TYPE}_ERROR", range(num_qubits), 1e-3)

    for stabilizer in stabilizers:
        if stabilizer.pauli_indices(obs_type):
            circuit.append('MPP', stim.target_combined_paulis(stabilizer))
            circuit.append('DETECTOR', [stim.target_rec(-stab_record_step), stim.target_rec(-1)])
    
    for k, observable in enumerate(logical_paulis):
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = []
        for i, (x, z) in enumerate(zip(X_part, Z_part)):
            if x and z:
                stim_obs.append(stim.target_y(i))
            elif x:
                stim_obs.append(stim.target_x(i))
            elif z:
                stim_obs.append(stim.target_z(i))
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k)

    return circuit

def stab_make_circuit(stabilizers, logical_paulis):
    """
    Make a distance computation circuit for a general, possibly non-CSS code.
    The upside is that this code will work for all stabilizer codes. 
    The downside is that it is significantly slower than `CSS_make_circuit()`.
    For example, for the [[144, 12, 12]] BB code, `CSS_make_circuit()` completes in
    about 2 minutes, whereas this function runs for much more than 12 hours.
    Nonetheless, it's the fastest we're able to get it in terms of non-CSS codes.
    Unlike `CSS_make_circuit()`, you will put in all your logical Paulis at once.

    Params:
        * stabilizers (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
        * logical_paulis (list[stim.PauliString]): logical paulis list in stim form.
    
    Returns:
        * stim.Circuit representing an error simulation and detection circuit for the code.
    
    Adapted from code written by Craig Gidney on StackExchange, with his advice/support
    on a separate StackExchange post.
    https://quantumcomputing.stackexchange.com/questions/37289/compute-the-exact-minimum-distance-of-a-qecc-with-integer-linear-programming-met
    https://quantumcomputing.stackexchange.com/questions/43988/stim-for-the-distance-calculation-for-an-arbitrary-stabilizer-code
    """
    num_qubits = len(stabilizers[0])
    num_stabilizers = len(stabilizers)
    stab_record_step = num_stabilizers + 1
    circuit = stim.Circuit()

    # Declare that we will be measuring the logical paulis and stabilizers
    for stabilizer in stabilizers:
        circuit.append('MPP', stim.target_combined_paulis(stabilizer))

    for k, observable in enumerate(logical_paulis):
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = []
        for i, (x, z) in enumerate(zip(X_part, Z_part)):
            if x and z:
                stim_obs.append(stim.target_y(i))
            elif x:
                stim_obs.append(stim.target_x(i))
            elif z:
                stim_obs.append(stim.target_z(i))
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k)

    # Execute a depolarizing noise model, which includes all possible Pauli strings
    circuit.append('DEPOLARIZE1', range(num_qubits), 1e-3)

    # Check whether the error commutes with the stabilizers or not
    # This is done by detecting any stabilizers which measure to -1 now instead of +1
    for k, stabilizer in enumerate(stabilizers):
        circuit.append('MPP', stim.target_combined_paulis(stabilizer))
        circuit.append('DETECTOR', [stim.target_rec(-stab_record_step), stim.target_rec(-1)])

    # Check whether the error anticommutes with at least one logical Pauli or not
    # This is done by detecting any logical Paulis which measure to -1 instead of +1
    for k, observable in enumerate(logical_paulis):
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = []
        for i, (x, z) in enumerate(zip(X_part, Z_part)):
            if x and z:
                stim_obs.append(stim.target_y(i))
            elif x:
                stim_obs.append(stim.target_x(i))
            elif z:
                stim_obs.append(stim.target_z(i))
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k)

    return circuit

def distance(stim_stab_tableau, IS_CSS = False, verbose = True):
    """
    Calculate the distance of a stabilizer code.

    Params:
        * stim_stab_tableau (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
        * IS_CSS (bool): flag for whether to use the CSS decoder or not. Turn this on if every check in the tableau
          is pure X's or pure Z's. It will make the distance calculation much, much faster.
        * verbose (bool): flag on whether or not to include print statements on time taken and distance calculated.

    Returns:
        * Distance of the code.

    Function adapted from code written by Craig Gidney on a StackExchange post.
    https://quantumcomputing.stackexchange.com/questions/37289/compute-the-exact-minimum-distance-of-a-qecc-with-integer-linear-programming-met
    """
    stabilizers, obs_xs, obs_zs = make_code(stim_stab_tableau)
    r = len(stabilizers); k = len(obs_xs) # r = num stabs = n - k
    if verbose:
        print(f"Code is [[r, k]] = [[{r}, {k}]]")
    dist = -1

    if IS_CSS:
        t0 = time.monotonic()
        Xcircuit = CSS_make_circuit(stabilizers, obs_xs, obs_type='X')
        Xwcnf_string = Xcircuit.shortest_error_sat_problem(format='WDIMACS')
        Xwcnf = WCNF(from_string=Xwcnf_string)
        Xdist = -1
        t1 = time.monotonic()
        if verbose:
            print(f"X-problem created in {t1 - t0:0.3f}s")

        with RC2(Xwcnf) as rc2:
            rc2.compute()
            Xdist = rc2.cost
            if verbose:
                print(f"X-distance = {Xdist}")
        t2 = time.monotonic()
        if verbose:
            print(f"X-distance calculated in {t2 - t1:0.3f}s")

        t3 = time.monotonic()
        Zcircuit = CSS_make_circuit(stabilizers, obs_zs, obs_type='Z')
        Zwcnf_string = Zcircuit.shortest_error_sat_problem(format='WDIMACS')
        Zwcnf = WCNF(from_string=Zwcnf_string)
        Zdist = -1
        t4 = time.monotonic()
        if verbose:
            print(f"Z-problem created in {t4 - t3:0.3f}s")

        with RC2(Zwcnf) as rc2:
            rc2.compute()
            Zdist = rc2.cost
            if verbose:
                print(f"Z-distance = {Zdist}")
        t5 = time.monotonic()
        dist = min(Zdist, Xdist)
        if verbose:
            print(f"Distance = {dist}")
            print(f"Z-distance calculated in {t5 - t4:0.3f}s")
            print(f"Distance calculated in {t5 - t1:0.3f}s")

    else:
        obs = [*obs_xs, *obs_zs]
        t0 = time.monotonic()
        assert k == len(obs_zs), f"There should always be the same number of logical X's and Z's"
        Zcircuit = stab_make_circuit(stabilizers, obs)
        Zwcnf_string = Zcircuit.shortest_error_sat_problem(format='WDIMACS')
        dist = 0
        t1 = time.monotonic()
        if verbose:
            print(f"Problem created in {t1 - t0:0.3f}s")

        wcnf = WCNF(from_string=Zwcnf_string)
        with RC2(wcnf) as rc2:
            rc2.compute()
            dist = rc2.cost
        t2 = time.monotonic()
        if verbose:
            print(f"Problem solved in {t2 - t1:0.3f}s")
            print(f"Code distance = {dist}")

    return int(dist)
