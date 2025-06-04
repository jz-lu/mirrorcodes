"""
`distance.py`
Code file to calculate the distance of a LDPC stabilizer code.

This calculation can either be done exactly or approximately (upper bound given).
The approximate version is faster and is done by random error models.

TODO: make code faster in case the code is CSS
"""
import time
import stim
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import numpy as np

def is_CSS(stabs):
    pass

def make_code(stabilizers) -> tuple[list[stim.PauliString], list[stim.PauliString], list[stim.PauliString]]:

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

# def make_circuit(stabilizers: list[stim.PauliString],
#                  logical_paulis: list[stim.PauliString]) -> stim.Circuit:
#     num_qubits = len(stabilizers[0])
#     num_stabilizers = len(stabilizers)
#     num_logicals = len(logical_paulis)
#     stab_record_step = num_logicals + num_stabilizers + 1
#     logical_record_step = num_stabilizers + num_logicals + 1

#     circuit = stim.Circuit()

#     # Declare that we will be measuring the logical paulis and stabilizers
#     for stabilizer in stabilizers:
#         circuit.append('MPP', stim.target_combined_paulis(stabilizer))
#     for k, observable in enumerate(logical_paulis):
#         circuit.append("MPP", stim.target_combined_paulis(observable))

#     # Execute a depolarizing noise model, which includes all possible Pauli strings
#     circuit.append('DEPOLARIZE1', range(num_qubits), 1e-3)

#     # Check whether the error commutes with the stabilizers or not
#     # This is done by detecting any stabilizers which measure to -1 now instead of +1
#     for k, stabilizer in enumerate(stabilizers):
#         circuit.append('MPP', stim.target_combined_paulis(stabilizer))
#         circuit.append('DETECTOR', [stim.target_rec(-stab_record_step), stim.target_rec(-1)])

#     # Check whether the error anticommutes with at least one logical Pauli or not
#     # This is done by detecting any logical Paulis which measure to -1 instead of +1
#     for k, observable in enumerate(logical_paulis):
#         circuit.append("MPP", stim.target_combined_paulis(observable))
#         circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-logical_record_step), stim.target_rec(-1)], k)

#     return circuit

def make_circuit(stabilizers: list[stim.PauliString],
                 logical_paulis: list[stim.PauliString]) -> stim.Circuit:
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

def distance(stim_stab_tableau):
    stabilizers, obs_xs, obs_zs = make_code(stim_stab_tableau)
    obs = [*obs_xs, *obs_zs]
    t0 = time.monotonic()
    r = len(stabilizers); k = len(obs_xs) # r = num stabs = n - k
    print(f"Code is [[r, k]] = [[{r}, {k}]]")
    assert k == len(obs_zs), f"There should always be the same number of logical X's and Z's"
    circuit = make_circuit(stabilizers, obs)
    wcnf_string = circuit.shortest_error_sat_problem(format='WDIMACS')
    dist = 0
    t1 = time.monotonic()
    print(f"Problem created in {t1 - t0:0.3f}s")

    wcnf = WCNF(from_string=wcnf_string)
    with RC2(wcnf) as rc2:
        rc2.compute()
        dist = rc2.cost
    t2 = time.monotonic()
    print(f"Problem solved in {t2 - t1:0.3f}s")

    print(f"Code distance = {dist}")
    return dist

# def distance(stim_stab_tableau):
#     stabilizers, obs_xs, obs_zs = make_code(stim_stab_tableau)
#     t0 = time.monotonic()
#     r = len(stabilizers); k = len(obs_xs) # r = num stabs = n - k
#     print(f"Code is [[r, k]] = [[{r}, {k}]]")
#     assert k == len(obs_zs), f"There should always be the same number of logical X's and Z's"
#     circuit_z = make_circuit(stabilizers, obs_zs)
#     wcnf_string_z = circuit_z.shortest_error_sat_problem(format='WDIMACS')
#     circuit_x = make_circuit(stabilizers, obs_xs)
#     wcnf_string_x = circuit_x.shortest_error_sat_problem(format='WDIMACS')
#     dist_z = 0; dist_x = 0
#     t1 = time.monotonic()
#     print(f"Problem created in {t1 - t0:0.3f}s")

#     wcnf_z = WCNF(from_string=wcnf_string_z)
#     with RC2(wcnf_z) as rc2:
#         rc2.compute()
#         dist_z = rc2.cost
#         print(f"Distance against Z-logicals = {dist_z}")
#     t2 = time.monotonic()
#     print(f"Z-Problem solved in {t2 - t1:0.3f}s")

#     t3 = time.monotonic()
#     wcnf_x = WCNF(from_string=wcnf_string_x)
#     with RC2(wcnf_x) as rc2:
#         rc2.compute()
#         dist_x = rc2.cost
#         print(f"Distance against X-logicals = {dist_x}")

#     t4 = time.monotonic()
#     print(f"X-Problem solved in {t4 - t3:0.3f}s")

#     dist = min(dist_z, dist_x)
#     print(f"Code distance = {dist}")
#     return dist
