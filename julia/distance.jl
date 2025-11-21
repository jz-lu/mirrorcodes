#=
distance.jl
Code file to calculate the distance of a LDPC stabilizer code.
The code need not be CSS. If it is CSS, we use `CSS_make_circuit()`
to make it go much faster. If it is not CSS, we use `stab_make_circuit()`
which is significantly slower but works.

We define a code to be CSS if every check is either pure X or pure Z.
=#
using PythonCall

const stim = pyimport("stim")
const RC2 = pyimport("pysat.examples.rc2").RC2
const WCNF = pyimport("pysat.formula").WCNF

#=
Takes a list of Paulis and outputs the stabilizers and logical operators.

Params:
    * stabilizers (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.

Returns:
    * (stabilizers, observable_xs, observable_zs). The first is just the input. The other two
      are the k logical X's and k logical Z's of the code, where k is the logical dimension of the code.
=#
function make_code(stabilizers)
    completed_tableau = stim.Tableau.from_stabilizers(
        stabilizers;
        allow_redundant = true,
        allow_underconstrained = true,
    )
    obs_indices = [k for k in 0:Int(completed_tableau.__len__() - 1)
        if !(completed_tableau.z_output(k) in stabilizers)]
    observable_xs = [completed_tableau.x_output(k) for k in obs_indices]
    observable_zs = [completed_tableau.z_output(k) for k in obs_indices]
    return stabilizers, observable_xs, observable_zs
end

#=
Make a distance computation circuit for a CSS code.
This circuit alone does not solve the distance finding problem.
You'll have to run it twice, once for Z-type stabs/logicals and once for X-type.

Input in a CSS stabilizer code in stim representation, a set of logical Paulis which are
all 'X'-type ('Z'-type) and set `obs_type` to 'X' ('Z').

Params:
    * stabilizers (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
    * logical_paulis (list[stim.PauliString]): logical paulis list in stim form.
    * obs_type (str): either 'Z' or 'X'. Specifies whether the logicals are X-type or Z-type.

Returns:
    * stim.Circuit representing an error simulation and detection circuit for the code.
=#
function CSS_make_circuit(stabilizers, logical_paulis, obs_type::AbstractString)
    @assert obs_type in ("Z", "X") "obs_type should be 'Z' or 'X', but is $obs_type"
    num_qubits = Int(stabilizers[1].__len__())
    circuit = stim.Circuit()
    stab_record_step = 1

    for stabilizer in stabilizers
        if !(stabilizer.pauli_indices(obs_type) == ())
            circuit.append("MPP", stim.target_combined_paulis(stabilizer))
            stab_record_step += 1
        end
    end

    for (k, observable) in enumerate(logical_paulis)
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = PythonCall.PyList()
        for i in 0:num_qubits-1
            x = Bool(X_part[i]); z = Bool(Z_part[i])
            if x && z
                stim_obs.append(stim.target_y(i))
            elseif x
                stim_obs.append(stim.target_x(i))
            elseif z
                stim_obs.append(stim.target_z(i))
            end
        end
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k - 1)
    end

    ERROR_TYPE = obs_type == "Z" ? "X" : "Z"
    circuit.append("$(ERROR_TYPE)_ERROR", 0:num_qubits - 1, 1e-3)

    for stabilizer in stabilizers
        if !(stabilizer.pauli_indices(obs_type) == ())
            circuit.append("MPP", stim.target_combined_paulis(stabilizer))
            circuit.append("DETECTOR", [stim.target_rec(-stab_record_step), stim.target_rec(-1)])
        end
    end

    for (k, observable) in enumerate(logical_paulis)
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = PythonCall.PyList()
        for i in 0:num_qubits - 1
            x = Bool(X_part[i]); z = Bool(Z_part[i])
            if x && z
                stim_obs.append(stim.target_y(i))
            elseif x
                stim_obs.append(stim.target_x(i))
            elseif z
                stim_obs.append(stim.target_z(i))
            end
        end
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k - 1)
    end

    return circuit
end

#=
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
=#
function stab_make_circuit(stabilizers, logical_paulis)
    num_qubits = Int(stabilizers[1].__len__())
    num_stabilizers = length(stabilizers)
    stab_record_step = num_stabilizers + 1
    circuit = stim.Circuit()

    for stabilizer in stabilizers
        circuit.append("MPP", stim.target_combined_paulis(stabilizer))
    end

    for (k, observable) in enumerate(logical_paulis)
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = PythonCall.PyList()
        for i in 0:num_qubits-1
            x = Bool(X_part[i]); z = Bool(Z_part[i])
            if x && z
                stim_obs.append(stim.target_y(i))
            elseif x
                stim_obs.append(stim.target_x(i))
            elseif z
                stim_obs.append(stim.target_z(i))
            end
        end
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k-1)
    end

    circuit.append("DEPOLARIZE1", 0:num_qubits-1, 1e-3)

    for (k, stabilizer) in enumerate(stabilizers)
        circuit.append("MPP", stim.target_combined_paulis(stabilizer))
        circuit.append("DETECTOR", [stim.target_rec(-stab_record_step), stim.target_rec(-1)])
    end

    for (k, observable) in enumerate(logical_paulis)
        X_part, Z_part = stim.PauliString.to_numpy(observable)
        stim_obs = PythonCall.PyList()
        for i in 0:num_qubits-1
            x = Bool(X_part[i]); z = Bool(Z_part[i])
            if x && z
                stim_obs.append(stim.target_y(i))
            elseif x
                stim_obs.append(stim.target_x(i))
            elseif z
                stim_obs.append(stim.target_z(i))
            end
        end
        circuit.append("OBSERVABLE_INCLUDE", stim_obs, k-1)
    end

    return circuit
end

#=
Calculate the distance of a stabilizer code.

Params:
    * stim_stab_tableau (list[stim.PauliString]): stabilizer tableau given as a list of Paulis in stim form.
    * IS_CSS (bool): flag for whether to use the CSS decoder or not. Turn this on if every check in the tableau
      is pure X's or pure Z's. It will make the distance calculation much, much faster.
    * verbose (bool): flag on whether or not to include print statements on time taken and distance calculated.

Returns:
    * Distance of the code.
=#
function distance(stim_stab_tableau; IS_CSS::Bool=false, verbose::Bool=true)
    stabilizers, obs_xs, obs_zs = make_code(stim_stab_tableau)
    r = length(stabilizers); k = length(obs_xs)
    verbose && println("Code is [[r, k]] = [[", r, ", ", k, "]]")
    dist = -1

    if IS_CSS
        Xcircuit = CSS_make_circuit(stabilizers, obs_xs, "X")
        Xwcnf_string = Xcircuit.shortest_error_sat_problem(format="WDIMACS")
        Xwcnf = WCNF(from_string = Xwcnf_string)
        Xdist = -1
        if verbose; println("X-problem created") end
        rc2x = RC2(Xwcnf); rc2x.compute(); Xdist = Int(rc2x.cost)
        verbose && println("X-distance = ", Xdist)

        Zcircuit = CSS_make_circuit(stabilizers, obs_zs, "Z")
        Zwcnf_string = Zcircuit.shortest_error_sat_problem(format="WDIMACS")
        Zwcnf = WCNF(from_string=Zwcnf_string)
        Zdist = -1
        if verbose; println("Z-problem created") end
        rc2z = RC2(Zwcnf); rc2z.compute(); Zdist = Int(rc2z.cost)

        dist = min(Zdist, Xdist)
        if verbose
            println("Z-distance = ", Zdist)
            println("Distance = ", dist)
        end
    else
        obs = PythonCall.PyList()
        for o in obs_xs; obs.append(o); end
        for o in obs_zs; obs.append(o); end
        Zcircuit = stab_make_circuit(stabilizers, obs)
        Zwcnf_string = Zcircuit.shortest_error_sat_problem(format="WDIMACS")
        wcnf = WCNF(from_string=Zwcnf_string)
        rc2 = RC2(wcnf); rc2.compute(); dist = Int(rc2.cost)
        if verbose
            println("Problem solved")
            println("Code distance = ", dist)
        end
    end
    return Int(dist)
end
