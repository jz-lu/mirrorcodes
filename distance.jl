#=
distance.jl

Compute the distance of a stabilizer code using Stim and PySAT.
=#
using PythonCall
using Printf

const stim = pyimport("stim")
const _rc2 = pyimport("pysat.examples.rc2")
const _formula = pyimport("pysat.formula")
const RC2 = _rc2.RC2
const WCNF = _formula.WCNF

"""
    make_code(stabilizers)

Given a list of `stim.PauliString` stabilizers, return
`(stabilizers, observable_xs, observable_zs)`. The logical operators are
deduced from the completed tableau.
"""
function make_code(stabilizers)
    completed = stim.Tableau.from_stabilizers(
        stabilizers; allow_redundant=true, allow_underconstrained=true
    )

    obs_indices = Int[]
    for k in 0:PythonCall.pylen(completed)-1
        z_out = completed.z_output(k)
        if !(z_out in stabilizers)
            push!(obs_indices, k)
        end
    end

    observable_xs = [completed.x_output(k) for k in obs_indices]
    observable_zs = [completed.z_output(k) for k in obs_indices]
    return stabilizers, observable_xs, observable_zs
end

"""
    css_make_circuit(stabilizers, logical_paulis, obs_type::AbstractString)

Build a Stim circuit for CSS distance. `obs_type` must be `"X"` or `"Z"`.
Returns a `stim.Circuit`.
"""
function css_make_circuit(stabilizers, logical_paulis, obs_type::AbstractString)
    @assert obs_type == "X" || obs_type == "Z" "obs_type must be \"Z\" or \"X\""
    num_qubits = Int(PythonCall.pylen(stabilizers[1]))
    circuit = stim.Circuit()
    stab_record_step = 1

    # Measure relevant stabilizers once
    for s in stabilizers
        has_type = PythonCall.pylen(s.pauli_indices(obs_type)) > 0
        if has_type
            circuit.append("MPP", stim.target_combined_paulis(s))
            stab_record_step += 1
        end
    end

    # Tag logicals as observables
    for (k, obs) in enumerate(logical_paulis)
        Xp, Zp = stim.PauliString.to_numpy(obs)
        X = pyconvert(Vector{Bool}, Xp)
        Z = pyconvert(Vector{Bool}, Zp)
        stim_obs = PythonCall.PyList()
        for (i0, (x, z)) in enumerate(zip(X, Z))
            i = i0 - 1
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

    err_type = obs_type == "Z" ? "X" : "Z"
    circuit.append("$(err_type)_ERROR", collect(0:num_qubits-1), 1e-3)

    # Re-measure stabilizers and add detectors
    for s in stabilizers
        has_type = PythonCall.pylen(s.pauli_indices(obs_type)) > 0
        if has_type
            circuit.append("MPP", stim.target_combined_paulis(s))
            circuit.append("DETECTOR", [stim.target_rec(-stab_record_step), stim.target_rec(-1)])
        end
    end

    # Re-include logical observables
    for (k, obs) in enumerate(logical_paulis)
        Xp, Zp = stim.PauliString.to_numpy(obs)
        X = pyconvert(Vector{Bool}, Xp)
        Z = pyconvert(Vector{Bool}, Zp)
        stim_obs = PythonCall.PyList()
        for (i0, (x, z)) in enumerate(zip(X, Z))
            i = i0 - 1
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

"""
    stab_make_circuit(stabilizers, logical_paulis)

Build a Stim circuit for general (non-CSS) distance. Returns a `stim.Circuit`.
"""
function stab_make_circuit(stabilizers, logical_paulis)
    num_qubits = Int(PythonCall.pylen(stabilizers[1]))
    num_stabilizers = length(stabilizers)
    stab_record_step = num_stabilizers + 1
    circuit = stim.Circuit()

    # Declare measured stabilizers
    for s in stabilizers
        circuit.append("MPP", stim.target_combined_paulis(s))
    end

    # Tag logicals as observables
    for (k, obs) in enumerate(logical_paulis)
        Xp, Zp = stim.PauliString.to_numpy(obs)
        X = pyconvert(Vector{Bool}, Xp)
        Z = pyconvert(Vector{Bool}, Zp)
        stim_obs = PythonCall.PyList()
        for (i0, (x, z)) in enumerate(zip(X, Z))
            i = i0 - 1
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

    # Depolarizing noise on all qubits
    circuit.append("DEPOLARIZE1", collect(0:num_qubits-1), 1e-3)

    # Detect stabilizer violations
    for s in stabilizers
        circuit.append("MPP", stim.target_combined_paulis(s))
        circuit.append("DETECTOR", [stim.target_rec(-stab_record_step), stim.target_rec(-1)])
    end

    # Detect logical anticommutes
    for (k, obs) in enumerate(logical_paulis)
        Xp, Zp = stim.PauliString.to_numpy(obs)
        X = pyconvert(Vector{Bool}, Xp)
        Z = pyconvert(Vector{Bool}, Zp)
        stim_obs = PythonCall.PyList()
        for (i0, (x, z)) in enumerate(zip(X, Z))
            i = i0 - 1
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

"""
    code_distance(stim_stab_tableau; is_css::Bool=false, verbose::Bool=true) -> Int

Compute the code distance. `stim_stab_tableau` is a list of `stim.PauliString`.
If `is_css` is true, runs the faster CSS pipeline.
"""
function code_distance(stim_stab_tableau; is_css::Bool=false, verbose::Bool=true)::Int
    stabilizers, obs_xs, obs_zs = make_code(stim_stab_tableau)
    r = length(stabilizers)
    k = length(obs_xs)
    if verbose
        println("Code is [[r, k]] = [[$r, $k]]")
    end

    if is_css
        t0 = time()
        Xcirc = css_make_circuit(stabilizers, obs_xs, "X")
        Xwcnf_str = Xcirc.shortest_error_sat_problem(; format="WDIMACS")
        Xwcnf = WCNF(; from_string=Xwcnf_str)
        t1 = time()
        if verbose
            @printf("X-problem created in %.3fs\n", t1 - t0)
        end

        rc2 = RC2(Xwcnf)
        rc2.compute()
        Xdist = Int(rc2.cost)
        rc2.delete()
        if verbose
            println("X-distance = $Xdist")
        end

        t2 = time()
        Zcirc = css_make_circuit(stabilizers, obs_zs, "Z")
        Zwcnf_str = Zcirc.shortest_error_sat_problem(; format="WDIMACS")
        Zwcnf = WCNF(; from_string=Zwcnf_str)
        t3 = time()
        if verbose
            @printf("Z-problem created in %.3fs\n", t3 - t2)
        end

        rc2z = RC2(Zwcnf)
        rc2z.compute()
        Zdist = Int(rc2z.cost)
        rc2z.delete()
        if verbose
            println("Z-distance = $Zdist")
        end

        t4 = time()
        dist = min(Zdist, Xdist)
        if verbose
            println("Distance = $dist")
            @printf("Z-distance calculated in %.3fs\n", t4 - t3)
            @printf("Distance calculated in %.3fs\n", t4 - t1)
        end
        return dist
    else
        obs = vcat(obs_xs, obs_zs)
        @assert k == length(obs_zs) "There must be the same number of logical X and Z operators"
        t0 = time()
        Zcirc = stab_make_circuit(stabilizers, obs)
        Zwcnf_str = Zcirc.shortest_error_sat_problem(; format="WDIMACS")
        t1 = time()
        if verbose
            @printf("Problem created in %.3fs\n", t1 - t0)
        end

        wcnf = WCNF(; from_string=Zwcnf_str)
        rc2 = RC2(wcnf)
        rc2.compute()
        dist = Int(rc2.cost)
        rc2.delete()
        t2 = time()
        if verbose
            @printf("Problem solved in %.3fs\n", t2 - t1)
            println("Code distance = $dist")
        end
        return dist
    end
end
