#=
filter.jl

Main wrapper for iteratively filtering for good codes
that we search through in a clever version of brute force in `search.py`.
The filtering process will be dependent on a fixed n and is a 3-step procedure.

Stage 1) Find all inequivalent codes for a given n. (Equivalence is defined by a series of automorphisms).
         This is the "canonical set". This is extremely fast for each code, but there are a huge number of 
         codes to check. Check the rate (n - rank of matrix) and discard if it zero.
Stage 2) Find all codes which passed stage 1 and whose rate is above a threshold. This is quite fast.
Stage 3) Find all codes which passed stage 2 and whose distance is good. Here, good depends on whether
         we can calculate the distance in a reasonable time. If we can, then the distance * rate
         should be comparable to n. If we cannot, then that's a good sign, and we should keep the code
         for further investigation.
Stage 4) TBD. For good enough codes, we want to be even more fine grained and see decoding, geometric
         properties, etc.

The multistage process is disconnected in the sense that you specify a stage k. 
At the end of each stage, a file is saved containing all the codes which passed
that stage. This code will then assume you have already went through all stages 
up to and including k-1. It will import the file from the previous stage and 
process those codes. Therefore, at each stage the next file saved is substantially
smaller than the previous file saved.
=#
include("constants.jl")

using JLD2
using ArgParse
using Printf

# External APIs expected from your project:
# - find_all_codes(n, Z_wt, X_wt; min_k=3) -> Vector of tuples (group, z0, x0, is_css::Bool, k::Int)
# - HelixCode(group, z0, x0; n::Int, k::Int, is_css::Bool)
# - get_d(code) -> Int

const CodeTuple = Tuple{Any,Any,Any,Bool,Int}                 # (group, z0, x0, is_css, k)
const CodeTupleStage3 = Tuple{Any,Any,Any,Bool,Int,Int,Float64}  # + d, k*d/n

"""
Stage 1 filtering. 

Params:
    * n (int): number of qubits.
    * Z_wt (int): number of elements of Z_0.
    * X_wt (int): number of elements of X_0.
    * min_k (int, optional): Whether to filter stage 1 to not include codes with k < min_k.

Returns:
    * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 1,
"""
function stage1(n::Int, Z_wt::Int, X_wt::Int; min_k::Int=3)::Vector{CodeTuple}
    return find_all_codes(n, Z_wt, X_wt; min_k=min_k)
end

"""
Stage 2 filtering. 

Params:
    * n (int): number of qubits.
    * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
      which passed stage 1.

Returns:
    * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 2.
"""
function stage2(n::Int, codes::Vector{CodeTuple}; verbose::Bool=false)::Vector{CodeTuple}
    passing_codes = CodeTuple[]
    for code_data in codes
        group, z0, x0, is_css, k = code_data
        rate = k / n
        if rate >= RATE_THRESHOLD
            push!(passing_codes, code_data)
            if verbose
                @printf("Added [[%d, %d]] code of rate %.4f\n", n, k, rate)
            end
        end
    end
    return passing_codes
end

# Timeout helper to replace the Python multiprocessing worker.
# Returns (true, value) if finished, or (false, nothing) on timeout.
function with_timeout(f::Function, seconds::Real)
    result_ch = Channel{Any}(1)
    task = @async begin
        try
            put!(result_ch, f())
        catch e
            put!(result_ch, e)
        end
    end
    timeout_ch = Channel{Symbol}(1)
    timer = Timer(_ -> put!(timeout_ch, :timeout), seconds)
    sel = Base.selectch((result_ch, identity), (timeout_ch, identity))
    close(timer)
    if sel[1] == 1
        val = sel[2]
        if val isa Exception
            rethrow(val)
        else
            return true, val
        end
    else
        Base.throwto(task, InterruptException())
        return false, nothing
    end
end

"""
Stage 3 filtering. 

Params:
    * n (int): number of qubits.
    * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
      which passed stage 2.
    * t (int): how many seconds you are willing to spend on the distance
      calculation.

Returns:
    * list of helix codes in (group, Z_0, X_0, k, d, k*d/n) form which pass
      stage 3. (d -> -1, k*d/n -> -1 if distance failed to calculate in time t)
"""
function stage3(n::Int, codes::Vector{CodeTuple}; t::Int=3, verbose::Bool=false)::Vector{CodeTupleStage3}
    passing_codes = CodeTupleStage3[]
    seen = Set{Tuple{Int,Int,Bool}}()
    for code_data in codes
        group, z0, x0, is_css, k = code_data
        code = HelixCode(group, z0, x0; n=n, k=k, is_css=is_css)
        d = -1

        ok, dval = with_timeout(() -> get_d(code), t)
        if ok
            d = Int(dval)
        else
            if verbose
                css_str = is_css ? "" : "non-"
                println("Distance calculation timed out at $(t)s for $(css_str)CSS [[${n}, ${k}]] code",
                        "\ngroup =\n$group\nz0 =\n$z0\nx0 =\n$x0")
            end
        end

        goodness = d > 0 ? (k * d) / n : -1.0
        goodness_str = d > 0 ? @sprintf(" (GR = %.4f)", goodness) : ""
        if d == -1 || (d >= DISTANCE_THRESHOLD && goodness >= DISTANCE_RATE_THRESHOLD)
            if verbose && !((k, d, is_css) in seen)
                css_str = is_css ? "" : "non-"
                println("[[${n}, ${k}, ${d}]]$(goodness_str) $(css_str)CSS code found")
                if goodness >= 0.9
                    println("*******  Someone has a bright future! *******")
                end
            end
            push!(seen, (k, d, is_css))
            push!(passing_codes, (group, z0, x0, is_css, k, d,
                                  d == -1 ? -1.0 : round((k * d) / n; digits=5)))
        end
        # else:
        #     if verbose:
        #         print(f"[[{n}, {k}, {d}]]{goodness} code is BAD")
    end
    return passing_codes
end

"""
TODO
"""
function stage4(n::Int, codes)
    throw(ErrorException("Stage 4 has not been implemented yet."))
end

# JLD2 helpers to mirror pickle usage.
save_data(path::AbstractString, data) = jldsave(path; data)
load_data(path::AbstractString) = load(path, "data")

function main(args)
    VERBOSE = args["verbose"]
    SAVE_DATA = !get(args, "nosave", false)
    in_directory = args["input"]
    out_directory = get(args, "output", nothing)
    if out_directory === nothing
        out_directory = in_directory
    end
    Z_wt = args["Zweight"]; Z_wt = Z_wt === nothing ? 3 : Z_wt
    X_wt = args["Xweight"]; X_wt = X_wt === nothing ? 3 : X_wt
    t = args["time"]; t = t === nothing ? 3 : t
    mink = args["mink"]; mink = mink === nothing ? 3 : mink
    stages = args["stages"]
    n = args["size"]
    r = get(args, "range", nothing)
    width = args["width"]; width = width === nothing ? 100 : width

    println("Running: n = $n")
    out_data = nothing

    if args["fullsend"]
        println("[Fullsend] Starting stage 1")
        out_data = stage1(n, Z_wt, X_wt; min_k=mink)
        println("Filtered to $(length(out_data)) codes")
        println("[Fullsend] Starting stage 2")
        out_data = stage2(n, out_data)
        println("Filtered to $(length(out_data)) codes")
        println("[Fullsend] Starting stage 3")
        out_data = stage3(n, out_data; t=t, verbose=VERBOSE)
        println("Filtered to $(length(out_data)) codes")

        if SAVE_DATA
            out_file = joinpath(out_directory, get_filename(3, n))
            save_data(out_file, out_data)
        end
    else
        for ch in collect(string(stages))
            stage = parse(Int, string(ch))
            if stage == 1
                out_data = stage1(n, Z_wt, X_wt; min_k=mink)
            else
                in_file = joinpath(in_directory, get_filename(stage - 1, n))
                in_data = load_data(in_file)

                if stage == 2
                    out_data = stage2(n, in_data; verbose=VERBOSE)
                elseif stage == 3
                    if r !== nothing
                        start = min(width * r + 1, length(in_data) + 1)
                        stop  = min(width * (r + 1), length(in_data))
                        in_data = start <= stop ? in_data[start:stop] : eltype(in_data)[]
                    end
                    out_data = stage3(n, in_data; t=t, verbose=VERBOSE)
                elseif stage == 4
                    out_data = stage4(n, in_data)
                end
            end

            if SAVE_DATA
                out_file = joinpath(out_directory, get_filename(stage, n, r))
                save_data(out_file, out_data)
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings(; prog="filter", description="Filter for good helix codes")
    @add_arg_table! s begin
        "--size", "-n"
            arg_type = Int
            required = true
            help = "Number of qubits"
        "--mink", "-k"
            arg_type = Int
            default = 3
        "--Zweight", "-z"
            arg_type = Int
            default = 3
        "--Xweight", "-x"
            arg_type = Int
            default = 3
        "--stages", "-s"
            arg_type = Int
            default = 1
            help = "One or more stages like 1, 2, 3, 4, or combined as 123"
        "--time", "-t"
            arg_type = Int
            default = 3
        "--range", "-r"
            arg_type = Int
        "--width", "-w"
            arg_type = Int
            default = 100
        "--fullsend", "-f"
            action = :store_true
            help = "Run stages 1-3 all at once (don't do this for large n)"
        "--input", "-i"
            arg_type = String
            default = "."
            help = "Location of input files (default ./)"
        "--output", "-o"
            arg_type = String
            help = "Where to write output files (default the same as input directory)"
        "--verbose", "-v"
            action = :store_true
            help = "Include print statements"
        "--nosave"
            action = :store_true
            help = "Don't save files"
    end
    args = parse_args(s)
    main(args)
end
