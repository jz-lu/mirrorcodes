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
up to and including k - 1. It will import the file from the previous stage and 
process those codes. Therefore, at each stage the next file saved is substantially
smaller than the previous file saved.
=#
include("constants.jl")
include("helix.jl")
include("search.jl")

using JLD2
using ArgParse
using Printf

#=
Stage 1 filtering. 

Params:
    * n (int): number of qubits.
    * Z_wt (int): number of elements of Z_0.
    * X_wt (int): number of elements of X_0.
    * min_k (int, optional): Whether to filter stage 1 to not include codes with k < min_k.

Returns:
    * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 1,
=#
stage1(n::Int, Z_wt::Int, X_wt::Int; min_k::Int=3) = find_all_codes(n, Z_wt, X_wt; min_k=min_k)

#=
Stage 2 filtering. 

Params:
    * n (int): number of qubits.
    * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
      which passed stage 1.

Returns:
    * list of helix codes in (group, Z_0, X_0, IS_CSS, k) form which pass stage 2.
=#
function stage2(n::Int, codes; verbose::Bool=false)
    passing_codes = typeof(codes)()
    for code_data in codes
        group, z0, x0, _, k = code_data
        rate = k / n
        if rate >= RATE_THRESHOLD
            push!(passing_codes, code_data)
            if verbose
                println("Added [[", n, ", ", k, "]] code of rate ", round(rate, digits=4))
            end
        end
    end
    passing_codes
end

#=
Worker method for slow operation.

Params:
    * queue: something for timeout thread
    * code (tuple): code whose distance we want to find
=#
function worker(f::Function)
    f()
end

# helper: timeout wrapper
function _with_timeout(f::Function, seconds::Real)
    ch = Channel{Any}(1)
    t = @async try
        put!(ch, f())
    catch e
        put!(ch, e)
    end
    tm = Timer(_ -> put!(ch, :timeout), seconds)
    val = take!(ch)
    close(tm)
    if val === :timeout
        Base.throwto(t, InterruptException())
        return false, nothing
    elseif val isa Exception
        throw(val)
    else
        return true, val
    end
end

#=
Stage 3 filtering. 

Params:
    * n (int): number of qubits.
    * codes (list): list of helix codes in (group, Z_0, X_0, IS_CSS, k) form
      which passed stage 2.
    * t (int): how many seconds you are willing to spend on the distance
      calculation.

Returns:
    * list of helix codes in (group, Z_0, X_0, k, d, k * d / n) form which pass
      stage 3. (d -> -1, k * d / n -> -1 if distance failed to calculate in time t)
=#
function stage3(n::Int, codes; t::Int=3, verbose::Bool=false)
    passing_codes = Any[]
    seen = Set{Tuple{Int,Int,Bool}}()
    for code_data in codes
        group, z0, x0, is_css, k = code_data
        code = HelixCode(group, z0, x0; n = n, k = k, is_css = is_css)
        ok, dval = _with_timeout(() -> get_d(code), t)
        d = ok ? Int(dval) : -1

        goodness = k * d / n
        goodness_str = d > 0 ? " (GR = $(round(goodness, digits=4)))" : ""
        if d == -1 || (d >= DISTANCE_THRESHOLD && goodness >= DISTANCE_RATE_THRESHOLD)
            if verbose && !((k, d, is_css) in seen)
                println("[[", n, ", ", k, ", ", d, "]]", goodness_str, " ", is_css ? "" : "non-", "CSS code found")
                if goodness >= 0.9
                    println("*******  Someone has a bright future! *******")
                end
            end
            push!(seen, (k, d, is_css))
            push!(passing_codes, (group, z0, x0, is_css, k, d, d == -1 ? -1.0 : round(k * d / n, digits=5)))
        end
    end
    passing_codes
end

function save_data(path::AbstractString, data) jldsave(path; data) end
function load_data(path::AbstractString) load(path, "data") end

#=
# TODO
Stage 4 has not been implemented yet.
=#
stage4(n::Int, codes) = error("Stage 4 has not been implemented yet.")

function main(args)
    VERBOSE = args["verbose"]
    SAVE_DATA = !get(args, "nosave", false)
    in_directory = args["input"]
    out_directory = get(args, "output", nothing)
    if out_directory === nothing
        out_directory = in_directory
    end
    Z_wt = args["Zweight"]; X_wt = args["Xweight"]
    time = args["time"]; mink = args["mink"]
    stages = args["stages"]
    n = args["size"]
    r = get(args, "range", nothing)
    width = args["width"]
    println("Running: n = ", n)
    out_data = nothing

    if args["fullsend"]
        println("[Fullsend] Starting stage 1")
        out_data = stage1(n, Z_wt, X_wt; min_k=mink)
        println("Filtered to ", length(out_data), " codes")
        println("[Fullsend] Starting stage 2")
        out_data = stage2(n, out_data; verbose=VERBOSE)
        println("Filtered to ", length(out_data), " codes")
        println("[Fullsend] Starting stage 3")
        out_data = stage3(n, out_data; t=time, verbose=VERBOSE)
        println("Filtered to ", length(out_data), " codes")

        if SAVE_DATA
            out_file = joinpath(out_directory, get_filename(3, n))
            save_data(out_file, out_data)
        end
    else
        for stage_char in collect(string(stages))
            stage = parse(Int, string(stage_char))
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
                    out_data = stage3(n, in_data; t=time, verbose=VERBOSE)
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
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings(; prog="filter", description="Filter for good helix codes")
    @add_arg_table! s begin
        "--size", "-n"; arg_type=Int; required=true; help="Number of qubits"
        "--mink", "-k"; arg_type=Int; default=3
        "--Zweight", "-z"; arg_type=Int; default=3
        "--Xweight", "-x"; arg_type=Int; default=3
        "--stages", "-s"; arg_type=Int; default=1
        "--time", "-t"; arg_type=Int; default=3
        "--range", "-r"; arg_type=Int
        "--width", "-w"; arg_type=Int; default=100
        "--fullsend", "-f"; action=:store_true
        "--input", "-i"; arg_type=String; default="."
        "--output", "-o"; arg_type=String
        "--verbose", "-v"; action=:store_true
        "--nosave"; action=:store_true
    end
    args = parse_args(s)
    main(args)
end
