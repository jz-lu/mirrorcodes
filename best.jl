#=
best.jl

Find the best code parameters.
=#
include("constants.jl")
using JLD2

function best_codes(data_dir::AbstractString = "data")
    codes = Set{NTuple{3,Int}}()

    for n in 1:200
        path = joinpath(data_dir, get_filename(3, n))
        if isfile(path)
            data = load(path, "data")
            for rec in data
                k = Int(rec[5])
                d = Int(rec[6])
                push!(codes, (n, k, d))
            end
        end
    end

    sorted_codes = [Int[] for _ in 1:50, _ in 1:100]

    for (n, k, d) in codes
        if 1 <= d <= 50 && 1 <= k <= 100
            push!(sorted_codes[d, k], n)
        end
    end

    filtered = NTuple{3,Int}[]
    for d in 1:50, k in 1:100
        sort!(sorted_codes[d, k])
        for n_val in sorted_codes[d, k]
            best = true
            for (n2, k2, d2) in codes
                if (n2 <= n_val && k2 * n_val >= k * n2 && d2 >= d &&
                    (n2 < n_val || k2 * n_val > k * n2 || d2 > d))
                    best = false
                    break
                end
            end
            if best
                push!(filtered, (n_val, k, d))
            end
        end
    end

    return filtered
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(best_codes())
end
