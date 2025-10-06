#=
best.jl

Find the best code parameters.
=#
include("constants.jl")
using JLD2

codes = Set{NTuple{3,Int}}()

for n in 1:200
    path = joinpath("data", get_filename(3, n))
    if isfile(path)
        data = load(path, "data")
        for code in data
            k = Int(code[5])
            d = Int(code[6])
            push!(codes, (n, k, d))
        end
    end
end

sortedCodes = [Int[] for _ in 1:50, _ in 1:100]

for (n, k, d) in codes
    if 1 <= d <= 50 && 1 <= k <= 100
        push!(sortedCodes[d, k], n)
    end
end

filtered = NTuple{3,Int}[]
for d in 1:50, k in 1:100
    sort!(sortedCodes[d, k])
    for n in sortedCodes[d, k]
        best = true
        for (n2, k2, d2) in codes
            if (n2 <= n && k2 * n >= k * n2 && d2 >= d &&
               (n2 <  n || k2 * n >  k * n2 || d2 > d))
                best = false
                break
            end
        end
        if best
            push!(filtered, (code, k, d))
        end
    end
end

println(filtered)
