#=
search.jl

Code file for conducting a search for good helix codes.
This consists of two steps:
    1) Generate a helix code according to some systematic protocol.
    2) Evaluate if the code is good. If so, keep it. If not, delete and continue.
We can start by searching through codes for which n ~ 100 +/- 100, and check
weight <= 6.
But let n >> check weight so that the LDPCness kicks in, e.g. n >= 30.

The precise meaning of "good" is debatable, but we will adopt the following
two-stage filtering method.
Stage 1 (distance-rate tradeoff):
    * Evaluate the rate R of the code. If R < 1/16, discard.
    * Evaluate the distance d of the code. If evaluation of the distance takes >3
      min, keep the code.
    * If the distance is calculated successfully, discard if Rd < 1/2. Keep
      otherwise.

Stage 2 (practicality):
    * Evaluate the pseudo-threshold using BP-OSD. If it is above some TBD cutoff,
      keep.
    * Evaluate the circuit distance?
=#

include("helix.jl")
include("util.jl")

using Primes

# local helpers
gcd_list(x::Integer, y::Integer) = gcd(x, y)
gcd_list(x::Integer, ys::Integer...) = foldl(gcd, ys; init=x)

# generate all integer partitions of n with parts ≥ I
function _partitions_list(n::Int; I::Int=1)
    out = Vector{Vector{Int}}()
    function rec(rem::Int, minv::Int, acc::Vector{Int})
        push!(out, vcat(acc, rem))
        for i in minv:div(rem, 2)
            rec(rem - i, i, vcat(acc, i))
        end
    end
    rec(n, I, Int[])
    out
end

#=
Check whether `helix_code` is in canonical form.
We say that a helix code (group, Z0, X0) is in canonical form if TODO

Input:
    * helix_code (Tuple): Tuple of (group, Z0, X0) where group is a tuple
      and Z0 and X0 are sets

Returns:
    * Binary indication of whether `helix_code` is in canonical form.
=#
function is_canonical(helix_code)
    group, Z0, X0 = helix_code
    cZ, cX = canonicalize(group, Z0, X0)
    return all(Z0 .== cZ) && all(X0 .== cX)
end

#=
Search helix codes over `n` qubits with Z-weight `Z_wt` and X-weight `X_wt`.
Save the ones which pass stage 1.

Input:
    * n (Int): number of qubits
    * Z_wt (Int): weight of the Z's in each check
    * X_wt (Int): weight of the X's in each check

Output:
    * A .jld2 file for each code which passes the stage 1 test

Returns:
    * Nothing
=#
function n_level_search(n::Int, Z_wt::Int, X_wt::Int)
    return nothing
end

#=
Compute number of possible codes on n qubits with given weights. This does not
assume any canonicalization whatsoever, other than setting Z0[1] to 0. This
function exists to have nice ranges over which to index. This returns the
number of partitions of the powers of n times n to the power of Z_wt + X_wt - 1.

Params:
    * n (Int): The number of qubits we want to search over
    * Z_wt (Int): The number of terms in Z0
    * X_wt (Int): The number of terms in X0

Returns:
    * An Int with the number of possible codes on n qubits. Calculates the
      number of partitions of the powers of n times number of ways to pick
      qubits in Z0 and X0, not counting Z0[1].
=#
function num_indices(n::Int, Z_wt::Int, X_wt::Int)::Int
    result = 1
    for (_, e) in factor(n)
        result *= length(_partitions_list(Int(e)))
    end
    return result * n^(Z_wt + X_wt - 1)
end

#=
Compute several tuples representing some qubits in group, which correspond to
the values in Z0 and X0. This is done given an index
from 0 to n ^ (Z_wt + X_wt - 1) - 1. This is no different than expressing
several numbers "base group". Notably, this works for larger indices too, but
will only consider the index mod n ^ (Z_wt + X_wt). Regardless of what the
index says, Z0[1] is force-set to the all 0's tuple.

Params:
    * group (Array): the group we are decomposing the index into
    * Z_wt (Int): The number of terms in Z0
    * X_wt (Int): The number of terms in X0
    * index (Int): A number from 0 to n ^ (Z_wt + X_wt - 1) - 1 corresponding
      to several tuples mod group.

Returns:
    * Four arrays. The first has length Z_wt and is a Vector of N-tuples containing
      the values of Z0 as given by index (after dividing by n ^ X_wt) with a
      tuple of zeros prepended. The second has length X_wt and is a Vector of
      N-tuples containing the values of X0 as given by index (after modding by
      n ^ X_wt). The third and fourth arrays contain the indices used to
      construct the tuples of the first and second arrays, respectively. Each
      entry in the third and fourth arrays is from 0 to n - 1.
=#
function multi_index_to_tuples(group, Z_wt::Int, X_wt::Int, index::Int)
    n = Int(prod(group))
    d = length(group)
    Zs = NTuple{d,Int}[]
    Xs = NTuple{d,Int}[]
    Z_nums = Int[]
    X_nums = Int[]

    idx = index
    for _ in 1:X_wt
        t = index_to_tuple(group, idx)
        push!(Xs, t)
        push!(X_nums, idx % n)
        idx ÷= n
    end

    for _ in 1:(Z_wt - 1)
        t = index_to_tuple(group, idx)
        push!(Zs, t)
        push!(Z_nums, idx % n)
        idx ÷= n
    end

    push!(Zs, ntuple(_ -> 0, d))
    push!(Z_nums, 0)

    return reverse(Zs), reverse(Xs), reverse(Z_nums), reverse(X_nums)
end

#=
Find all non-isomorphic abelian groups of n qubits.

Params:
    * n (Int): The number of qubits

Returns:
    * A Vector of tuples, each containing the sizes of various abelian groups,
      such that the product of all the groups is n. Each group must be a power
      of a prime.
=#
function n_partitions(n::Int)
    primes = Int[]
    powers = Int[]

    for (p, e) in factor(n)
        push!(primes, Int(p))
        push!(powers, Int(e))
    end

    combos = [ _partitions_list(e) for e in powers ]

    result = Vector{Tuple}()
    for i in Base.Iterators.product(combos...)
        group_sizes = Int[]
        for (j, p) in enumerate(primes)
            for k in i[j]
                push!(group_sizes, p ^ k)
            end
        end
        push!(result, Tuple(group_sizes))
    end
    
    return result
end

#=
Wrapper for processing only codes which are canonical. Currently returns the
number of valid codes.

Params:
    * n (Int): The number of physical qubits of the desired codes
    * Z_wt (Int): The number of terms in Z0
    * X_wt (Int): The number of terms in X0
    * index_start (Int, optional): The index at which the loop should start
      counting codes. Indices can range from 0 to n ^ (Z_wt + X_wt - 1) - 1. 0
      by default. 
    * index_end (Int, optional): The index at which the loop should stop
      counting codes. The code at index_end is not counted. Indices can range
      from 0 to n ^ (Z_wt + X_wt - 1) - 1. `nothing` by default, which is then set
      to the max value. 
    
Returns:
    * Number of canonical codes with n qubits.
=#
function process_codes(n::Int, Z_wt::Int, X_wt::Int; index_start::Int=0, index_end::Union{Nothing,Int}=nothing)
    if index_end === nothing
        index_end = num_indices(n, Z_wt, X_wt)
    end

    groups = n_partitions(n)

    index = index_start
    COUNTER = 0
    while index < index_end
        quotient, remainder = divrem(index, n^(Z_wt + X_wt - 1))
        group = groups[quotient + 1]
        Zs, Xs, Z_nums, X_nums = multi_index_to_tuples(group, Z_wt, X_wt, remainder)
        jump =  -1
        for i in 1:(Z_wt + X_wt - 1)
            if i == 1
                if any((j > 0) && (g % j > 0) for (j, g) in zip(Zs[2], group))
                    jump = n ^ (Z_wt + X_wt - 2)
                    break
                end
            end
            if i < Z_wt
                if Z_nums[i + 1] <= Z_nums[i]
                    jump = n ^ (Z_wt + X_wt - 1 - i)
                    break
                end
            elseif i > Z_wt
                xi = i - Z_wt + 1
                if X_nums[xi] <= X_nums[xi - 1]
                    jump = n ^ (Z_wt + X_wt - 1 - i)
                    break
                end
            else
                x0 = Xs[1]
                if maximum(collect(x0)) > 1 || any((j > 0) && (g % 2 > 0) for (j, g) in zip(x0, group))
                    jump = n ^ (Z_wt + X_wt - 1 - i)
                    break
                end
            end
        end

        groupA = collect(Int.(group))
        d = length(groupA)
        Zmat = reshape(collect(Iterators.flatten(Zs)), (Z_wt, d))
        Xmat = reshape(collect(Iterators.flatten(Xs)), (X_wt, d))

        if jump < 0
            cZ, cX = canonicalize(groupA, Zmat, Xmat)
            if !(all(Zmat .== cZ) && all(Xmat .== cX))
                jump = 1
            end
        end
        if jump >= 0
            index = ((index ÷ jump) + 1) * jump
            continue
        end
        COUNTER += 1
        index += 1
    end
    return COUNTER
end

#=
Finds possible elements of Z0[:, k] that are worth searching over for some k.

Params:
    * Z_wt (Int): The number of terms in Z0
    * size (Int): The size of the cyclic group we are finding candidates for

Returns:
    * Vector of tuples. Each tuple contains a Vector of entries of Z0[:, k] and
      the gcd mod size of the elements and the array size, needed for computing
      the list of isomorphisms that leaves them invariant.
=#
function decomposed_Z0_candidates(Z_wt::Int, size::Int)
    candidates = [Int[0] for _ in 1:Z_wt]
    if Z_wt >= 2
        candidates[2] = [i for i in 0:size-1 if i == 0 || size % i == 0]
    end
    for i in 3:Z_wt
        candidates[i] = collect(0:size-1)
    end

    result = Vector{Tuple{Vector{Int},Int}}()
    isos = find_isos(size)
    strides = find_strides(fill(size, Z_wt))
    for zs in Base.Iterators.product((candidates...)...)
        z_list = collect(zs)
        base = sum(z_list[i] * strides[i] for i in 1:Z_wt)
        mins = minimum([ sum(mod(z_list[i] * j, size) * strides[i] for i in 1:Z_wt) for j in isos ])
        if base == mins
            g = mod(gcd_list(z_list..., size), size)
            push!(result, (z_list, g))
        end
    end
    return result
end

#=
Finds possible elements of X0[:, k] that are worth searching over for some k.

Params:
    * X_wt (Int): The number of terms in X0
    * size (Int): The size of the cyclic group we are finding candidates for

Returns:
    * 3d array. The outer dimension is loops over all values from 0
      to size, inclusive. Only the entries which are factors of size have any
      elements. The second dimension is just listing candidates. The inner
      dimension loops over the terms of X0, and thus has length X_wt.
=#
function decomposed_X0_candidates(X_wt::Int, size::Int)
    candidates = [ (size % 2 == 1 ? Int[0] : Int[0,1]) for _ in 1:X_wt ]
    for i in 2:X_wt
        candidates[i] = collect(0:size-1)
    end

    isos_by_gcd = [Int[] for _ in 1:size]
    for i in 1:size
        if size % i > 0
            continue
        end
        isos_by_gcd[mod(i, size) + 1] = [j for j in 1:size-1 
                                 if mod(j - 1, Int(size ÷ i)) == 0 && gcd_list(j, size) == 1]
    end

    result = [Vector{Vector{Int}}() for _ in 1:(size + 1)]
    strides = find_strides(fill(size, X_wt))
    for i in 0:size-1
        iso = isos_by_gcd[(i % size) + 1]
        isempty(iso) && continue
        for xs in Base.Iterators.product((candidates...)...)
            x_list = collect(xs)
            base = sum(x_list[t] * strides[t] for t in 1:X_wt)
            mins = minimum([ sum(mod(x_list[t] * j, size) * strides[t] for t in 1:X_wt) for j in iso ])
            if base == mins
                push!(result[i + 1], x_list)
            end
        end
    end
    return result
end

#=
Finds possible tuples Z0 that are worth searching over.

Params:
    * Z_wt (Int): The number of terms in Z0
    * group (Array): The group whose codes we are finding
    * min_k (Int, optional): Whether codes should be filtered to exclude those with k < min_k

Returns:
    * Vector of Z0 candidates with possible isomorphisms. Each candidate is a
      tuple containing (Z0, isomorphisms). Each Z0 is a sorted Vector of
      tuples mod group. Each Z0[1] is always all zeros, and each Z0[2]
      only contains terms which divide the group size. Each candidate
      isomorphisms is an index of gcds for each dimension, communicating
      potential isomorphisms which canonicalize Z0 as much as possible.
=#
function build_Z0_candidates(Z_wt::Int, group; min_k::Int = 3)
    n = Int(prod(group))
    d = length(group)
    candidates = [decomposed_Z0_candidates(Z_wt, Int(g)) for g in group]
    lengths = [length(c) for c in candidates]
    strides = find_strides(lengths)
    
    gcd_strides = find_strides(collect(group))

    max_index = Int(prod(lengths))
    index_val = 0
    result = Vector{Tuple{Matrix{Int},Int}}()
    isos = find_isos(collect(group))
    odd_indices = [i for i in 1:d if Int(group[i]) % 2 == 1]
    while index_val < max_index
        jump = -1
        index = index_to_tuple(lengths, index_val)
        zs = zeros(Int, Z_wt, d)
        z_indices = nothing
        for i in 1:d
            zs[:, i] = candidates[i][index[i] + 1][1]
            z_indices = zs * strides
            if any(z_indices[1:end-1] .> z_indices[2:end])
                jump = strides[i]
                break
            end
        end
        if (jump > 0 || any(z_indices[1:end-1] .>= z_indices[2:end])
            || !is_Z_canonical(collect(group), zs, isos)
            || (min_k > 0 && (all(zs[:, odd_indices] .== 0)
                || compute_rank_from_tuples(collect(group), zs) > n - min_k)))
            jump = max(jump, 1)
            index_val = ((index_val ÷ jump) + 1) * jump
            continue
        end
        push!(result, (zs, Int(sum(candidates[i][index[i] + 1][2] * gcd_strides[i] for i in 1:d))))
        index_val += 1
    end
    return result
end

#=
Finds possible tuples X0 that are worth searching over. Split up by the gcd of
the Zs, which defines isomorphisms under which the Xs are minimal.

Params:
    * X_wt (Int): The number of terms in X0
    * group (Array): The group whose codes we are finding
    * min_k (Int, optional): Whether codes should be filtered to exclude those with k < min_k

Returns:
    * Vector of Vectors of X0 candidates. The outer Vector is indexed by the gcd's of
      the Zs which determine the isomorphisms under which the Zs are equivalent,
      meaning we must minimize the Xs over this set. Each X0 is a sorted
      Vector of tuples mod group. Each X0[1] is always all 0 or 1.
=#
function build_X0_candidates(X_wt::Int, group; min_k::Int = 3)
    n = Int(prod(group))
    d = length(group)
    candidates = [decomposed_X0_candidates(X_wt, Int(g)) for g in group]
    result = [Vector{Matrix{Int}}() for _ in 1:n]
    for i in 0:n-1
        gcds = index_to_tuple(collect(group), i)
        lengths = [length(candidates[j][gcds[j] + 1]) for j in 1:d]
        
        strides = find_strides(lengths)
        max_index = Int(prod(lengths))
        if max_index == 0
            continue
        end
        isos = [[k for k in 1:Int(g)-1 if mod(k - 1,
                 (gcds[j] == 0 ? 1 : Int(g ÷ gcds[j]))) == 0 && gcd_list(k, Int(g)) == 1]
                for (j, g) in enumerate(group)]
        odd_indices = [i for i in 1:d if Int(group[i]) % 2 == 1]
        index_val = 0
        while index_val < max_index
            jump = -1
            index = index_to_tuple(lengths, index_val)
            xs = zeros(Int, X_wt, d)
            x_indices = nothing
            for j in 1:d
                xs[:, j] = candidates[j][gcds[j] + 1][index[j] + 1]
                x_indices = xs * strides
                if any(x_indices[1:end-1] .> x_indices[2:end])
                    jump = strides[j]
                    break
                end
            end
            if (jump > 0 || any(x_indices[1:end-1] .>= x_indices[2:end])
                || !is_X_canonical(collect(group), xs, isos)
                || (min_k > 0 && (all(xs[:, odd_indices] .== 0) ||
                                     compute_rank_from_tuples(collect(group), xs) > n - min_k)))
                jump = max(jump, 1)
                index_val = ((index_val ÷ jump) + 1) * jump
                continue
            end
            push!(result[i + 1], xs)
            index_val += 1
        end
    end
    return result
end

#=
Finds all codes of weights Z_wt and X_wt for a given group.

Params:
    * Z_wt (Int): The number of terms in Z0
    * X_wt (Int): The number of terms in X0
    * group (Array): The group whose codes we are finding
    * min_k (Int, optional): Whether codes should be filtered to exclude those with k < min_k
    * return_k (Bool, optional): Whether to return k or not

Returns:
    * Vector of tuples of the form (group, Z0, X0, IS_CSS, k) for valid codes. Each of Z0 and
      X0 is a Vector of tuples mod group. If return_k is true, also
      adds the k, the logical dimension of the code, to the tuple.
=#
function find_all_codes_in_group(Z_wt::Int, X_wt::Int, group; min_k::Int = 3, return_k::Bool = true)
    n = Int(prod(group))
    zs = build_Z0_candidates(Z_wt, collect(group))
    xs = build_X0_candidates(X_wt, collect(group))
    codes = Tuple[]
    for i in zs
        for j in xs[i[2] + 1]
            code = HelixCode(group, i[1], j; n = n)
            if min_k > 0 && get_k(code) < min_k
                continue
            end
            canon_Z, canon_X = canonicalize(group, i[1], j)
            if all(i[1] .== canon_Z) && all(j .== canon_X)
                push!(codes, return_k ? (group, i[1], j, is_CSS(code), get_k(code))
                                      : (group, i[1], j, is_CSS(code)))
            end
        end
    end
    return codes
end

#=
Finds all codes for a given number of qubits, n, of given weight.

Params:
    * n (Int): The number of physical qubits of the desired codes
    * Z_wt (Int): The number of terms in Z0
    * X_wt (Int): The number of terms in X0
    * min_k (Int, optional): Whether codes should be filtered to exclude those with k < min_k

Returns:
    * Vector of tuples of the form (group, Z0, X0) for valid codes. Each of Z0
      and X0 is a Vector of tuples mod group. If return_k is true, the
      tuple also has k at the end, the logical dimension of the code.
=#
function find_all_codes(n::Int, Z_wt::Int, X_wt::Int; min_k::Int = 3)
    if n < 2
        return Tuple[]
    end
    if min_k > 0 && (Z_wt == 3 || X_wt == 3)
        p = n
        while true
            if p == 1
                return Tuple[]
            end
            if p % 2 == 1
                break
            end
            p ÷= 2
        end
    end
    
    result = Tuple[]
    for group in n_partitions(n)
        append!(result, find_all_codes_in_group(Z_wt, X_wt, group; min_k=min_k, return_k = min_k > 0))
    end
    return result
end

function main()
    for i in 0:31
        println(i, " ", length(find_all_codes(i, 3, 3)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
