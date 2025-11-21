#=
helix.jl

Code file to construct a helix code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} x ... x Z_{aN}
    * Z0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to Z0
    * X0 [{u1, u2, ..., uQ}]: a set of N-tuples specifying elements of the group which belong to X0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the helix code.
The check weight is exactly |Z0| + |X0| - |Z0 ^ X0| <= |Z0| + |X0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form—there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
=#

include("util.jl")
include("distance.jl")

using LinearAlgebra

# ---------- internal helpers ----------

# compute lexical index value: (comb' * M) ⋅ strides
@inline function _lex_index(M::AbstractMatrix{<:Integer},
                            comb::AbstractVector{<:Integer},
                            strides::AbstractVector{<:Integer})::Int
    @inbounds sum(((comb' * Matrix{Int}(M)) .* strides))
end

@inline function _shift_rows_mod(A::AbstractMatrix{<:Integer},
                                 row0::AbstractVector{<:Integer},
                                 group::AbstractVector{<:Integer})
    R, C = size(A)
    out = Array{Int}(undef, R, C)
    @inbounds for i in 1:R, j in 1:C
        out[i, j] = mod(Int(A[i, j]) - Int(row0[j]), Int(group[j]))
    end
    out
end

@inline function _scale_cols_mod(A::AbstractMatrix{<:Integer},
                                 v::AbstractVector{<:Integer},
                                 group::AbstractVector{<:Integer})
    R, C = size(A)
    out = Array{Int}(undef, R, C)
    @inbounds for i in 1:R, j in 1:C
        out[i, j] = mod(Int(A[i, j]) * Int(v[j]), Int(group[j]))
    end
    out
end

# generate all row permutations without dependencies
function _row_permutations(A::AbstractMatrix{<:Integer})
    R = size(A, 1)
    rows = [copy(@view A[i, :]) for i in 1:R]
    return Channel{Matrix{Int}}() do ch
        function rec(k::Int)
            if k > R
                put!(ch, reduce(vcat, (permutedims(r, (2, 1)) for r in rows)))
                return
            end
            for i in k:R
                rows[k], rows[i] = rows[i], rows[k]
                rec(k + 1)
                rows[k], rows[i] = rows[i], rows[k]
            end
        end
        rec(1)
    end
end

# ----------
# API
# ----------

#=
Secondary canonicalizer. This accepts the sets z0 and x0 and returns a guess
for the canonical z0 and x0. Applies all tricks other than swapping z0 and x0.
Various equivalences that this tries are reordering Z0 and X0, applying an
automorphism to any component group, and shifting the X0's by 2g, for some g.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (Array): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the Z weight. Also
      accepts tuples at any level.
    * x0 (Array): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the X weight. Also
      accepts tuples at any level.
    * isos (Array): list of lists containing isomorphisms of group
    * strides (Array): strides for indexing qubits

Returns:
    * Tuple containing two elements. The first is the canonical version of z0,
      the second is the canonical version of x0.
=#
function canonicalize_perms(group, z0, x0, isos, strides)
    group  = collect(Int.(group))
    z0     = Array{Int}(z0)
    x0     = Array{Int}(x0)
    n      = prod(group)

    min_z, min_x = copy(z0), copy(x0)
    z_comb = find_strides(fill(n, size(z0, 1)))
    x_comb = find_strides(fill(n, size(x0, 1)))
    min_zi = _lex_index(z0, z_comb, strides)
    min_xi = _lex_index(x0, x_comb, strides)
    max_xi = n ^ size(x0, 1)

    for zperm in _row_permutations(z0)
        zshift = _shift_rows_mod(zperm, vec(zperm[1, :]), group)
        for j in isos
            ziso = _scale_cols_mod(zshift, j, group)
            zidx = _lex_index(ziso, z_comb, strides)
            if zidx > min_zi
                continue
            end
            if zidx < min_zi
                min_z  = ziso
                min_zi = zidx
                min_xi = max_xi
            end
            base_x = _shift_rows_mod(x0, vec(zperm[1, :]), group)
            for xperm in _row_permutations(base_x)
                xiso = shift_X(group, _scale_cols_mod(xperm, j, group))
                xidx = _lex_index(xiso, x_comb, strides)
                if xidx > min_xi
                    continue
                end
                min_xi = xidx
                min_x  = xiso
            end
        end
    end
    return min_z, min_x
end

#=
Main canonicalizer. This accepts the sets z0 and x0 and returns the canonical
isomorphic z0 and x0. The canonical form over equivalent codes is not unique.
Here are some guarantees about this function. z0[1] will only contain 0s. The
arrays z0[1], z0[2], z0[3], ... will be sorted. The same is true for the arrays
in x0. z0[2] will only contain entries x such that x is a divisor of the group
size (0 counts). x0[2] will only contain 0 or 1 entries, with 1 entries only
being acceptable if the corresponding group size is even. Various equivalences
that this tries are swapping Z0 and X0, reordering Z0 and X0, applying an
automorphism to any component group, and shifting the X0's by 2g, for some g.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (Array): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the Z weight. Also
      accepts tuples at any level.
    * x0 (Array): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the X weight. Also
      accepts tuples at any level.

Returns:
    * Tuple containing two elements. The first is the canonical version of z0,
      the second is the canonical version of x0.
=#
function canonicalize(group, z0, x0)
    if size(z0, 1) > size(x0, 1)
        return canonicalize(group, x0, z0)
    end
    group   = collect(Int.(group))
    z0      = Array{Int}(z0)
    x0      = Array{Int}(x0)
    isos    = find_isos(group)
    strides = find_strides(group)

    if size(z0, 1) < size(x0, 1)
        return canonicalize_perms(group, z0, x0, isos, strides)
    end

    z1, x1 = canonicalize_perms(group, z0, x0, isos, strides)
    z2, x2 = canonicalize_perms(group, x0, z0, isos, strides)
    n      = prod(group)
    zc     = find_strides(fill(n, size(z0, 1)))
    xc     = find_strides(fill(n, size(x0, 1)))
    z1i    = _lex_index(z1, zc, strides)
    z2i    = _lex_index(z2, zc, strides)
    x1i    = _lex_index(x1, xc, strides)
    x2i    = _lex_index(x2, xc, strides)

    return (z1i < z2i || (z1i == z2i && x1i < x2i)) ? (z1, x1) : (z2, x2)
end

#=
Canonicalization checker for just Z's. Does not check legality. Merely checks
whether it can find a smaller equivalent z0 instance.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (Array): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the Z weight.
      Also accepts tuples at any level.
    * isos (Array of Arrays): list of all isomorphisms of each factor of group

Returns:
    * Boolean with whether z0 is canonical.
=#
function is_Z_canonical(group, z0, isos)
    group = collect(Int.(group))
    z0    = Array{Int}(z0)

    strides   = find_strides(group)
    comb      = find_strides(fill(prod(group), size(z0, 1)))
    z0_index  = _lex_index(z0, comb, strides)

    for zperm in _row_permutations(z0)
        shifted = _shift_rows_mod(zperm, vec(zperm[1, :]), group)
        for j in isos
            if _lex_index(_scale_cols_mod(shifted, j, group), comb, strides) < z0_index
                return false
            end
        end
    end
    return true
end

#=
Canonicalization checker for just X's. Does not check legality. Merely checks
whether it can find a smaller equivalent x0 instance.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * x0 (Array): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the X weight.
      Also accepts tuples at any level.
    * isos (Array of Arrays): list of all isomorphisms we wish to consider

Returns:
    * Boolean with whether z0 is canonical.
=#
function is_X_canonical(group, x0, isos)
    group = collect(Int.(group))
    x0    = Array{Int}(x0)

    strides   = find_strides(group)
    comb      = find_strides(fill(prod(group), size(x0, 1)))
    x0_index  = _lex_index(x0, comb, strides)

    for xperm in _row_permutations(x0)
        base = [ (j, g) for (j, g) in zip(vec(xperm[1, :]), group) ]
        adj  = [ (g % 2 == 1) ? j : 2 * (j ÷ 2) for (j, g) in base ]
        shifted = _shift_rows_mod(xperm, adj, group)
        for j in isos
            if _lex_index(_scale_cols_mod(shifted, j, group), comb, strides) < x0_index
                return false
            end
        end
    end
    return true
end

#=
Utility function. Given two sets of arrays, finds all possible differences
between an element of one array and an element of the other.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * a (Array): 2D Array. The first array of the two whose differences
      we will be finding. It is a list of all the elements of the group in the
      first array. Number of columns should match length of group.
    * b (Array): 2D Array. The second array of the two whose differences
      we will be finding. It is a list of all the elements of the group in the
      first array. Number of columns should match length of group.

Returns:
    * 2D Array containing all the elements of the group that are the
      difference between something in a and something in b.
=#
function build_set(group, a, b)
    group = collect(Int.(group))
    A = Array{Int}(a)
    B = Array{Int}(b)
    d = size(A, 2)

    seen = Set{NTuple{N,Int}}() where {N}
    rows = Vector{Vector{Int}}()
    @inbounds for i in 1:size(A, 1), j in 1:size(B, 1)
        s = [mod(Int(B[j, t]) - Int(A[i, t]), Int(group[t])) for t in 1:d]
        tup = Tuple(s)
        if !(tup in seen)
            push!(seen, tup)
            push!(rows, s)
        end
    end
    isempty(rows) && return Array{Int}(undef, 0, d)
    M = Array{Int}(undef, length(rows), d)
    @inbounds for (r, v) in enumerate(rows)
        M[r, :] = v
    end
    return M
end

#=
Compute the stabilizer tableau of a code. Tableau is returned in symplectic
form with the convention [Z | X]. Automatically checks if the code is CSS and
converts the tableau to CSS form if so.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (Array): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the Z weight.
      Also accepts tuples at any level.
    * x0 (Array): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the X weight.
      Also accepts tuples at any level.

Returns:
    * Tuple containing two terms. The first is a bool which is True iff the
      code passed was CSS. The second is an Array of qubits (in tuple /
      array form) which need to be hadamarded in order to make the code CSS.
    * The second argument is [] if the first is False.
=#
function css_flips(group, z0, x0)
    group = collect(Int.(group))
    z0    = Array{Int}(z0)
    x0    = Array{Int}(x0)
    n     = prod(group)

    same_diffs = vcat(build_set(group, z0, z0), build_set(group, x0, x0))
    if !isempty(same_diffs)
        same_diffs = unique(same_diffs; dims = 1)
    end

    zx = build_set(group, z0, x0)

    flips = reshape(zeros(Int, length(group)), 1, :)
    for r in 1:size(same_diffs, 1)
        g = vec(same_diffs[r, :])
        gen_g = [g]
        cur = copy(g)
        while maximum(cur) > 0
            cur = mod.(cur .+ g, group)
            push!(gen_g, copy(cur))
        end
        cur_flips = copy(flips)
        for i in 1:size(cur_flips, 1)
            for v in gen_g
                flips = vcat(flips, vec(mod.(cur_flips[i, :] .+ v, group))')
            end
        end
        flips = unique(flips; dims = 1)
        if size(flips, 1) == n
            return false, Array{Int}(undef, 0, length(group))
        end
    end

    bad = Set{Int}()
    strides = find_strides(group)
    for i in Base.Iterators.product((0:((a % 2 == 0) ? 2 : 1):a-1 for a in group)...)
        for r in 1:size(zx, 1)
            v = mod.(collect(Int.(i)) .+ vec(zx[r, :]), group)
            push!(bad, sum(v .* strides))
        end
    end
    for r in 1:size(flips, 1)
        if sum(vec(flips[r, :]) .* strides) in bad
            return false, Array{Int}(undef, 0, length(group))
        end
    end
    return true, flips
end

#=
Compute the stabilizer tableau of a code. Tableau is returned in symplectic
form with the convention [Z | X]. Automatically checks if the code is CSS and
converts the tableau to CSS form if so.

Params:
    * group (Array): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (Array): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the Z weight.
      Also accepts tuples at any level.
    * x0 (Array): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of
      group and not exceed the terms of group. Number of rows is the X weight.
      Also accepts tuples at any level.

Returns:
    * 2D Array in symplectic form (Z|X) with the stabilizers of the code.
      Returns n stabilizers, so contains linearly dependent checks. Should
      automatically be in CSS form if code is CSS.
    * Boolean on whether or not the code is CSS.
=#
function find_stabilizers(group, z0, x0)
    group = collect(Int.(group))
    z0    = Array{Int}(z0)
    x0    = Array{Int}(x0)

    n = prod(group)
    d = length(group)

    strides     = find_strides(group)
    stabilizers = zeros(UInt8, n, 2n)

    can_flip, flips = css_flips(group, z0, x0)

    row = 0
    for g in Base.Iterators.product((0:(a - 1) for a in group)...)
        row += 1
        # Z-part
        @inbounds for r in 1:size(z0, 1)
            col = sum(mod(Int(z0[r, t]) + Int(g[t]), Int(group[t])) * strides[t] for t in 1:d)
            stabilizers[row, col + 1] = 0x01
        end
        # X-part
        @inbounds for r in 1:size(x0, 1)
            col = sum(mod(Int(x0[r, t]) - Int(g[t]), Int(group[t])) * strides[t] for t in 1:d)
            stabilizers[row, col + 1 + n] = 0x01
        end
    end

    if can_flip
        for r in 1:size(flips, 1)
            idx = sum(vec(flips[r, :]) .* strides) + 1
            stabilizers[:, (idx, idx + n)] = stabilizers[:, (idx + n, idx)]
        end
    end
    return stabilizers, can_flip
end

#=
HelixCode container with lazy computations of n, k, d and (optionally) stim tableau.
=#
mutable struct HelixCode
    group::Vector{Int}
    z0::Matrix{Int}
    x0::Matrix{Int}
    stabilizers::Union{Nothing, Matrix{UInt8}}
    stim_tableau::Any
    CSS::Union{Nothing, Bool}
    n::Union{Nothing, Int}
    k::Union{Nothing, Int}
    d::Union{Nothing, Int}
end

HelixCode(group, z0, x0; n=nothing, k=nothing, d=nothing, is_css=nothing) =
    HelixCode(collect(Int.(group)), Array{Int}(z0), Array{Int}(x0),
              nothing, nothing, is_css, n, k, d)

function get_stabilizers(h::HelixCode)
    if h.stabilizers === nothing
        h.stabilizers, h.CSS = find_stabilizers(h.group, h.z0, h.x0)
    end
    h.stabilizers::Matrix{UInt8}
end

function get_stim_tableau(h::HelixCode)
    if h.stim_tableau === nothing
        M = get_stabilizers(h)
        rows = [Vector{UInt8}(M[i, :]) for i in 1:size(M, 1)]
        h.stim_tableau = stimify_symplectic(rows)
    end
    h.stim_tableau
end

get_n(h::HelixCode) = (h.n === nothing && (h.n = prod(h.group)); h.n::Int)

get_k(h::HelixCode) = (h.k === nothing && (h.k = get_n(h) - binary_rank(get_stabilizers(h))); h.k::Int)

function get_d(h::HelixCode; verbose::Bool=false)
    if h.d === nothing
        tableau = get_stim_tableau(h)
        @assert h.CSS !== nothing "You screwed up somewhere?"
        h.d = code_distance(tableau; IS_CSS = h.CSS::Bool, verbose = verbose)
    end
    h.d::Int
end

is_CSS(h::HelixCode) = (h.CSS === nothing && (h.stabilizers, h.CSS) = find_stabilizers(h.group, h.z0, h.x0); h.CSS::Bool)

get_rate(h::HelixCode)     = get_k(h) / get_n(h)
get_rel_dist(h::HelixCode) = get_d(h) / get_n(h)

if abspath(PROGRAM_FILE) == @__FILE__
    #=
    Run unit tests.
    =#

    # Make some CSS codes and check if they are CSS
    CSS_group = (4, 7, 3)
    n = prod(CSS_group)
    X0 = [0 4 1; 2 3 2]
    Z0 = [1 6 2; 3 1 0; 1 1 1]
    println(canonicalize(CSS_group, X0, Z0))
    CSS_stabs, _can = find_stabilizers(CSS_group, Z0, X0)
    println("Your CSS stabs are:")
    for i in 1:size(CSS_stabs, 1)
        println(symp2Pauli(Vector{UInt8}(CSS_stabs[i, :]), n))
    end

    # Make some non-CSS codes and check if they are CSS
end
