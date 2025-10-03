#=
helix.jl

Code file to construct a helix code via specification in standard form.

The standard form is given as follows:
    * group [(a1, a2, ..., aN)]: positive integers that specify the abelian group Z_{a1} x ... x Z_{aN}
    * Z0 [{v1, v2, ..., vP}]: a set of N-tuples specifying elements of the group which belong to Z0
    * X0 [{u1, u2, ..., uQ}]: a set of N-tuples specifying elements of the group which belong to X0

The return is a stabilizer tableau of size n x 2n, where n = a1 * ... * aN, which are the stabilizers of the helix code.
The check weight is exactly |Z0| + |X0| - |Z0 ^ X0| <= |Z0| + |X0|, so keep these small if you want LDPC.
The tableau is NOT in reduced form---there are dependent stabilizers! (E.g. think of the last 2 stabilizers in the toric code.)
=#
include("util.jl")
include("distance.jl")

using LinearAlgebra
using Printf
using Iterators

# --- small helpers ---

# elementwise mod on a matrix with per-column moduli in `group`
function _mod_group(A::AbstractMatrix{<:Integer}, group::AbstractVector{<:Integer})
    out = Array{Int}(undef, size(A))
    @inbounds for i in 1:size(A,1), j in 1:size(A,2)
        out[i,j] = mod(Int(A[i,j]), Int(group[j]))
    end
    return out
end

# rowwise shift by a vector, then mod per column
function _shift_and_mod(A::AbstractMatrix{<:Integer}, row0::AbstractVector{<:Integer}, group::AbstractVector{<:Integer})
    out = Array{Int}(undef, size(A))
    @inbounds for i in 1:size(A,1), j in 1:size(A,2)
        out[i,j] = mod(Int(A[i,j]) - Int(row0[j]), Int(group[j]))
    end
    return out
end

# multiply each column j by scalar v[j], then mod per column
function _scale_cols_mod(A::AbstractMatrix{<:Integer}, v::AbstractVector{<:Integer}, group::AbstractVector{<:Integer})
    out = Array{Int}(undef, size(A))
    @inbounds for i in 1:size(A,1), j in 1:size(A,2)
        out[i,j] = mod(Int(A[i,j]) * Int(v[j]), Int(group[j]))
    end
    return out
end

# dot for 2D: (w' * M) * strides, where w and strides are 1D
function _lex_index(M::AbstractMatrix{<:Integer}, w::AbstractVector{<:Integer}, strides::AbstractVector{<:Integer})::Int
    @assert size(M,1) == length(w)
    @assert size(M,2) == length(strides)
    tmp = zeros(Int, size(M,2))
    @inbounds for j in 1:size(M,2)
        s = 0
        for i in 1:size(M,1)
            s += Int(w[i]) * Int(M[i,j])
        end
        tmp[j] = s
    end
    s2 = 0
    @inbounds for j in 1:length(strides)
        s2 += tmp[j] * Int(strides[j])
    end
    return s2
end

# generate all row permutations of a matrix as matrices (channel-based iterator)
function _row_permutations(A::AbstractMatrix{<:Integer})
    rows = [collect(@view A[i, :]) for i in 1:size(A,1)]
    return Channel{Matrix{Int}}(ctype=Matrix) do ch
        function rec(start::Int)
            if start > length(rows)
                put!(ch, reduce(vcat, (permutedims(r, (2,1)) for r in rows)))
                return
            end
            for i in start:length(rows)
                rows[start], rows[i] = rows[i], rows[start]
                rec(start + 1)
                rows[start], rows[i] = rows[i], rows[start]
            end
        end
        rec(1)
    end
end

# cartesian product over a vector of vectors -> iterator of Vector{Int}
function _product_vectors(vs::Vector{<:AbstractVector})
    return (collect(t) for t in Iterators.product((vs...)...))
end

# --- core algorithms ---

"""
Secondary canonicalizer. This accepts the sets z0 and x0 and returns a guess
for the canonical z0 and x0. Applies all tricks other than swapping z0 and x0.
Various equivalences that this tries are reordering Z0 and X0, applying an
automorphism to any component group, and shifting the X0's by 2g, for some g.

Params:
    * group (np.ndarray): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the Z weight. Also
      accepts tuples at any level.
    * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the X weight. Also
      accepts tuples at any level.
    * isos (np.ndarray): list of lists containing isomorphisms of group
    * strides (np.ndarray): strides for indexing qubits

Returns:
    * Tuple containinng two elements. The first is the canonical version of z0,
      the second is the canonical version of x0.
"""
function canonicalize_perms(group, z0, x0, isos, strides)
    group = collect(Int.(group))
    z0 = Array{Int}(z0)
    x0 = Array{Int}(x0)
    n = prod(group)

    min_z, min_x = copy(z0), copy(x0)
    z_combiner = find_strides(fill(n, size(z0,1)))
    x_combiner = find_strides(fill(n, size(x0,1)))
    min_z_index = _lex_index(z0, z_combiner, strides)
    min_x_index = _lex_index(x0, x_combiner, strides)
    max_x_index = n ^ size(x0,1)

    for zperm in _row_permutations(z0)
        zshift = _shift_and_mod(zperm, vec(zperm[1, :]), group)
        for j in _product_vectors(isos)
            ziso = _scale_cols_mod(zshift, j, group)
            zindex = _lex_index(ziso, z_combiner, strides)
            if zindex > min_z_index
                continue
            end
            if zindex < min_z_index
                min_z = ziso
                min_z_index = zindex
                min_x_index = max_x_index
            end
            base_x = _shift_and_mod(x0, vec(zperm[1, :]), group)
            for xperm in _row_permutations(base_x)
                xiso = shift_X(group, _scale_cols_mod(xperm, j, group))
                xindex = _lex_index(xiso, x_combiner, strides)
                if xindex > min_x_index
                    continue
                end
                min_x_index = xindex
                min_x = xiso
            end
        end
    end
    return min_z, min_x
end

"""
Main canonicalizer. This accepts the sets z0 and x0 and returns the canonical
isomorphic z0 and x0. The canonical form over equivalent codes is not unique.
Here are some guarantees about this function. z0[0] will only contain 0s. The
arrays z0[0], z0[1], z0[2], ... will be sorted. The same is true for the arrays
in x0. z0[1] will only contain entries x such that x is a divisor of the group
size (0 counts). x0[1] will only contain 0 or 1 entries, with 1 entries only
being acceptable if the corresponding group size is even. Various equivalences
that this tries are swapping Z0 and X0, reordering Z0 and X0, applying an
automorphism to any component group, and shifting the X0's by 2g, for some g.

Params:
    * group (np.ndarray): 1D Array of all the group sizes in the factored
      version of the group. Also accepts tuple.
    * z0 (np.ndarray): 2D Array of all the elements of Z0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the Z weight. Also
      accepts tuples at any level.
    * x0 (np.ndarray): 2D Array of all the elements of X0, already decomposed
      along the group dimensions. Number of columns should match length of group
      and not exceed the terms of group. Number of rows is the X weight. Also
      accepts tuples at any level.

Returns:
    * Tuple containinng two elements. The first is the canonical version of z0,
      the second is the canonical version of x0.
"""
function canonicalize(group, z0, x0)
    if size(z0,1) > size(x0,1)
        return canonicalize(group, x0, z0)
    end
    group = collect(Int.(group))
    z0 = Array{Int}(z0)
    x0 = Array{Int}(x0)
    isos = find_isos(group)
    strides = find_strides(group)

    if size(z0,1) < size(x0,1)
        return canonicalize_perms(group, z0, x0, isos, strides)
    end

    z1, x1 = canonicalize_perms(group, z0, x0, isos, strides)
    z2, x2 = canonicalize_perms(group, x0, z0, isos, strides)
    n = prod(group)
    z_comb = find_strides(fill(n, size(z0,1)))
    x_comb = find_strides(fill(n, size(x0,1)))
    z1i = _lex_index(z1, z_comb, strides)
    z2i = _lex_index(z2, z_comb, strides)
    x1i = _lex_index(x1, x_comb, strides)
    x2i = _lex_index(x2, x_comb, strides)
    if z1i < z2i || (z1i == z2i && x1i < x2i)
        return z1, x1
    else
        return z2, x2
    end
end

"""
Canonicalization checker for just Z's. Does not check legality. Merely checks
whether it can find a smaller equivalent z0 instance.
"""
function is_Z_canonical(group, z0, isos)
    group = collect(Int.(group))
    z0 = Array{Int}(z0)

    strides = find_strides(group)
    comb = find_strides(fill(prod(group), size(z0,1)))
    z0_index = _lex_index(z0, comb, strides)

    for zperm in _row_permutations(z0)
        shifted = _shift_and_mod(zperm, vec(zperm[1, :]), group)
        for j in _product_vectors(isos)
            cand = _scale_cols_mod(shifted, j, group)
            if _lex_index(cand, comb, strides) < z0_index
                return false
            end
        end
    end
    return true
end

"""
Canonicalization checker for just X's. Does not check legality. Merely checks
whether it can find a smaller equivalent x0 instance.
"""
function is_X_canonical(group, x0, isos)
    group = collect(Int.(group))
    x0 = Array{Int}(x0)

    strides = find_strides(group)
    comb = find_strides(fill(prod(group), size(x0,1)))
    x0_index = _lex_index(x0, comb, strides)

    for xperm in _row_permutations(x0)
        # subtract adjusted first row per component: if g even use 2*floor(j/2), else j
        first = vec(xperm[1, :])
        adj = [ (g % 2 == 1) ? first[t] : 2 * (first[t] ÷ 2) for (t, g) in enumerate(group) ]
        shifted = _shift_and_mod(xperm, adj, group)
        for j in _product_vectors(isos)
            cand = _scale_cols_mod(shifted, j, group)
            if _lex_index(cand, comb, strides) < x0_index
                return false
            end
        end
    end
    return true
end

"""
Utility function. Given two sets of arrays, finds all possible differences
between an element of one array and an element of the other.

Returns a 2D array of unique differences mod `group` (rows are elements).
"""
function build_set(group, a, b)
    group = collect(Int.(group))
    A = Array{Int}(a)
    B = Array{Int}(b)
    d = size(A,2)
    seen = Set{NTuple{Int,Int}}()  # row tuples
    rows = NTuple{Int,Int}[]  # will define length later

    for i in 1:size(A,1)
        for j in 1:size(B,1)
            tmp = Vector{Int}(undef, d)
            @inbounds for t in 1:d
                tmp[t] = mod(Int(B[j,t]) - Int(A[i,t]), Int(group[t]))
            end
            tup = Tuple(tmp)
            if !(tup in seen)
                push!(seen, tup)
                push!(rows, tup)
            end
        end
    end
    if isempty(rows)
        return Array{Int}(undef, 0, d)
    end
    M = Array{Int}(undef, length(rows), d)
    for (ri, tup) in enumerate(rows)
        @inbounds for t in 1:d
            M[ri, t] = tup[t]
        end
    end
    return M
end

"""
Compute the stabilizer tableau of a code. Tableau is returned in symplectic
form with the convention [Z | X]. Automatically checks if the code is CSS and
converts the tableau to CSS form if so.

Returns:
    * Tuple containing (is_css::Bool, flips::Array) as described below, used by find_stabilizers.
"""
function css_flips(group, z0, x0)
    group = collect(Int.(group))
    z0 = Array{Int}(z0)
    x0 = Array{Int}(x0)
    n = prod(group)

    # build sets containing differences between two qubits with the same pauli on
    same_diffs = vcat(build_set(group, z0, z0), build_set(group, x0, x0))
    # throw away duplicates
    if !isempty(same_diffs)
        same_diffs = unique(same_diffs; dims=1)
    end

    # differences of qubits with different paulis
    zx = build_set(group, z0, x0)

    # for each difference between two elements of the same set...
    flips = reshape(zeros(Int, length(group)), 1, :)
    for r in 1:size(same_diffs,1)
        g = vec(same_diffs[r, :])
        # find the group generated by it...
        gen_g = [g]
        cur = copy(g)
        while maximum(cur) > 0
            cur = mod.(cur .+ g, group)
            push!(gen_g, copy(cur))
        end
        cur_flips = copy(flips)
        # ...and use those groups to generate the full group of things connected by steps
        for i in 1:size(cur_flips,1)
            for v in gen_g
                push!(flips, vec(mod.(cur_flips[i, :] .+ v, group))')
            end
        end
        flips = unique(flips; dims=1)
        # if all the qubits lie in the same block, the code cannot be CSS
        if size(flips,1) == n
            return false, Array{Int}(undef, 0, length(group))
        end
    end

    # find full set of differences between hadamarded and non hadamarded differences.
    bad = Set{Int}()
    strides = find_strides(group)
    steps = (0:((a % 2 == 0) ? 2 : 1):a-1 for a in group)
    for tup in Iterators.product(steps...)
        base = collect(Int.(tup))
        for r in 1:size(zx,1)
            v = mod.(base .+ vec(zx[r, :]), group)
            push!(bad, dot(v, strides))
        end
    end
    for r in 1:size(flips,1)
        if dot(vec(flips[r, :]), strides) in bad
            return false, Array{Int}(undef, 0, length(group))
        end
    end
    return true, flips
end

"""
Compute the stabilizer tableau of a code. Tableau is returned in symplectic
form with the convention [Z | X]. Automatically checks if the code is CSS and
converts the tableau to CSS form if so.

Returns:
    * 2D array in symplectic form (Z|X) with the stabilizers of the code.
      Returns n stabilizers, so contains linearly dependent checks. Should
      automatically be in CSS form if code is CSS.
    * boolean on whether or not the code is CSS
"""
function find_stabilizers(group, z0, x0)
    # convert to arrays
    group = collect(Int.(group))
    z0 = Array{Int}(z0)
    x0 = Array{Int}(x0)

    n = prod(group)
    d = length(group)

    # compute strides
    strides = find_strides(group)
    stabilizers = zeros(UInt8, n, 2n)

    # if can_flip is true, the code is CSS and flips contains the qubits that need
    # to be hadamarded (the format of these is not indices, but arrays)
    can_flip, flips = css_flips(group, z0, x0)

    # iterate over all qubits / all tuples in the group / all stabilizers
    ranges = (0:(a-1) for a in group)
    row = 0
    for g in Iterators.product(ranges...)
        row += 1
        # Z part
        for r in 1:size(z0,1)
            col = 0
            @inbounds for t in 1:d
                col += mod(Int(z0[r,t]) + Int(g[t]), Int(group[t])) * strides[t]
            end
            stabilizers[row, col + 1] = 0x01
        end
        # X part
        for r in 1:size(x0,1)
            col = 0
            @inbounds for t in 1:d
                col += mod(Int(x0[r,t]) - Int(g[t]), Int(group[t])) * strides[t]
            end
            stabilizers[row, col + 1 + n] = 0x01
        end
    end

    # flip qubits that need hadamarding if code is css
    if can_flip
        for r in 1:size(flips,1)
            idx = dot(vec(flips[r, :]), strides)
            stabilizers[:, (idx + 1, idx + 1 + n)] = stabilizers[:, (idx + 1 + n, idx + 1)]
        end
    end
    return stabilizers, can_flip
end

# --- HelixCode type ---

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

function HelixCode(group, z0, x0; n=nothing, k=nothing, d=nothing, is_css=nothing)
    g = collect(Int.(group))
    z = Array{Int}(z0)
    x = Array{Int}(x0)
    return HelixCode(g, z, x, nothing, nothing, is_css, n, k, d)
end

function get_stabilizers(h::HelixCode)
    if h.stabilizers === nothing
        stabs, css = find_stabilizers(h.group, h.z0, h.x0)
        h.stabilizers = stabs
        h.CSS = css
    end
    return h.stabilizers::Matrix{UInt8}
end

function get_stim_tableau(h::HelixCode)
    if h.stim_tableau === nothing
        M = get_stabilizers(h)
        rows = [vec(M[i, :]) for i in 1:size(M,1)]
        h.stim_tableau = stimify_symplectic(rows)
    end
    return h.stim_tableau
end

function get_n(h::HelixCode)
    if h.n === nothing
        h.n = prod(h.group)
    end
    return h.n::Int
end

function get_k(h::HelixCode)
    if h.k === nothing
        h.k = get_n(h) - binary_rank(get_stabilizers(h))
    end
    return h.k::Int
end

function get_d(h::HelixCode; verbose::Bool=false)
    if h.d === nothing
        tableau = get_stim_tableau(h)
        @assert h.CSS !== nothing "CSS status must be known before distance."
        h.d = code_distance(tableau; is_css=h.CSS::Bool, verbose=verbose)
    end
    return h.d::Int
end

function is_CSS(h::HelixCode)
    if h.CSS === nothing
        _, css = find_stabilizers(h.group, h.z0, h.x0)
        h.CSS = css
    end
    return h.CSS::Bool
end

get_rate(h::HelixCode) = get_k(h) / get_n(h)
get_rel_dist(h::HelixCode) = get_d(h) / get_n(h)

# Optional quick check block (mirrors the Python __main__ snippet)
if abspath(PROGRAM_FILE) == @__FILE__
    # Make some CSS codes and check if they are CSS
    CSS_group = (4, 7, 3)
    n = prod(CSS_group)
    X0 = [(0, 4, 1), (2, 3, 2)]
    Z0 = [(1, 6, 2), (3, 1, 0), (1, 1, 1)]
    println(canonicalize(CSS_group, X0, Z0))
    CSS_stabs, can_css = find_stabilizers(CSS_group, Z0, X0)
    println("Your CSS stabs are:")
    for i in 1:size(CSS_stabs,1)
        println(symp2Pauli(vec(CSS_stabs[i, :]), n))
    end
end
