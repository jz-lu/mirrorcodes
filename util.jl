#=
util.jl

Code file with a bunch of simple helper functions that process and convert between
stim, Pauli string, and symplectic representations of codes.
=#
using LinearAlgebra
using PythonCall

const stim = pyimport("stim")

"""
Return a sign-free Pauli string representation of the length 2n symplectic vector `x`.

Input:
    * n (int): number of qubits.
    * x (Vector{<:Integer}): binary vector of length 2n, symplectically representing an n-qubit Pauli string.

The convention for the symplectic vector is [Z | X] .
    
Returns:
    * length-n string over {I, X, Y, Z} where the ith character is the Pauli on the ith qubit
"""
function symp2Pauli(x::AbstractVector{<:Integer}, n::Integer)::String
    @assert length(x) == 2n
    z = x[1:n]
    y = x[n+1:2n]
    chars = Vector{Char}(undef, n)
    @inbounds for i in 1:n
        if z[i] == 0 && y[i] == 1
            chars[i] = 'X'
        elseif z[i] == 1 && y[i] == 1
            chars[i] = 'Y'
        elseif z[i] == 1 && y[i] == 0
            chars[i] = 'Z'
        else
            chars[i] = 'I'
        end
    end
    return String(chars)
end

"""
Convert a stabilizer tableau in Pauli string notation into stim convention.
"""
function stimify_stabs(stabs)::Any
    out = PythonCall.PyList()
    for s in stabs
        out.append(stim.PauliString(s))
    end
    return out
end

"""
Convert a symplectic tableau into stim convention.
"""
function stimify_symplectic(stabs)::Any
    @assert !isempty(stabs)
    @assert length(stabs[1]) % 2 == 0
    n = length(stabs[1]) ÷ 2
    paulis = (symp2Pauli(vec, n) for vec in stabs)
    return stimify_stabs(paulis)
end

"""
Test whether a stabilizer tableau is CSS. The code is defined to be CSS
if every check is either all X's or all Z's.

Params:
    * stabs (Matrix{<:Integer}): r x 2n binary matrix giving the stabilizer tableau in symplectic representation.
    * n (int): number of qubits.
    
Returns:
    * bool indicating if `stabs` represents a CSS code.
"""
function is_CSS(stabs::AbstractMatrix{<:Integer}, n::Integer)::Bool
    @assert size(stabs, 2) == 2n
    unmixed(row) = all(row[1:n] .== 0) || all(row[n+1:2n] .== 0)
    return all(unmixed(view(stabs, i, :)) for i in 1:size(stabs, 1))
end

"""
Finds index incrementing values for a given group.

Params:
    * group (Vector{<:Integer}): A list containing the group.

Returns:
    * Array with index incrementing values, the backwards cumulative product.
"""
function find_strides(group::AbstractVector{<:Integer})::Vector{Int}
    L = length(group)
    L == 1 && return [1]
    strides = Vector{Int}(undef, L)
    prod_tail = 1
    for i in L:-1:1
        strides[i] = prod_tail
        prod_tail *= group[i]
    end
    return strides
end

"""
Finds all partitions of n. I a lower bound on elements of the partitions,
which is 1 by default. This function should be moved to util.py

Input:
    * n (int): the number whose partitions we want to compute.
    * I (int, optional): the minimum entry in the partitions, 1 by default

Returns:
    * Iterable of tuples containing the partitions of n with minimum I or more
"""
function partitions(n::Int; I::Int=1)
    return Channel{NTuple{N,Int}}(ctype = NTuple) do ch
        # Emit as tuples of variable length via vectors then convert.
        function _gen(rem::Int, minv::Int, acc::Vector{Int})
            put!(ch, Tuple(vcat(acc, rem)))
            for i in minv:rem ÷ 2
                _gen(rem - i, i, vcat(acc, i))
            end
        end
        _gen(n, I, Int[])
    end
end

"""
Compute a tuple representing a qubit in group, given an index from 0 to n - 1,
the size of the group. This is no different than expressing a number "base
group". Notably, this works for larger indices too, but will only consider the
index mod n.

Params:
    * group (Vector{<:Integer}): the group we are decomposing the index into
    * index (int): A number from 0 to n - 1 corresponding to a tuple mod group.
    
Returns:
    * A tuple with the same length as group, corresponding to the indexth
      tuple mod group.
"""
function index_to_tuple(group::AbstractVector{<:Integer}, index::Integer)
    n = prod(group)
    idx = mod(index, n)
    res = Vector{Int}(undef, length(group))
    for (ri, g) in Iterators.reverse(enumerate(group))
        res[ri] = idx % g
        idx ÷= g
    end
    return Tuple(res)
end

"""
Find the gcd of args.

Params:
    * args (any number of ints): values whose gcd we want to find

Returns:
    * Int containing the gcd of args.
"""
function gcd_list(x::Integer, ys::Integer...)::Int
    g = Int(x)
    for y in ys
        g = gcd(g, Int(y))
    end
    return g
end

"""
Find all values relatively prime to group, with threading over a list.
    
Params:
    * group (int or Vector{Int}): group size or list of group sizes

Returns:
    * List (of lists) of values relatively prime to each group size
"""
function find_isos(group::Integer)
    g = Int(group)
    return [i for i in 1:g-1 if gcd_list(i, g) == 1]
end
function find_isos(group::AbstractVector{<:Integer})
    return [[i for i in 1:Int(g)-1 if gcd_list(i, Int(g)) == 1] for g in group]
end

"""
Finds the rank mod 2 of a matrix A without explicit row swapping,
using XOR to absorb the pivot row.

Params:
    * A (Matrix{<:Integer}): the matrix whose rank we want to find

Returns:
    * int containing the rank of A
"""
function binary_rank(A::AbstractMatrix{<:Integer})::Int
    M = UInt8.(A .& 1)
    m, n = size(M)
    r = 0
    for c in 1:n
        # Find pivot at or below row r+1.
        pivot = 0
        for i in r+1:m
            if M[i, c] == 0x01
                pivot = i
                break
            end
        end
        pivot == 0 && continue
        # Move pivot to row r+1 by XOR if needed (same effect without swapping).
        if pivot != r + 1
            @inbounds M[r+1, :] .= xor.(M[r+1, :], M[pivot, :])
        end
        # Eliminate below.
        for i in r+2:m
            if M[i, c] == 0x01
                @inbounds M[i, :] .= xor.(M[i, :], M[r+1, :])
            end
        end
        r += 1
        r == m && break
    end
    return r
end

"""
Finds an upper bound for the rank of the stabilizer matrix by just using one
side of the parity checks.

Params:
    * group (Vector{<:Integer}): the group we are counting our qubits in
    * qubits (Matrix{<:Integer}): the qubits making up Z0 or X0, mod group

Returns:
    * int which is the rank of the binary matrix with shifts of qubits
"""
function compute_rank_from_tuples(group::AbstractVector{<:Integer}, qubits::AbstractMatrix{<:Integer})::Int
    n = prod(group)
    M = zeros(UInt8, n, n)
    strides = find_strides(group)               # length == length(group)
    # Iterate j over Cartesian product of 0:(g-1) for each dimension.
    ranges = (0:(Int(g)-1) for g in group)
    row_idx = 0
    for j in Iterators.product(ranges...)
        row_idx += 1
        # For each qubit vector q (a row of `qubits`), mark its shifted column.
        for r in 1:size(qubits, 1)
            # Compute column index: ((q + j) mod group) ⋅ strides
            col = 0
            @inbounds for d in 1:length(group)
                val = mod(Int(qubits[r, d]) + Int(j[d]), Int(group[d]))
                col += val * strides[d]
            end
            M[row_idx, col + 1] = 0x01
        end
    end
    return binary_rank(M)
end

"""
Method for shifting x0 to make the first element have 0's or 1's.

Params:
    * group (Vector{<:Integer}): the group we are counting our qubits in
    * x0 (Matrix{<:Integer}): the qubits making up X0, mod group

Returns:
    * shifted version of x0
"""
function shift_X(group::AbstractVector{<:Integer}, x0::AbstractMatrix{<:Integer})
    bump = [ (g % 2 == 0 && (Int(x0[1, i]) % 2 == 1)) ? 1 : 0 for (i, g) in enumerate(group) ]
    X = similar(x0, Int, size(x0))
    @inbounds for r in 1:size(x0, 1), c in 1:size(x0, 2)
        X[r, c] = mod(Int(x0[r, c]) - Int(x0[1, c]) + bump[c], Int(group[c]))
    end
    return X
end
