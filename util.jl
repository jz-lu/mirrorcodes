#=
util.jl

Code file with a bunch of simple helper functions that process and convert between
stim, Pauli string, and symplectic representations of codes.
=#
using PythonCall
using LinearAlgebra

pyimport("pip").main(["install", "stim"])
const stim = pyimport("stim")

#=
Return a sign-free Pauli string representation of the length 2`n` symplectic vector `x`.

Input:
    * n (Int): number of qubits.
    * x (Array): binary vector of length 2n, symplectically representing a n-qubit Pauli string.

The convention for the symplectic vector is [Z | X] .
    
Returns:
    * length-n string over {I, X, Y, Z} where the ith character is the Pauli on the ith qubit
=#
function symp2Pauli(x, n)
    vec = Vector{Char}(undef, n)
    @inbounds for i in 1:n
        char = 'I'
        if x[i] == 0 && x[i+n] == 1
            char = 'X'
        elseif x[i] == 1 && x[i+n] == 1
            char = 'Y'
        elseif x[i] == 1 && x[i+n] == 0
            char = 'Z'
        end
        vec[i] = char
    end
    String(vec)
end

#=
Convert a stabilizer tableau in Pauli string notation into stim convention.
=#
function stimify_stabs(stabs)
    out = PythonCall.PyList()
    for x in stabs
        out.append(stim.PauliString(String(x)))
    end
    out
end

#=
Convert a symplectic tableau into stim convention.
=#
function stimify_symplectic(stabs)
    @assert length(stabs[1]) % 2 == 0 && length(stabs[1]) > 0
    n = div(length(stabs[1]), 2)
    stabs2 = [symp2Pauli(vec, n) for vec in stabs]
    stimify_stabs(stabs2)
end

#=
Test whether a stabilizer tableau is CSS. The code is defined to be CSS
if every check is either all X's or all Z's.

Params:
    * stabs (Array): r x 2n binary matrix giving the stabilizer tableau in symplectic representation.
    * n (Int): number of qubits.
    
Returns:
    * bool indicating if `stabs` represents a CSS code.
=#
function is_CSS(stabs, n)
    is_unmixed(x) = all(x[1:n] .== 0) || all(x[n+1:end] .== 0)
    all(is_unmixed(vec) for vec in eachrow(stabs))
end

#=
Finds index incrementing values for a given group.

Params:
    * group (Array): A list containing the group.

Returns:
    * Array with index incrementing values, the backwards cumulative product.
=#
function find_strides(group)
    if length(group) == 1
        return [1]
    end
    v = reverse!(cumsum(reverse!(log.(group)); dims=1)) # placeholder to allocate
    # compute strides explicitly
    L = length(group)
    s = Vector{Int}(undef, L)
    s[L] = 1
    for i in (L-1):-1:1
        s[i] = s[i+1] * Int(group[i+1])
    end
    s
end

#=
Finds all partitions of n. I a lower bound on elements of the partitions,
which is 1 by default. This function should be moved to util.py
    
Input:
    * n (Int): the number whose partitions we want to compute.
    * I (Int, optional): the minimum entry in the partitions, 1 by default
    
Returns:
    * Iterable of tuples containing the partitions of n with minimum I or more
=#
function partitions(n, I=1)
    function rec(rem, minv, acc, out)
        push!(out, (acc..., rem))
        for i in minv:(rem ÷ 2)
            rec(rem - i, i, (acc..., i), out)
        end
        out
    end
    rec(n, I, (), Tuple[])
end

#=
Compute a tuple representing a qubit in group, given an index from 0 to n - 1,
the size of the group. This is no different than expressing a number "base
group". Notably, this works for larger indices too, but will only consider the
index mod n.

Params:
    * group (Array): the group we are decomposing the index into
    * index (Int): A number from 0 to n - 1 corresponding to a tuple mod group.
    
Returns:
    * A tuple with the same length as group, corresponding to the indexth
    tuple mod group.
=#
function index_to_tuple(group, index)
    n = prod(group)
    idx = mod(index, n)
    result = Vector{Int}(undef, length(group))
    for gidx in length(group):-1:1
        g = Int(group[gidx])
        result[gidx] = mod(idx, g)
        idx ÷= g
    end
    Tuple(result)
end

#=
Find the gcd of args.

Params:
    * args (any number of ints): values whose gcd we want to find

Returns:
    * Int containing the gcd of args.
=#
function gcd_list(args...)
    result = Int(args[1])
    for num in args[2:end]
        result = gcd(result, Int(num))
    end
    result
end

#=
Find all values relatively prime to group, with threading over a list.
    
Params:
    * group (Int or Array): group size or list of group sizes

Returns:
    * List (of lists) of values relatively prime to each group size
=#
function find_isos(group)
    if group isa Integer
        g = Int(group)
        return [i for i in 1:g-1 if gcd(i, g) == 1]
    end
    [ [i for i in 1:Int(g)-1 if gcd(i, Int(g)) == 1] for g in group ]
end

#=
Finds the rank mod 2 of a matrix A without explicit row swapping,
using XOR to absorb the pivot row.

Params:
    * A (Array): the matrix whose rank we want to find

Returns:
    * int containing the rank of A
=#
function binary_rank(A)
    M = UInt8.(A .& 1)
    m, n = size(M)
    r = 0
    for c in 1:n
        pivot = 0
        for i in r+1:m
            if M[i, c] == 0x01
                pivot = i
                break
            end
        end
        pivot == 0 && continue
        if pivot != r + 1
            M[r+1, :] .= xor.(M[r+1, :], M[pivot, :])
        end
        for i in r+2:m
            if M[i, c] == 0x01
                M[i, :] .= xor.(M[i, :], M[r+1, :])
            end
        end
        r += 1
        r == m && break
    end
    r
end

#=
Finds an upper bound for the rank of the stabilizer matrix by just using one
side of the parity checks.

Params:
    * group (Array): the group we are counting our qubits in
    * qubits (Array): the qubits making up Z0 or X0, mod group

Returns:
    * int which is the rank of the binary matrix with shifts of qubits
=#
function compute_rank_from_tuples(group, qubits)
    n = prod(group)
    matrix = zeros(UInt8, n, n)
    strides = find_strides(group)
    i = 0
    for j in Base.Iterators.product((0:(Int(g)-1) for g in group)...)
        i += 1
        matrix[i, sum((mod.(qubits .+ collect(j), group)) * strides) + 1] = 0x01
    end
    binary_rank(matrix)
end

#=
Method for shifting x0 to make the first element have 0's or 1's.

Params:
    * group (Array): the group we are counting our qubits in
    * x0 (Array): the qubits making up X0, mod group

Returns:
    * shifted version of x0
=#
function shift_X(group, x0)
    shift_bump = [(g % 2 == 0 && x0[1, i] % 2 == 1) ? 1 : 0 for (i, g) in enumerate(group)]
    mod.(x0 .- x0[1, :] .+ shift_bump, group)
end
