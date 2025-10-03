#=
test_cases.jl
=#
include("util.jl")
include("distance.jl")

using ArgParse
using PythonCall

const stim = pyimport("stim")

function find_stabilizers(group, z0, x0)
    n = Int(prod(group))
    d = length(group)
    strides = find_strides(collect(group))
    M = zeros(UInt8, n, 2n)

    ranges = (0:(Int(g)-1) for g in group)
    i = 0
    for g in Iterators.product(ranges...)
        i += 1
        # Z part
        for p in z0
            col = 0
            for t in 1:d
                col += mod(Int(p[t]) + Int(g[t]), Int(group[t])) * strides[t]
            end
            M[i, col + 1] = 0x01
        end
        # X part
        for p in x0
            col = 0
            for t in 1:d
                col += mod(Int(p[t]) - Int(g[t]), Int(group[t])) * strides[t]
            end
            M[i, col + 1 + n] = 0x01
        end
    end
    return M
end

"""
Get the stim stabilizers of a given code, and whether or not they are CSS.

Params:
    * code (str): description of the code. There are a few options.
    
Returns:
    * stabs (list[stim.PauliString]): stabilizer tableau in stim form (list of stim.PauliString objects)
    * IS_CSS (bool): bit indicating if code is CSS.
"""
function get_stabilizers(code::AbstractString)
    stabs = nothing
    IS_CSS = false
    if code == "5qubit"
        stabs = stimify_stabs(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    elseif code == "repetition"
        stabs = stimify_stabs(["ZZI", "ZIZ"])
        IS_CSS = true
    elseif code == "cookie"
        group = (6, 12)
        z0 = [(0, 0), (0, 1), (-3, 2)]
        x0 = [(-1, -1), (-2, -1), (-3, 2)]
        stabs_symp = find_stabilizers(group, z0, x0)
        # Convert matrix rows to vectors for stimify_symplectic
        rows = [vec(stabs_symp[i, :]) for i in 1:size(stabs_symp, 1)]
        stabs = stimify_symplectic(rows)
    elseif code == "BB"
        IS_CSS = true
        offsets = Set([1 + 0im, -1 + 0im, 0 + 1im, 0 - 1im, 3 + 6im, -6 + 3im])
        w = 24
        h = 12
        q = w * h ÷ 2

        wrap(c::Complex) = Complex(mod(real(c), w), mod(imag(c), h))
        index_of(c::Complex) = begin
            cw = wrap(c)
            Int((real(cw) + imag(cw) * w) ÷ 2)
        end

        plist = PythonCall.PyList()
        for x in 0:w-1, y in 0:h-1
            if (x % 2) != (y % 2)
                continue  # This is a data qubit.
            end
            m = Complex(x, y)
            basis = (x % 2 == 0) ? 'X' : 'Z'
            sign = (basis == 'Z') ? -1 : 1
            chars = fill('I', q)
            for off in offsets
                idx = index_of(m + off * sign) + 1  # 1-based
                chars[idx] = basis
            end
            ps = stim.PauliString(String(chars))
            plist.append(ps)
        end
        stabs = plist
    else
        error("Unrecognized code name $code")
    end
    return stabs, IS_CSS
end

function main(args)
    code = args["code"]
    stabs, IS_CSS = get_stabilizers(code)
    dist = code_distance(stabs; is_css=IS_CSS, verbose=true)
    return dist
end

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings(; prog="test_cases", description="Test the distance of some CSS and non-CSS codes")
    @add_arg_table! s begin
        "--code", "-c"
            arg_type = String
            default = "BB"
    end
    args = parse_args(s)
    main(args)
end
