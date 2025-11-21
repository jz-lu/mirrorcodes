#=
test_cases.jl

Test the distance of some CSS and non-CSS codes
=#
include("util.jl")
include("distance.jl")
using ArgParse
using PythonCall

const stim = pyimport("stim")

function find_stabilizers(group, z0, x0)
    n = Int(prod(group))
    d = length(group)
    strides = zeros(Int, d)
    strides[1:end-1] .= cumsum(reverse!(Int.(group)))[end-1:-1:2]
    strides[end] = 1
    stabilizers = zeros(UInt8, n, 2n)
    idx = 0
    for g in Base.Iterators.product((0:(Int(a)-1) for a in group)...)
        idx += 1
        stabilizers[idx, sum((mod.(z0 .+ collect(g), group)) * strides) + 1] = 0x01
        stabilizers[idx, sum((mod.(x0 .- collect(g), group)) * strides) + 1 .+ n] = 0x01
    end
    stabilizers
end

#=
Get the stim stabilizers of a given code, and whether or not they are CSS.

Params:
    * code (str): description of the code. There are a few options.
    
Returns:
    * stabs (list[stim.PauliString]): stabilizer tableau in stim form (list of stim.PauliString objects)
    * IS_CSS (bool): bit indicating if code is CSS.
=#
function get_stabilizers(code)
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
        rows = [vec(stabs_symp[i, :]) for i in 1:size(stabs_symp, 1)]
        stabs = stimify_symplectic(rows)
    elseif code == "BB"
        IS_CSS = true
        offsets = Set([1, -1, 0 + 1im, 0 - 1im, 3 + 6im, -6 + 3im])
        w = 24
        h = 12
        wrap(c) = Complex(mod(real(c), w), mod(imag(c), h))
        index_of(c) = Int((real(wrap(c)) + imag(wrap(c)) * w) ÷ 2)
        stabs_list = PythonCall.PyList()
        for x in 0:w-1
            for y in 0:h-1
                if x % 2 != y % 2
                    continue
                end
                m = Complex(x, y)
                basis = (x % 2 == 0) ? 'X' : 'Z'
                sign = basis == 'Z' ? -1 : 1
                st = fill('I', w * h ÷ 2)
                for off in offsets
                    st[index_of(m + off * sign) + 1] = basis
                end
                stabs_list.append(stim.PauliString(String(st)))
            end
        end
        stabs = stabs_list
    else
        error("Unrecognized code name $code")
    end
    return stabs, IS_CSS
end

function main(args)
    code = args["code"]
    stabs, IS_CSS = get_stabilizers(code)
    dist = distance(stabs; IS_CSS=IS_CSS, verbose=true)
    return dist
end

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings(; prog="test_cases", description="Test the distance of some CSS and non-CSS codes")
    @add_arg_table! s begin
        "--code", "-c"; arg_type=String; default="BB"
    end
    args = parse_args(s)
    main(args)
end
