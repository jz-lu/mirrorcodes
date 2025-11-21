#=
`constants.jl`

Code file storing all the constants, e.g. filename conventions,
used in this library. There are also some functions which generate constants.
=#
const STAGEPFX = "STAGE"
const FILE_EXT = ".jld2"

generate_identifier(n)::String = "n$(n)"

function get_filename(stage, n, part::Union{Nothing,Int,AbstractString}=nothing)
    id = generate_identifier(n)
    if stage == 3 && part !== nothing
        return "$(STAGEPFX)$(stage)_$(id)_part$(part)$(FILE_EXT)"
    end
    return "$(STAGEPFX)$(stage)_$(id)$(FILE_EXT)"
end

const RATE_THRESHOLD::Float64 = 1/16
const DISTANCE_THRESHOLD::Int = 4
const DISTANCE_RATE_THRESHOLD::Float64 = 0.4  # const c such that we want R*d > c*n
