#=
constants.jl

Constants and helpers for filename conventions.
=#
const STAGEPFX::String = "STAGE"
const FILE_EXT::String = ".jld2"

generate_identifier(n::Integer)::String = "n$(n)"

function filename(stage::Integer, n::Integer; part::Union{Nothing,Integer,AbstractString}=nothing)::String
    id = generate_identifier(n)
    if stage == 3 && part !== nothing
        return "$(STAGEPFX)$(stage)_$(id)_part$(part)$(FILE_EXT)"
    else
        return "$(STAGEPFX)$(stage)_$(id)$(FILE_EXT)"
    end
end

# Backwards-compatible shim
get_filename(stage, n, part::Union{Nothing,Int,AbstractString}=nothing) = filename(stage, n; part=part)

const RATE_THRESHOLD::Float64 = 1 / 16
const DISTANCE_THRESHOLD::Int = 4
const DISTANCE_RATE_THRESHOLD::Float64 = 0.4  # R*d > c*n
