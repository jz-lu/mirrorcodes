#=
missing.jl

Find missing or partial stage outputs and consolidate part files.
=#
include("constants.jl")
using JLD2

for s in 1:3
    missing = String[]
    for i in 1:200
        if isfile(joinpath("data", get_filename(s, i, 0)))
            result = Any[]
            cur = 0
            while isfile(joinpath("data", get_filename(s, i, cur)))
                fpath = joinpath("data", get_filename(s, i, cur))
                data = load(fpath, "data")
                data isa AbstractVector ? append!(result, data) : push!(result, data)
                cur += 1
            end
            # count parts
            ext_rx = replace(FILE_EXT, "." => "\\.")
            rx = Regex("^$(STAGEPFX)$(s)_n$(i)_part\\d+$(ext_rx)\$")
            n_parts = count(name -> occursin(rx, name), readdir("data"))
            if cur == n_parts
                savepath = joinpath("data", get_filename(s, i))
                jldsave(savepath; data=result)
                continue
            elseif isfile(joinpath("data", get_filename(s, i)))
                rm(joinpath("data", get_filename(s, i)))
            end
        end
        if isfile(joinpath("data", get_filename(s, i)))
            if s != 3
                continue
            end
            data = load(joinpath("data", get_filename(3, i)), "data")
            timeouts = count(code -> Int(code[6]) == -1, data)
            l = load(joinpath("data", get_filename(2, i)), "data")
            if timeouts > 0
                push!(missing, "$(i) ($(timeouts) timeouts of $(length(l)) codes)")
            end
            continue
        end
        if s == 3 && isfile(joinpath("data", get_filename(2, i)))
            l = load(joinpath("data", get_filename(2, i)), "data")
            push!(missing, "$(i) ($(length(l)) codes)")
        else
            push!(missing, string(i))
        end
    end
    println("Missing ", missing, " for stage ", s)
end
