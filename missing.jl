#=
missing.jl

Find missing or partial stage outputs and consolidate part files.
=#
include("constants.jl")
using JLD2

# Count part files matching STAGEx_ny_part*.jld2 in a directory.
function _count_part_files(data_dir::AbstractString, s::Int, n::Int)
    ext_rx = replace(FILE_EXT, "." => "\\.")
    rx = Regex("^$(STAGEPFX)$(s)_n$(n)_part\\d+$(ext_rx)\$")
    return count(name -> occursin(rx, name), readdir(data_dir))
end

# Load JLD2 key "data".
_load(path::AbstractString) = load(path, "data")

# Save under key "data".
_save(path::AbstractString, data) = jldsave(path; data)

function main(data_dir::AbstractString = "data")
    for s in 1:3
        missing = String[]
        for n in 1:200
            # Consolidate staged part files (only stage 3 uses parts in this project).
            if s == 3
                part0 = joinpath(data_dir, get_filename(s, n, 0))
                if isfile(part0)
                    result = Any[]
                    cur = 0
                    # Gather all contiguous parts from part0, part1, ...
                    while isfile(joinpath(data_dir, get_filename(s, n, cur)))
                        part_path = joinpath(data_dir, get_filename(s, n, cur))
                        part_data = _load(part_path)
                        if part_data isa AbstractVector
                            append!(result, part_data)
                        else
                            push!(result, part_data)
                        end
                        cur += 1
                    end
                    # If all parts are present, write the combined file; else remove stale whole file.
                    if cur == _count_part_files(data_dir, s, n)
                        whole_path = joinpath(data_dir, get_filename(s, n))
                        _save(whole_path, result)
                        continue
                    else
                        whole_path = joinpath(data_dir, get_filename(s, n))
                        if isfile(whole_path)
                            rm(whole_path)
                        end
                    end
                end
            end

            whole = joinpath(data_dir, get_filename(s, n))
            if isfile(whole)
                if s != 3
                    continue
                end
                # For stage 3, report timeouts present in the whole file.
                data = _load(whole)
                timeouts = 0
                for code in data
                    # Stage 3 tuples look like (group, z0, x0, is_css, k, d, kd_over_n)
                    # d is the 6th element in Julia.
                    if Int(code[6]) == -1
                        timeouts += 1
                    end
                end
                stage2_path = joinpath(data_dir, get_filename(2, n))
                if timeouts > 0 && isfile(stage2_path)
                    l = _load(stage2_path)
                    push!(missing, string(n, " (", timeouts, " timeouts of ", length(l), " codes)"))
                end
                continue
            end

            if s == 3
                # If stage 2 exists but stage 3 whole file does not, report count of pending codes.
                stage2_path = joinpath(data_dir, get_filename(2, n))
                if isfile(stage2_path)
                    l = _load(stage2_path)
                    push!(missing, string(n, " (", length(l), " codes)"))
                    continue
                end
            end

            push!(missing, string(n))
        end
        println("Missing ", missing, " for stage ", s)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
