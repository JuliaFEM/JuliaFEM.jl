# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


function aster_parse_nodes(section::ASCIIString; strip_characters=true)
    nodes = Dict{Any, Vector{Float64}}()
    has_started = false
    for line in split(section, '\n')
        m = matchall(r"[\w.-]+", line)
        if (length(m) != 1) && (!has_started)
            continue
        end
        if length(m) == 1
            if (m[1] == "COOR_2D") || (m[1] == "COOR_3D")
                has_started = true
                continue
            end
            if m[1] == "FINSF"
                break
            end
        end
        if length(m) == 4
            nid = m[1]
            if strip_characters
                nid = matchall(r"\d", nid)
                nid = parse(Int, nid[1])
            end
            nodes[nid] = float(m[2:end])
        end
    end
    return nodes
end

function parse(mesh::ASCIIString, ::Type{Val{:CODE_ASTER_MAIL}})
    model = Dict{ASCIIString, Any}()
    header = nothing
    data = ASCIIString[]
    for line in split(mesh, '\n')
        length(line) != 0 || continue
        info("line: $line")
        if is_aster_mail_keyword(strip(line))
            header = parse_aster_header(line)
            empty!(data)
            continue
        end
        if line == "FINSF"
            info(data)
            header = nothing
            process_aster_section!(model, join(data, ""), header, Val{header[1]})
        end
    end
    return model
end
