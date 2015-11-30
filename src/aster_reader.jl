# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md



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
