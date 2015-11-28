# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

global ABQ_ELTYPE_HAS_NODES = Dict(
    :C3D10 => 10,
    :C3D4 => 4,
)


function parse_nodes(model, header, data)
    nodes = create_or_get(model, "nodes")
    for line in split(data, "\n")
        m = matchall(r"[-0-9.]+", line)
        id = parse(Int, m[1])
        coords = float(m[2:end])
        nodes[id] = coords
    end
end

#function parse_elements(model, header, data)
#    info("Parsing elements")
#    eldims = Dict(
#        "C3D10" => 10,
#        "C3D4" => 4)
#    eltype = header["options"]["TYPE"]
#    if !(eltype in keys(eldims))
#        throw("Element $eltype dimension information missing")
#    end
#    eldim = eldims[eltype]
#    test_match = matchall(r"[0-9]+", "234, 242")
#    m = matchall(r"[0-9]+", data)   
#    m = map((s) -> parse(Int, s), m)
#    elements = create_or_get(model, "elements")
#    m = reshape(m, eldim+1, round(Int, length(m)/(eldim+1)))
#    nel = size(m)[2]
#    info("$nel elements found")
#    for i=1:nel
#        elements[m[1,i]] = m[2:end,i]
#    end
#    if "ELSET" in keys(header["options"])
#        elsets = create_or_get(model, "elsets")
#        elset_name = header["options"]["ELSET"]
#        info("Creating ELSET $elset_name")
#        elsets[elset_name] = Int64[]
#        for i=1:nel
#            push!(elsets[elset_name], m[1,i])
#        end
#    end
#end
#
#function parse_nodeset(model, header, data)
#    nset_name = header["options"]["NSET"]
#    info("Creating node set $nset_name")
#    data = ASCIIString[]
#    m = matchall(r"[0-9]+", data)
#    node_ids = map((s) -> parse(Int, s), m)
#    nsets = create_or_get(model, "nsets")
#    nsets[nset_name] = Int64[]
#    for j in node_ids
#        push!(nsets[nset_name], j)
#    end
#end

parse_section = Dict(:NODE => parse_nodes) # ,
                     #:ELEMENT => parse_elements,
                     #:NSET => parse_nodeset,
                     #:ELSET => parse_elementset)

"""
Find lines, which contain keywords, for example "*NODE"
"""
function find_keywords(lines)
    indexes = Integer[]
    for (idx, line) in enumerate(lines)
        if startswith(line, "*") && !startswith(line, "**")
            push!(indexes, idx)
        end
    end
    push!(indexes, length(lines))
    indexes
end

function parse_abaqus(fid::IOStream)
#    model = Dict()
    lines = readlines(fid)
    info("Registered handlers: $(keys(parse_section))")
    keyword_indexes = find_keywords(lines)
    idx_start = keyword_indexes[1]
    keyword_sym::Symbol = :none
    for idx_end in keyword_indexes[2:end]
        keyword_line = lines[idx_start]
        keyword = match(r"\s*(\w+)", keyword_line).match
        keyword_sym = :($(uppercase(keyword)))
        try
            parser = parse_section(keyword_sym)
        catch
            warn("Keyword not implemented: ", keyword_sym)
            idx_start = idx_end
            continue
        end
        idx_start = idx_end
#        parser(model, lines, idx_start, idx_end)
    end
#    return model
end

