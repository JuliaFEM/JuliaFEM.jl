# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

global ABQ_ELTYPE_HAS_NODES = Dict(
    :C3D4 => 4,
    :C3D10 => 10,
    :C3D20 => 20,
    :C3D20E => 20,
    :S3 => 3,
)


"""
"""
function empty_or_comment_line(line)
    val_bool = false
    if startswith(line, "**") || length(line) == 0
        val_bool = true
    end
    val_bool
end

"""
Function for parsing nodes from the file
"""
function parse_nodes(model, lines, key, idx_start, idx_end)
    info("Parsing nodes")
    definition = lines[idx_start]
    for line in lines[idx_start + 1: idx_end]
        if !(empty_or_comment_line(line))
            m = matchall(r"[-0-9.]+", line)
            node_id = parse(Int, m[1])
            coords = float(m[2:end])
            node = jNode(node_id, coords)
            push!(model.nodes, node)
        end
    end
end

"""
Custon regex to find match from string
"""
function regex_match(regex_str, line, idx)
    return match(regex_str, line).captures[idx]
end

"""
"""
function consumeList(arr, start, stop)
    function _it()
        for i=start:stop
            produce(arr[i])
        end
    end
    Task(_it)
end

"""
"""
function parse_elements(model, lines, key, idx_start, idx_end)
    definition = uppercase(lines[idx_start])
    element_type = regex_match(r"TYPE=([\w\-\_]+)", definition, 1)
    eltype_sym = symbol(element_type)
    eltype_nodes = ABQ_ELTYPE_HAS_NODES[eltype_sym]
    info("Parsing elements. Type: $(element_type)")
    list_iterator = consumeList(lines, idx_start+1, idx_end)
    line = consume(list_iterator)
    while line != nothing
        arr_num_as_str = matchall(r"[0-9]+", line)
        numbers = map(x-> parse(Int, x), arr_num_as_str)
        if !(empty_or_comment_line(line))
            id = numbers[1]
            connectivity = numbers[2:end]
            element = jElement(id, connectivity, element_type)
            while length(element.connectivity) != eltype_nodes
                @assert length(element.connectivity) < eltype_nodes
                line = consume(list_iterator)
                arr_num_as_str = matchall(r"[0-9]+", line)
                numbers = map(x-> parse(Int, x), arr_num_as_str)
                map(x->push!(element.connectivity, x), numbers)
            end
            push!(model.elements, element)
        end
        line = consume(list_iterator)
    end
end

"""
"""
function parse_set(model, lines, key, idx_start, idx_end)
    set_regex_string = Dict(:NSET  => r"NSET=([\w\-\_]+)",
                            :ELSET => r"ELSET=([\w\-\_]+)" )
    definition = uppercase(lines[idx_start])
    regex_string = set_regex_string[key]
    set_name = regex_match(regex_string, definition, 1)
    info("Creating $(lowercase(string(key))) $set_name")
    data = Integer[]
    if endswith(strip(definition), "GENERATE")
        line = lines[idx_start + 1]
        m = matchall(r"[0-9]+", line)
        first_id, last_id, step = map(x-> parse(Int, x), m)
        set_ids = collect(first_id:step:last_id)
        map(x->push!(data, x), set_ids)
    else
        for line in lines[idx_start + 1: idx_end]
            if !(empty_or_comment_line(line))
                m = matchall(r"[0-9]+", line)
                set_ids = map(s -> parse(Int, s), m)
                map(x->push!(data, x), set_ids)
            end
        end
    end
    selected_set = key == :NSET ? jNodeSet : jElementSet
    set_alloc = selected_set(set_name, data)
    model.sets[set_name] = set_alloc
end

# 
parse_section = Dict(:NODE => parse_nodes,
                     :ELEMENT => parse_elements,
                     :NSET => parse_set,
                     :ELSET => parse_set)

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

"""
"""
function parse_abaqus(fid::IOStream)
    model = jModel()
    lines = readlines(fid)
    info("Registered handlers: $(keys(parse_section))")
    keyword_indexes = find_keywords(lines)
    idx_start = keyword_indexes[1]
    keyword_sym::Symbol = :none
    parser::Function = x->()
    for idx_end in keyword_indexes[2:end]
        keyword_line = uppercase(lines[idx_start])
        keyword = regex_match(r"\s*(\w+)", keyword_line, 1)
        keyword_sym = symbol(keyword)
        try
            parser = parse_section[keyword_sym]
        catch
            warn("Keyword not implemented: ", keyword_sym)
            idx_start = idx_end
            continue
        end
        parser(model, lines, keyword_sym, idx_start, idx_end-1)
        idx_start = idx_end
    end
    return model
end

