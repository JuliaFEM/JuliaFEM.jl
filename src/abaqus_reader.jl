# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

element_has_nodes(::Type{Val{:C3D4}}) = 4
element_has_type( ::Type{Val{:C3D4}}) = Tet4
element_has_nodes(::Type{Val{:C3D10}}) = 10
element_has_nodes(::Type{Val{:C3D20}}) = 20
element_has_nodes(::Type{Val{:C3D20E}}) = 20
element_has_nodes(::Type{Val{:S3}}) = 3
element_has_type( ::Type{Val{:S3}}) = Seg3

"""
Checks for if line is a comment line or just empty
"""
function empty_or_comment_line{T<:AbstractString}(line::T)
    startswith(line, "**") || (length(line) == 0)
end

"""
Function for parsing nodes from the file
"""
function parse_section(model::Model,
    lines, key::Symbol, idx_start, idx_end, ::Type{Val{:NODE}})
    info("Parsing nodes")
    ids = Integer[]
    definition = lines[idx_start]
    for line in lines[idx_start + 1: idx_end]
        if !(empty_or_comment_line(line))
            m = matchall(r"[-0-9.]+", line)
            node_id = parse(Int, m[1])
            coords = float(m[2:end])
            node = Node(node_id, coords)
            add_node!(model, node)
            push!(ids, node_id)
        end
    end
    has_set_def = match(r"NSET=([\w\_\-]+)", definition) 
    if has_set_def != nothing
      set_name = has_set_def[1]
      add_set!(model, Val{:NSET}, set_name, ids)
  end
end

"""
Custon regex to find match from string. Index used if there are multiple matches
"""
function regex_match(regex_str, line, idx)
    return match(regex_str, line).captures[idx]
end

"""
Simple iterator for comsuming element list. Depending 
on the used element, connectivity nodes might be listed
in multiple lines, which is why iterator is used to handle
this problem.
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
Parse elements from input.
"""
function parse_section(model::Model, lines, key, idx_start, idx_end, ::Type{Val{:ELEMENT}})
    definition = uppercase(lines[idx_start])
    element_type = regex_match(r"TYPE=([\w\-\_]+)", definition, 1)
    eltype_sym = symbol(element_type)
    eltype_nodes = element_has_nodes(Val{eltype_sym})
    element_type = element_has_type(Val{eltype_sym})
    info("Parsing elements. Type: $(element_type)")
    list_iterator = consumeList(lines, idx_start+1, idx_end)
    ids = Integer[]
    line = consume(list_iterator)
    while line != nothing
        arr_num_as_str = matchall(r"[0-9]+", line)
        numbers = map(x-> parse(Int, x), arr_num_as_str)
        if !(empty_or_comment_line(line))
            id = numbers[1]
            connectivity = numbers[2:end]
            while length(connectivity) != eltype_nodes
                @assert length(connectivity) < eltype_nodes
                line = consume(list_iterator)
                arr_num_as_str = matchall(r"[0-9]+", line)
                numbers = map(x-> parse(Int, x), arr_num_as_str)
                map(x->push!(connectivity, x), numbers)
            end
            element = Element{element_type}(connectivity)
            add_element!(model, element)
            push!(ids, id)
        end
        line = consume(list_iterator)
    end
    has_set_def = match(r"ELSET=([\w\_\-]+)", definition) 
    if has_set_def != nothing
      set_name = has_set_def[1]
      add_set!(model, Val{:ELSET}, set_name, ids)
  end
end

"""
Parsing Node- and ElementSets. 
"""
function parse_section(model, lines, key, idx_start, idx_end, ::Union{Type{Val{:NSET}}, Type{Val{:ELSET}}})
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
    selected_set = key == :NSET ? NodeSet : ElementSet
    set_alloc = selected_set(set_name, data)
    add_set!(model, set_alloc)
end


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
Main function for parsing Abaqus input file.
"""
function parse_abaqus(fid::IOStream)
    model = Model()
    lines = readlines(fid)
    keyword_indexes = find_keywords(lines)
    idx_start = keyword_indexes[1]
    keyword_sym::Symbol = :none
    parser::Function = x->()
    for idx_end in keyword_indexes[2:end]
        keyword_line = uppercase(lines[idx_start])
        keyword = regex_match(r"\s*(\w+)", keyword_line, 1)
        k_sym = symbol(keyword)
        if method_exists(parse_section, Tuple{Model, Array{Integer, 1}, Symbol,
            Integer, Integer, Type{Val{k_sym}}})
            parse_section(model, lines, k_sym, idx_start, idx_end-1, Val{k_sym})
        else
            warn("Unknown section: $(keyword)")
        end
        idx_start = idx_end
    end
    return model
end

