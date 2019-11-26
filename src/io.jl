# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using LightXML

mutable struct Xdmf <: AbstractResultsWriter
    name :: String
    xml :: XMLElement
    hdf :: HDF5File
    hdf_counter :: Int
    format :: String
end

function Xdmf()
    return Xdmf(tempname())
end

function h5file(xdmf::Xdmf)
    return xdmf.name*".h5"
end

function xmffile(xdmf::Xdmf)
    return xdmf.name*".xmf"
end

"""
    Xdmf(name, version="3.0", overwrite=false)

Initialize a new Xdmf object.
"""
function Xdmf(name::String; version="3.0", overwrite=false)
    xdmf = new_element("Xdmf")
    h5file = "$name.h5"
    xmlfile = "$name.xmf"

    if isfile(h5file)
        if overwrite
            @debug("Result file $h5file exists, removing old file.")
            rm(h5file)
        else
            error("Result file $h5file exists, use Xdmf($name; overwrite=true) to rewrite results")
        end
    end

    if isfile(xmlfile)
        if overwrite
            @debug("Result file $xmlfile exists, removing old file.")
            rm(xmlfile)
        else
            error("Result file $xmlfile exists, use Xdmf($name; overwrite=true) to rewrite results")
        end
    end

    set_attribute(xdmf, "xmlns:xi", "http://www.w3.org/2001/XInclude")
    set_attribute(xdmf, "Version", version)
    flag = isfile(h5file) ? "r+" : "w"
    hdf = h5open(h5file, flag)
    return Xdmf(name, xdmf, hdf, 1, "HDF")
end

"""
    get_temporal_collection(xdmf)

Return the basic structure of Xdmf document.
Creates a new TemporalCollection if not found.
Basic structure for XML part of Xdmf file is
<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection">
    </Grid>
  </Domain>
</Xdmf>
"""
function get_temporal_collection(xdmf::Xdmf)
    domain = find_element(xdmf.xml, "Domain")
    grid = nothing
    if domain == nothing
        domain = new_child(xdmf.xml, "Domain")
        grid = new_child(domain, "Grid")
        set_attribute(grid, "CollectionType", "Temporal")
        set_attribute(grid, "GridType", "Collection")
    end
    grid = find_element(domain, "Grid")
    return grid
end

"""
    xdmf_filter(child_elements, child_name)

Returns some spesific child xml element from an array of XMLElement based on,
"Xdmf extensions" see [1] for details.

Parameters
----------
child_elements :: Vector{XMLElement}
    A vector of XMLElements where to perform filtering.
child_name :: String
    Child element name, maybe containing Xdmf instructions

Returns
-------
nothing if nothing is found, otherwise XMLElement matching to filtering

#Examples

julia> grid1 = new_element("Grid")
julia> add_text(grid1, "I am first grid")
julia> grid2 = new_element("Grid")
julia> add_text(grid2, "I am second grid")
julia> set_attribute(grid2, "Name", "Frame 2")
julia> grid3 = new_element("Grid")
julia> add_text(grid3, "I am third grid")
julia> grids = [grid1, grid2, grid3]

To return second Grid element, one can use

julia> xdmf_filter(grids, "Grid[2]")

To return Grid which has attribute Name="Frame 2", use

julia> xdmf_filter(grids, "Grid[@name=Frame 2]")

To pick last Grid, use [end], e.g.

julia> xdmf_filter(grids, "Grid[end]").

References
----------
[1] http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""
function xdmf_filter(child_elements, child_name)
    if '/' in child_name # needs path traversal
        return nothing
    end

    # filter children elements using syntax child[X] -> rename child_name
    m = match(r"(\w+)\[(.+)\]", child_name)
    if m != nothing
        child_name = m[1]
    end

    # first find any relevant child elements (has same tag)
    childs = []
    for child in child_elements
        if LightXML.name(child) == child_name
            push!(childs, child)
        end
    end

    # childs not found at all
    length(childs) == 0 && return nothing

    # by default return first
    m == nothing && return first(childs)

    # if [end] return last
    m[2] == "end" && return childs[end]

    # otherwise try parse int and return nth children from list
    parsed_int = tryparse(Int, m[2])
    if !isnull(parsed_int)
        idx = get(parsed_int)
        if (idx > 0) && (idx <= length(childs))
            return childs[idx]
        else
            # wrong index
            return nothing
        end
    end

    # [X] is something else than integer,  filter children elements using syntax child[@attr=value]
    m2 = match(r"@(.+)=(.+)", m[2])
    m2 == nothing && throw("Unable to parse: $(m[2])")
    attr_name = convert(String, m2[1])
    attr_value = convert(String, m2[2])
    for child in childs
        has_attribute(child, attr_name) || continue
        if attribute(child, attr_name) == attr_value
            return child
        end
    end

    # nothing found
    return nothing
end

"""
    traverse(xdmf, x, attr_name)

Traverse XML path. Xdmf filtering can be used, so it's possible to find
data from xml using syntax e.g.

#Example

julia> traverse(xdmf, x, "/Domain/Grid[2]/Grid[@Name=Frame 1]/DataItem")
"""
function traverse(xdmf::Xdmf, x::XMLElement, attr_name::String)
    attr_name = strip(attr_name, '/')

    if has_attribute(x, attr_name)
        return attribute(x, attr_name)
    end

    childs = child_elements(x)

    if '/' in attr_name
        items = split(attr_name, '/')
        new_item = xdmf_filter(childs, first(items))
        if new_item == nothing
            @debug("traverse: childs:")
            for child in childs
                @debug(LightXML.name(child))
            end
            error("traverse: failed, items = $items, xdmf_filter not find child")
        end
        new_path = join(items[2:end], '/')
        return traverse(xdmf, new_item, new_path)
    end

    child = xdmf_filter(childs, attr_name)
    return child
end

"""
    read(xdmf, path)

Read data from Xdmf file.

#Example

Traversing is supported, so one can easily traverse XML tree e.g.
julia> read(xdmf, "/Domain/Grid/Grid[2]/Geometry")
"""
function read(xdmf::Xdmf, path::String)
    result = traverse(xdmf, xdmf.xml, path)
    if endswith(path, "DataItem")
        format = attribute(result, "Format"; required=true)
        if format == "HDF"
            h5file, path = map(String, split(content(result), ':'))
            h5file = dirname(xdmf.name) * "/" * h5file
            isfile(h5file) || throw("Xdmf: h5 file $h5file not found!")
            return read(xdmf.hdf, path)
        else
            error("Read from Xdmf, reading from $format not implemented")
        end
    else
        return result
    end
end

"""
    save!(xdmf)

Save the xdmf file.
"""
function save!(xdmf::Xdmf)
    doc = XMLDocument()
    set_root(doc, xdmf.xml)
    save_file(doc, xmffile(xdmf))
end

function Base.close(xdmf::Xdmf)
    close(xdmf.hdf)
end

function new_dataitem(xdmf::Xdmf, path::String, data::Array{T,N}) where {T,N}
    dataitem = new_element("DataItem")
    datatype = replace("$T", "64" => "")
    dimensions = join(reverse(size(data)), " ")
    set_attribute(dataitem, "DataType", datatype)
    set_attribute(dataitem, "Dimensions", dimensions)
    set_attribute(dataitem, "Format", xdmf.format)
    if xdmf.format == "HDF"
        hdf = basename(h5file(xdmf))
        if exists(xdmf.hdf, path)
            @debug("Xdmf: $path already existing in h5 file, not overwriting.")
        else
            write(xdmf.hdf, path, data)
        end
        add_text(dataitem, "$hdf:$path")
    elseif xdmf.format == "XML"
        text_data = string(data')
        text_data = strip(text_data, ['[', ']'])
        text_data = replace(text_data, ';', '\n')
        text_data = "\n" * text_data * "\n"
        add_text(dataitem, text_data)
    else
        error("Unsupported Xdmf big data format $(xdmf.format)")
    end
    return dataitem
end

"""
    new_dataitem(xdmf, data)

Create a new DataItem element, hdf path automatically determined.
"""
function new_dataitem(xdmf::Xdmf, data::Array{T,N}) where {T,N}
    if xdmf.format == "XML"
        # Path can be whatever as XML format does not store to HDF at all
        return new_dataitem(xdmf, "/whatever", data)
    else
        path = "/DataItem_$(xdmf.hdf_counter)"
        while exists(xdmf.hdf, path)
            xdmf.hdf_counter += 1
            path = "/DataItem_$(xdmf.hdf_counter)"
        end
        return new_dataitem(xdmf, path, data)
    end
end

global const xdmf_element_mapping = Dict(
        "Poi1" => "Polyvertex",
        "Seg2" => "Polyline",
        "Tri3" => "Triangle",
        "Quad4" => "Quadrilateral",
        "Tet4" => "Tetrahedron",
        "Pyramid5" => "Pyramid",
        "Wedge6" => "Wedge",
        "Hex8" => "Hexahedron",
        "Seg3" => "Edge_3",
        "Tri6" => "Tri_6",
        "Quad8" => "Quad_8",
        "Tet10" => "Tet_10",
        "Pyramid13" => "Pyramid_13",
        "Wedge15" => "Wedge_15",
        "Hex20" => "Hex_20")

get_xdmf_element_code(::Element{Poi1}) = 1
get_xdmf_element_code(::Element{Seg2}) = 2
# get_xdmf_element_code(::Element{Polygon}) =  3
get_xdmf_element_code(::Element{Tri3}) = 4
get_xdmf_element_code(::Element{Quad4}) = 5
get_xdmf_element_code(::Element{Tet4}) = 6
get_xdmf_element_code(::Element{Pyr5}) = 7
get_xdmf_element_code(::Element{Wedge6}) = 8
get_xdmf_element_code(::Element{Hex8}) = 9
# get_xdmf_element_code(::Element{Polyhedron}) = 16

get_xdmf_element_code(::Element{Seg3}) = 34
get_xdmf_element_code(::Element{Quad9}) = 35
get_xdmf_element_code(::Element{Tri6}) = 36
get_xdmf_element_code(::Element{Quad8}) = 37
get_xdmf_element_code(::Element{Tet10}) = 38
# get_xdmf_element_code(::Element{Pyr13}) = 39
get_xdmf_element_code(::Element{Wedge15}) = 40
# get_xdmf_element_code(::Element{Wedge18}) = 41
get_xdmf_element_code(::Element{Hex20}) = 48
# get_xdmf_element_code(::Element{Hex24}) = 49
get_xdmf_element_code(::Element{Hex27}) = 50

"""
    get_spatial_collection()

Return a SpatialCollection at given time either by creating new one or returning
existing one.
"""
function get_spatial_collection(temporal_collection, time)
    for spatial_collection in get_elements_by_tagname(temporal_collection, "Grid")
        time_element = find_element(spatial_collection, "Time")
        time_value = Meta.parse(attribute(time_element, "Value"; required=true))
        isapprox(time_value, time) && return spatial_collection
    end
    # did not find, create new one
    spatial_collection = new_child(temporal_collection, "Grid")
    set_attribute(spatial_collection, "GridType", "Collection")
    set_attribute(spatial_collection, "Name", "Problems")
    set_attribute(spatial_collection, "CollectionType", "Spatial")
    time_element = new_child(spatial_collection, "Time")
    set_attribute(time_element, "Value", time)
    return spatial_collection
end

"""
    update_xdmf!(xdmf, problem, time, fields)

Write new fields to Xdmf file.

#Example

To write displacement and temperature fields from p1 at time t=0.0:

julia> update_xdmf!(p1, 0.0, ["displacement", "temperature"])
"""
function update_xdmf!(xdmf::Xdmf, problem::Problem, time::Float64, fields::Vector)

    @debug("Xdmf: storing fields $fields of problem $(problem.name) at time $time")

    # 1. find domain
    xml = xdmf.xml
    domain = find_element(xml, "Domain")
    if domain == nothing
        @debug("Xdmf: Domain not found, creating.")
        domain = new_child(xml, "Domain")
    end

    # 2. find for TemporalCollection
    temporal_collection = find_element(domain, "Grid")
    if temporal_collection == nothing
        @debug("Xdmf: Temporal collection not found, creating.")
        temporal_collection = new_child(domain, "Grid")
        set_attribute(temporal_collection, "GridType", "Collection")
        set_attribute(temporal_collection, "Name", "Time")
        set_attribute(temporal_collection, "CollectionType", "Temporal")
    end

    # 2.1 make sure that Grid element we found really is TemporalCollection
    collection_type = attribute(temporal_collection, "CollectionType"; required=true)
    @assert collection_type == "Temporal"

    spatial_collection = get_spatial_collection(temporal_collection, time)

    for frame in get_elements_by_tagname(spatial_collection, "Grid")
        frame_name = attribute(frame, "Name")
        if frame_name == problem.name
            @warn("Xdmf: Already found Grid with name $frame_name for time $time, skipping.")
            return
        end
    end

    frame_name = problem.name
    @debug("Xdmf: Creating Grid for problem $frame_name")
    frame = new_child(spatial_collection, "Grid")
    set_attribute(frame, "Name", frame_name)

    # 4. save geometry
    X_dict = problem("geometry", time)
    node_ids = sort(collect(keys(X_dict)))
    node_mapping = Dict(j => i for (i, j) in enumerate(node_ids))
    X_array = hcat([X_dict[nid] for nid in node_ids]...)
    ndim, nnodes = size(X_array)
    geom_type = (ndim == 2 ? "XY" : "XYZ")
    @debug("Xdmf: Creating geometry, type = $geom_type, number of nodes = $nnodes")
    X_dataitem = new_dataitem(xdmf, X_array)
    geometry = new_child(frame, "Geometry")
    set_attribute(geometry, "Type", geom_type)
    add_child(geometry, X_dataitem)

    # 5. save topology
    mesh_type = "unstructured"
    if mesh_type == "unstructured"
        element_conn = Int64[]
        for element in get_elements(problem)
            xdmf_element_code = get_xdmf_element_code(element)
            xdmf_element_code > 0 || continue
            push!(element_conn, xdmf_element_code)
            if xdmf_element_code == 2
                push!(element_conn, length(element))
            end
            for j in get_connectivity(element)
                push!(element_conn, node_mapping[j]-1)
            end
        end
        topology_dataitem = new_dataitem(xdmf, element_conn)
        topology = new_child(frame, "Topology")
        set_attribute(topology, "TopologyType", "Mixed")
        add_child(topology, topology_dataitem)
    else
        all_elements = get_elements(problem)
        nelements = length(all_elements)
        element_types = unique(map(get_element_type, all_elements))
        nelement_types = length(element_types)
        @debug("Xdmf: Saving topology of $nelements elements total, $nelement_types different element types.")
        if nelement_types != 1
            error("Xdmf: only single type of element supported by structured grid type!")
        end
        for element_type in element_types
            elements = collect(filter_by_element_type(element_type, all_elements))
            nelements = length(elements)
            @debug("Xdmf: $nelements elements of type $element_type")
            sort!(elements, by=get_element_id)
            element_ids = map(get_element_id, elements)
            element_conn = map(element -> [node_mapping[j]-1 for j in get_connectivity(element)], elements)
            element_conn = hcat(element_conn...)
            element_code = split(string(element_type), ".")[end]
            topology_dataitem = new_dataitem(xdmf, element_conn)
            topology = new_child(frame, "Topology")
            set_attribute(topology, "TopologyType", xdmf_element_mapping[element_code])
            set_attribute(topology, "NumberOfElements", length(elements))
            add_child(topology, topology_dataitem)
        end
    end

    # 6. save requested fields
    for field_name in fields
        field_dict = problem(field_name, time)
        field_center = "Node"
        field_node_ids = sort(collect(keys(field_dict)))
        if node_ids != field_node_ids
            @error("geom node ids = $node_ids")
            @error("field node ids = $field_node_ids")
            error("!=, geometry does not match with field.")
        end
        field_dim = length(field_dict[first(field_node_ids)])
        if field_dim == 2
            @debug("Xdmf: Field dimension = 2, extending to 3")
            for nid in field_node_ids
                field_dict[nid] = [field_dict[nid]; 0.0]
            end
            field_dim = 3
        end
        field_type = Dict(1 => "Scalar", 3 => "Vector", 6 => "Tensor6")[field_dim]
        @debug("Xdmf: Saving field $field_name, type = $field_type, dimension = $field_dim, center = $field_center")

        field_array = hcat([field_dict[nid] for nid in field_node_ids]...)
        field_dataitem = new_dataitem(xdmf, field_array)
        attribute = new_child(frame, "Attribute")
        set_attribute(attribute, "Name", uppercasefirst(field_name))
        set_attribute(attribute, "Center", field_center)
        set_attribute(attribute, "AttributeType", field_type)
        add_child(attribute, field_dataitem)
    end

    save!(xdmf)
    @debug("Xdmf: all done.")
end
