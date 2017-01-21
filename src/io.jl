# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using LightXML

type Xdmf
    name :: String
    xml :: XMLElement
    hdf :: HDF5File
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

""" Initialize a new Xdmf object. """
function Xdmf(name::String; overwrite=false)
    xdmf = new_element("Xdmf")
    h5file = "$name.h5"
    xmlfile = "$name.xmf"
    
    if isfile(h5file)
        if overwrite
            info("Result file $h5file exists, removing old file.")
            rm(h5file)
        else
            error("Result file $h5file exists, use Xdmf($name; overwrite=true) to rewrite results")
        end
    end
    
    if isfile(xmlfile)
        if overwrite
            info("Result file $xmlfile exists, removing old file.")
            rm(xmlfile)
        else
            error("Result file $xmlfile exists, use Xdmf($name; overwrite=true) to rewrite results")
        end
    end

    set_attribute(xdmf, "xmlns:xi", "http://www.w3.org/2001/XInclude")
    set_attribute(xdmf, "Version", "2.1")
    flag = isfile(h5file) ? "r+" : "w"
    hdf = h5open(h5file, flag)
    return Xdmf(name, xdmf, hdf)
end

""" Return the basic structure of Xdmf document. Creates a new TemporalCollection if not found.

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
        debug("Xdmf: creating new temporal collection")
        domain = new_child(xdmf.xml, "Domain")
        grid = new_child(domain, "Grid")
        set_attribute(grid, "CollectionType", "Temporal")
        set_attribute(grid, "GridType", "Collection")
    end
    grid = find_element(domain, "Grid")
    return grid
end

""" Returns some spesific child xml element from a array of XMLElement based on,
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

Examples
--------
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

""" Traverse XML path. Xdmf filtering can be used, so it's possible to find
data from xml using syntax e.g. 

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
        new_path = join(items[2:end], '/')
        return traverse(xdmf, new_item, new_path)
    end
    
    child = xdmf_filter(childs, attr_name)
    return child
end

""" Read data from Xdmf file.

Traversing is supported, so one can easily traverse XML tree e.g.
julia> read(xdmf, "/Domain/Grid/Grid[2]/Geometry")
"""
function read(xdmf::Xdmf, path::String)
    result = traverse(xdmf, xdmf.xml, path)
    if endswith(path, "DataItem")
        format = attribute(result, "Format"; required=true)
        @assert format == "HDF"
        h5file, path = map(String, split(content(result), ':'))
        h5file = dirname(xdmf.name) * "/" * h5file
        isfile(h5file) || throw("Xdmf: h5 file $h5file not found!")
        return read(xdmf.hdf, path)
    else
        return result
    end
end


function save!(xdmf::Xdmf)
    doc = XMLDocument()
    set_root(doc, xdmf.xml)
    save_file(doc, xmffile(xdmf))
end

function new_dataitem{T,N}(xdmf::Xdmf, path::String, data::Array{T,N}; format="HDF")
    dataitem = new_element("DataItem")
    datatype = replace("$T", "64", "")
    dimensions = join(reverse(size(data)), " ")
    set_attribute(dataitem, "DataType", datatype)
    set_attribute(dataitem, "Dimensions", dimensions)
    set_attribute(dataitem, "Format", format)
    if format == "HDF"
        hdf = basename(h5file(xdmf))
        if exists(xdmf.hdf, path)
            info("Xdmf: $path already existing in h5 file, not overwriting.")
        else
            write(xdmf.hdf, path, data)
        end
        add_text(dataitem, "$hdf:$path")
    elseif format == "XML"
        add_text(dataitem, strip(string(data), ['[', ']']))
    end
    return dataitem
end

