# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using LightXML
importall LightXML
importall Base

function new_element(name::String, attrs::Dict)
    x = new_element(name)
    for (k, v) in attrs
        x[k] = v
    end
    return x
end

function setindex!(x::XMLElement, content::Any, attr_name::String)
    set_attribute(x, attr_name, content)
end

function haskey(x::XMLElement, key::String)
    return has_child(x, key) || has_attribute(x, key)
end

function has_child(x::XMLElement, child_name::String)
    return get_child(x, child_name) != nothing
end

function get_attribute(x::XMLElement, attr_name::String)
    attr = attribute(x, attr_name)
    numeric = tryparse(Int64, attr)
    isnull(numeric) && (numeric = tryparse(Float64, attr))
    isnull(numeric) && return attr
    return get(numeric)
end

function new_child(xparent::XMLElement, name::String, attrs::Dict)
    x = new_child(xparent, name)
    for (k, v) in attrs
        x[k] = v
    end
    return x
end

""" Basic traverse support, so that it's possible to find data from xml using
path syntax e.g. /foo/bar[2]/baz[@Name=Frame 1]/DataItem. If several elements
with same name exists in tree, pick first by default and next ones can be picked
using [] syntax or [@attr=value] syntax, see [1] for details. For last item use
[end].

[1] http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""
function get_child(x::XMLElement, child_name::String)
    '/' in child_name && return nothing
    m = match(r"(\w+)\[(.+)\]", child_name)
    if m != nothing
        child_name = m[1]
    end
    childs = []
    for child in child_elements(x)
        if name(child) == child_name
            push!(childs, child)
        end
    end
    length(childs) == 0 && return nothing
    m == nothing && return first(childs)
    j = tryparse(Int, m[2])
    isnull(j) || return childs[get(j)]
    m[2] == "end" && return childs[end]
    m2 = match(r"@(.+)=(.+)", m[2])
    m2 == nothing && throw("Unable to parse: $(m[2])")
    attr_name = m2[1]
    attr_value = m2[2]
    for child in childs
        has_attribute(child, attr_name) || continue
        if get_attribute(child, attr_name) == attr_value
            return child
        end
    end
    throw("Unable to parse: $(m[2])")
end

function getindex(x::XMLElement, attr_name::String)
    attr_name = strip(attr_name, '/')
    child = get_child(x, attr_name)
    child == nothing || return child
    has_attribute(x, attr_name) && return get_attribute(x, attr_name)
    if '/' in attr_name
        items = map(String, split(attr_name, '/'))
        attr_name = first(items)
        length(items) > 1 || throw(KeyError(attr_name))
        haskey(x, attr_name) || throw(KeyError(attr_name))
        path = join(items[2:end], '/')
        new_item = getindex(x, attr_name)
        return new_item[path]
    else
        throw(KeyError(attr_name))
    end
end

type Xdmf
    name :: String
    xml :: XMLElement
    hdf :: HDF5File
end

function new_child(xdmf::Xdmf, args...; kwargs...)
    new_child(xdmf.xml, args...; kwargs...)
end

function read(xdmf::Xdmf, path::String)
    result = getindex(xdmf.xml, path)
    if endswith(path, "DataItem")
        format = get_attribute(result, "Format")
        @assert format == "HDF"
        h5file, path = map(String, split(content(result), ':'))
        h5file = dirname(xdmf.name) * "/" * h5file
        isfile(h5file) || throw("Xdmf: h5 file $h5file not found!")
        return read(xdmf.hdf, path)
    else
        return result
    end
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

