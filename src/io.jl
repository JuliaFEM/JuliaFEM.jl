# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using HDF5
using LightXML

type ModelIO
    name :: AbstractString
    xdmf :: XMLElement
end

function ModelIO()
    return ModelIO(tempname())
end

function ModelIO(name::AbstractString)
    xdmf = new_element("Xdmf")
    set_attribute(xdmf, "xmlns:xi", "http://www.w3.org/2001/XInclude")
    set_attribute(xdmf, "Version", "2.1")
    return ModelIO(name, xdmf)
end

function h5file(mio::ModelIO)
    return mio.name*".h5"
end

function put!{T,N}(mio::ModelIO, path::AbstractString, data::Array{T,N})
    hdf = h5file(mio)
    h5write(hdf, path, data)
end

function get_dataitem{T,N}(mio::ModelIO, path::AbstractString, data::Array{T,N}; format="HDF")
    dataitem = new_element("DataItem")
    n, m = size(data)
    set_attribute(dataitem, "DataType", "$T")
    set_attribute(dataitem, "Dimensions", "$n $m")
    set_attribute(dataitem, "Format", format)
    if format == "HDF"
        hdf = basename(h5file(mio))
        add_text(dataitem, "$hdf:$path")
    end
    return dataitem
end

function get(mio::ModelIO, path::AbstractString)
    h5read(mio.name*".h5", path)
end

function save!(mio::ModelIO)
    doc = XMLDocument()
    set_root(doc, mio.xdmf)
    save_file(doc, mio.name*".xmf")
end
