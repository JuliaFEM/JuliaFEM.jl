# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module xdmf

using Logging
@Logging.configure(level=INFO)

using LightXML

VERSION < v"0.4-" && using Docile

# i add docstrings later
# element codes: http://www.paraview.org/pipermail/paraview/2013-July/028859.html

function xdmf_new_model(xdmf_version="2.1")
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "xmlns:xi", "http://www.w3.org/2001/XInclude")
    set_attribute(xroot, "Version", xdmf_version)
    domain = new_child(xroot, "Domain")
    return xdoc, domain
end

function xdmf_new_temporal_collection(model)
    temporal_collection = new_child(model, "Grid")
    set_attribute(temporal_collection, "CollectionType", "Temporal")
    set_attribute(temporal_collection, "GridType", "Collection")
    set_attribute(temporal_collection, "Name", "Collection")
    geometry = new_child(temporal_collection, "Geometry")
    set_attribute(geometry, "Type", "None")
    topology = new_child(temporal_collection, "Topology")
    set_attribute(topology, "Dimensions", "0")
    set_attribute(topology, "Type", "NoTopology")
    return temporal_collection
end

function xdmf_new_grid(temporal_collection; time=0)
    grid = new_child(temporal_collection, "Grid")
    set_attribute(grid, "Name", "Grid")
    time_ = new_child(grid, "Time")
    set_attribute(time_, "Value", time)
    return grid
end

function xdmf_new_mesh(grid, X, elmap)
    geometry = new_child(grid, "Geometry")
    set_attribute(geometry, "Type", "XYZ")
    dataitem = new_child(geometry, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    set_attribute(dataitem, "Dimensions", length(X))
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", "4")
    add_text(dataitem, join(X, " "))

    topology = new_child(grid, "Topology")
    set_attribute(topology, "Dimensions", "1")
    set_attribute(topology, "Type", "Mixed")
    dataitem = new_child(topology, "DataItem")
    set_attribute(dataitem, "DataType", "Int")
    set_attribute(dataitem, "Dimensions", length(elmap))
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 4)
    elmap2 = copy(elmap)
    elmap2[2:end,:] -= 1
    add_text(dataitem, join(elmap2, " "))
end

function xdmf_new_field(grid, name, source, data)
    loc = Dict("elements" => "Cell",
               "nodes" => "Node")

    dim1, dim2 = size(data'')
    if dim1 == 1
        Type = "Scalar"
    end
    if dim1 == 3
        Type = "Vector"
    end

    typ = string(typeof(data))
    datatype = "unknown"
    @debug("typeof: ", typ)
    for j in ["Int", "Float"]
        @debug(j)
        if contains(typ, j)
            datatype = j
        end
    end
    if datatype == "unknown"
        throw("unknown data type ", typ)
    end


    attribute = new_child(grid, "Attribute")
    set_attribute(attribute, "Center", loc[source])
    set_attribute(attribute, "Name", name)
    set_attribute(attribute, "Type", Type)
    dataitem = new_child(attribute, "DataItem")
    set_attribute(dataitem, "DataType", datatype)
    set_attribute(dataitem, "Dimensions", length(data))
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 4)
    add_text(dataitem, join(data, " "))
end


end