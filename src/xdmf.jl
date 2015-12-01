# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using LightXML

# element codes: http://www.paraview.org/pipermail/paraview/2013-July/028859.html
# > from  ./VTK/ThirdParty/xdmf2/vtkxdmf2/libsrc/XdmfTopology.h
# >
# > // Topologies
# > #define XDMF_NOTOPOLOGY     0x0
# > #define XDMF_POLYVERTEX     0x1
# > #define XDMF_POLYLINE       0x2
# > #define XDMF_POLYGON        0x3
# > #define XDMF_TRI            0x4
# > #define XDMF_QUAD           0x5
# > #define XDMF_TET            0x6
# > #define XDMF_PYRAMID        0x7
# > #define XDMF_WEDGE          0x8
# > #define XDMF_HEX            0x9
# > #define XDMF_EDGE_3         0x0022
# > #define XDMF_TRI_6          0x0024
# > #define XDMF_QUAD_8         0x0025
# > #define XDMF_QUAD_9         0x0023
# > #define XDMF_TET_10         0x0026
# > #define XDMF_PYRAMID_13     0x0027
# > #define XDMF_WEDGE_15       0x0028
# > #define XDMF_WEDGE_18       0x0029
# > #define XDMF_HEX_20         0x0030
# > #define XDMF_HEX_24         0x0031
# > #define XDMF_HEX_27         0x0032
# > #define XDMF_MIXED          0x0070
# > #define XDMF_2DSMESH        0x0100
# > #define XDMF_2DRECTMESH     0x0101
# > #define XDMF_2DCORECTMESH   0x0102
# > #define XDMF_3DSMESH        0x1100
# > #define XDMF_3DRECTMESH     0x1101
# > #define XDMF_3DCORECTMESH   0x1102
"""
Build a new model for outout

Parameters
----------

Examples
-------

```julia
@assert 1+1 == 3
```

"""
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

#function xdmf_new_mesh(grid, X, elmap)
#    geometry = new_child(grid, "Geometry")
#    set_attribute(geometry, "Type", "XYZ")
#    dataitem = new_child(geometry, "DataItem")
#    set_attribute(dataitem, "DataType", "Float")
#    set_attribute(dataitem, "Dimensions", length(X))
#    set_attribute(dataitem, "Format", "XML")
#    set_attribute(dataitem, "Precision", "4")
#    add_text(dataitem, join(X, " "))
#    topology = new_child(grid, "Topology")
#    set_attribute(topology, "Dimensions", "1")
#    set_attribute(topology, "Type", "Mixed")
#    dataitem = new_child(topology, "DataItem")
#    set_attribute(dataitem, "DataType", "Int")
#    set_attribute(dataitem, "Dimensions", length(elmap))
#    set_attribute(dataitem, "Format", "XML")
#    set_attribute(dataitem, "Precision", 4)
#    elmap2 = copy(elmap)
#    elmap2[2:end,:] -= 1
#    add_text(dataitem, join(elmap2, " "))
#end

function xdmf_new_mesh(grid, X, elmap)
    dim, nnodes = size(X)
    geometry = new_child(grid, "Geometry")
    set_attribute(geometry, "Type", "XYZ")
    dataitem = new_child(geometry, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    set_attribute(dataitem, "Dimensions", "$nnodes $dim")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 8)
    #add_text(dataitem, join(X, " "))
    s = "\n"
    
    for i=1:nnodes
        s *= "\t\t" * join(X[:,i], " ") * "\n"
    end
    s *= "       "
    add_text(dataitem, s)

    elmap2 = copy(elmap)
    elmap2[2:end,:] -= 1
    dim, nelements = size(elmap2)

    topology = new_child(grid, "Topology")
    #set_attribute(topology, "Dimensions", "1")
    set_attribute(topology, "TopologyType", "Mixed")
    set_attribute(topology, "NumberOfElements", nelements)
    dataitem = new_child(topology, "DataItem")
    set_attribute(dataitem, "DataType", "Int")
    set_attribute(dataitem, "Dimensions", "$nelements $dim")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 8)
    s = "\n"
    for i=1:nelements
        s *= "\t\t" * join(elmap2[:,i], " ") * "\n"
    end
    add_text(dataitem, s)
    #add_text(dataitem, join(elmap2, " "))    
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
    for j in ["Int", "Float"]
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

function xdmf_save_model(xdoc, filename)
  save_file(xdoc, filename)
end

