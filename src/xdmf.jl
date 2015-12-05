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

global eltypes = Dict{Symbol, Int}(
    :Tet4  => 0x6,
    :Quad4 => 0x5,
    :Tet10 => 0x0026)

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
#   geometry = new_child(temporal_collection, "Geometry")
#   set_attribute(geometry, "Type", "None")
#   topology = new_child(temporal_collection, "Topology")
#   set_attribute(topology, "Dimensions", "0")
#   set_attribute(topology, "Type", "NoTopology")
    return temporal_collection
end

function xdmf_new_grid(temporal_collection; time=0)
    grid = new_child(temporal_collection, "Grid")
    set_attribute(grid, "Name", "Grid")
    time_ = new_child(grid, "Time")
    set_attribute(time_, "Value", time)
    return grid
end

function xdmf_new_mesh!(grid, nodes, elements)

    # 1. write nodes
    geometry = new_child(grid, "Geometry")
    set_attribute(geometry, "Type", "XYZ")
    dataitem = new_child(geometry, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    ndim = sum([length(node) for node in nodes])
    info("XDFM: ndim = $ndim")
    set_attribute(dataitem, "Dimensions", "$ndim")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 8)
    s = join([join(node, " ") for node in round(nodes, 5)], "\n")
    add_text(dataitem, "\n"*s*"\n")

    # 2. write elements
    topology = new_child(grid, "Topology")
    eldim = sum([length(element[2]) for element in elements]) + length(elements)
    set_attribute(topology, "TopologyType", "Mixed")
    set_attribute(topology, "NumberOfElements", length(elements))
    dataitem = new_child(topology, "DataItem")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "DataType", "Int")
    set_attribute(dataitem, "Dimensions", "$eldim")
#   set_attribute(dataitem, "Precision", 8)
    # note: id numbers start from 0 in Xdmf
    s = join([join([eltypes[eltype]; connectivity-1], " ") for (eltype, connectivity) in elements], "\n")
    add_text(dataitem, "\n"*s*"\n")

end

""" Write Vector field to nodes. """
function xdmf_new_nodal_field!(grid, name, data)
    attribute = new_child(grid, "Attribute")
    set_attribute(attribute, "Center", "Node")
    set_attribute(attribute, "Name", name)
    set_attribute(attribute, "Type", "Vector")
    dataitem = new_child(attribute, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    ndim = sum([length(d) for d in data])
    set_attribute(dataitem, "Dimensions", "$ndim")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "Precision", 8)
    s = join([join(d, " ") for d in round(data, 5)], "\n")
    add_text(dataitem, "\n"*s*"\n")
end

function xdmf_save_model(xdoc, filename)
  save_file(xdoc, filename)
end

