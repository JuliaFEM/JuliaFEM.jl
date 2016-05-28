# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using LightXML
using JuliaFEM

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

get_xdmf_element_code(element::Element{Tri3})  = 0x0004
get_xdmf_element_code(element::Element{Quad4}) = 0x0005
get_xdmf_element_code(element::Element{Tet4})  = 0x0006
get_xdmf_element_code(element::Element{Hex8})  = 0x0009
get_xdmf_element_code(element::Element{Tet10}) = 0x0026

type XDMF
    dimension :: Int
    use_hdf :: Bool
    xdoc :: XMLDocument
    domain :: XMLElement
    temporal_collection :: XMLElement
    current_grid
    permutation :: Vector{Int}
end

function XDMF()
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "xmlns:xi", "http://www.w3.org/2001/XInclude")
    set_attribute(xroot, "Version", "2.1")
    domain = new_child(xroot, "Domain")
    temporal_collection = new_child(domain, "Grid")
    set_attribute(temporal_collection, "CollectionType", "Temporal")
    set_attribute(temporal_collection, "GridType", "Collection")
    set_attribute(temporal_collection, "Name", "Collection")
    return XDMF(3, false, xdoc, domain, temporal_collection, Union{}, [])
end

function xdmf_new_result!(xdmf::XDMF, elements, time)
    grid = new_child(xdmf.temporal_collection, "Grid")
    set_attribute(grid, "Name", "Grid")
    time_ = new_child(grid, "Time")
    set_attribute(time_, "Value", time)
    xdmf.current_grid = grid

    # 1. calculate permutation
    nids = Set()
    X = Dict{Int64, Vector{Float64}}()
    for element in elements
        conn = get_connectivity(element)
        push!(nids, conn...)
        X_el = element["geometry"](time)
        for (i, c) in enumerate(conn)
            X[c] = X_el[i]
        end
    end
    xdmf.permutation = sort(collect(nids))
    iperm = Dict{Int64, Int64}()
    for (i, j) in enumerate(xdmf.permutation)
        iperm[j] = i
    end

    # 2. write nodes
    geometry = new_child(grid, "Geometry")
    set_attribute(geometry, "Type", xdmf.dimension == 3 ? "XYZ" : "XY")
    dataitem = new_child(geometry, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    set_attribute(dataitem, "Format", "XML")
    #set_attribute(dataitem, "Precision", 8)
    s = ASCIIString[]
    ndim = 0
    for i in xdmf.permutation
        ndim += length(X[i])
        push!(s, join(round(X[i], 5), " "))
    end
    set_attribute(dataitem, "Dimensions", ndim)
    add_text(dataitem, "\n"*join(s, "\n")*"\n")

    # 3. write elements
    topology = new_child(grid, "Topology")
    set_attribute(topology, "TopologyType", "Mixed")
    set_attribute(topology, "NumberOfElements", length(elements))
    dataitem = new_child(topology, "DataItem")
    set_attribute(dataitem, "Format", "XML")
    set_attribute(dataitem, "DataType", "Int")
#   set_attribute(dataitem, "Precision", 8)
    s = ASCIIString[]
    eldim = 0
    for element in elements
        eltype = get_xdmf_element_code(element)
        # note: id numbers start from 0 in Xdmf
        conn = [iperm[j] for j in get_connectivity(element)] - 1
        data = [eltype; conn]
        eldim += length(data)
        push!(s, join(data, " "))
    end
    set_attribute(dataitem, "Dimensions", eldim)
    add_text(dataitem, "\n"*join(s, "\n")*"\n")
end

function xdmf_save_field!(xdmf, elements, time, field_name; field_type="Scalar")
    f = Dict()
    for element in elements
        g = element[field_name](time)
        conn = get_connectivity(element)
        for (i, c) in enumerate(conn)
            f[c] = g[i]
        end
    end

    attribute = new_child(xdmf.current_grid, "Attribute")
    set_attribute(attribute, "Center", "Node")
    set_attribute(attribute, "Name", ucfirst(field_name))
    set_attribute(attribute, "Type", field_type)
    dataitem = new_child(attribute, "DataItem")
    set_attribute(dataitem, "DataType", "Float")
    set_attribute(dataitem, "Format", "XML")
    #set_attribute(dataitem, "Precision", 8)
    s = ASCIIString[]
    dim = 0
    for i in xdmf.permutation
        push!(s, join(round(f[i], 5), " "))
        dim += length(f[i])
    end
    set_attribute(dataitem, "Dimensions", dim)
    add_text(dataitem, "\n"*join(s, "\n")*"\n")
end

function xdmf_save!(xdmf, filename)
    save_file(xdmf.xdoc, filename)
end

