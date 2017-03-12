# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using LightXML
    
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


""" Write new fields to Xdmf file.

Examples
--------

To write displacement and temperature fields from p1 at time t=0.0:

julia> update_xdmf!(p1, 0.0, ["displacement", "temperature"])
"""
function update_xdmf!(xdmf::Xdmf, problem::Problem, time::Float64, fields::Vector)

    info("Xdmf: storing fields $fields of problem $(problem.name) at time $time")

    # 1. find domain
    xml = xdmf.xml
    domain = find_element(xml, "Domain")
    if domain == nothing
        info("Xdmf: Domain not found, creating.")
        domain = new_child(xml, "Domain")
    else
        debug("Xdmf: Domain already defined, skipping.")
    end

    # 2. find for TemporalCollection
    temporal_collection = find_element(domain, "Grid")
    if temporal_collection == nothing
        info("Xdmf: Temporal collection not found, creating.")
        temporal_collection = new_child(domain, "Grid")
        set_attribute(temporal_collection, "GridType", "Collection")
        set_attribute(temporal_collection, "Name", "Time")
        set_attribute(temporal_collection, "CollectionType", "Temporal")
    else
        debug("Xdmf: Temporal collection found, skipping.")
    end

    # 2.1 make sure that Grid element we found really is TemporalCollection
    collection_type = attribute(temporal_collection, "CollectionType"; required=true)
    @assert collection_type == "Temporal"

    # 3. find for SpatialCollection at given time
    spatial_collection = nothing
    spatial_collection_exists = false
    for spatial_collection in get_elements_by_tagname(temporal_collection, "Grid")
        time_element = find_element(spatial_collection, "Time")
        time_value = parse(attribute(time_element, "Value"; required=true))
        if isapprox(time_value, time)
            info("Xdmf: SpatialCollection for time $time already exists.")
            spatial_collection_exists = true
            break
        end
    end

    if !spatial_collection_exists
        info("Xdmf: SpatialCollection for time $time not found, creating.")
        spatial_collection = new_child(temporal_collection, "Grid")
        set_attribute(spatial_collection, "GridType", "Collection")
        set_attribute(spatial_collection, "Name", "Problems")
        set_attribute(spatial_collection, "CollectionType", "Spatial")
        time_element = new_child(spatial_collection, "Time")
        set_attribute(time_element, "Value", time)
    end
    
    # 3.1 make sure that Grid element we found really is SpatialCollection
    collection_type = attribute(spatial_collection, "CollectionType"; required=true)
    @assert collection_type == "Spatial"

    for frame in get_elements_by_tagname(spatial_collection, "Grid")
        frame_name = attribute(frame, "Name")
        if frame_name == problem.name
            warn("Xdmf: Already found Grid with name $frame_name for time $time, skipping.")
            return
        end
    end

    frame_name = problem.name
    info("Xdmf: Creating Grid for problem $frame_name")
    frame = new_child(spatial_collection, "Grid")
    set_attribute(frame, "Name", frame_name)

    # 4. save geometry
    X_dict = problem("geometry", time)
    node_ids = sort(collect(keys(X_dict)))
    node_mapping = Dict(j => i for (i, j) in enumerate(node_ids))
    X_array = hcat([X_dict[nid] for nid in node_ids]...)
    ndim, nnodes = size(X_array)
    geom_type = (ndim == 2 ? "XY" : "XYZ")
    info("Xdmf: Creating geometry, type = $geom_type, number of nodes = $nnodes")
    X_dataitem = new_dataitem(xdmf, X_array)
    geometry = new_child(frame, "Geometry")
    set_attribute(geometry, "Type", geom_type)
    add_child(geometry, X_dataitem)

    # 5. save topology
    all_elements = get_elements(problem)
    nelements = length(all_elements)
    element_types = unique(map(get_element_type, all_elements))
    nelement_types = length(element_types)
    info("Xdmf: Saving topology of $nelements elements total, $nelement_types different element types.")

    for element_type in element_types
        elements = filter_by_element_type(element_type, all_elements)
        nelements = length(elements)
        info("Xdmf: $nelements elements of type $element_type")
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

    # 6. save requested fields
    for field_name in fields
        field_dict = problem(field_name, time)
        field_center = "Node"
        field_node_ids = sort(collect(keys(field_dict)))
        @assert node_ids == field_node_ids
        field_dim = length(field_dict[first(field_node_ids)])
        field_type = field_dim == 1 ? "Scalar" : "Vector"
        info("Xdmf: Saving field $field_name, type = $field_type, dimension = $field_dim, center = $field_center")
        if field_dim == 2
            info("Xdmf: Field dimension = 2, extending to 3")
            for nid in field_node_ids
                field_dict[nid] = [field_dict[nid]; 0.0]
            end
            field_dim == 3
        end

        field_array = hcat([field_dict[nid] for nid in field_node_ids]...)
        field_dataitem = new_dataitem(xdmf, field_array)
        attribute = new_child(frame, "Attribute")
        set_attribute(attribute, "Name", ucfirst(field_name))
        set_attribute(attribute, "Center", field_center)
        set_attribute(attribute, "AttributeType", field_type)
        add_child(attribute, field_dataitem)
    end

    save!(xdmf)
    info("Xdmf: all done.")
end

