# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/AbaqusReader.jl/blob/master/LICENSE

"""
    element_mapping

This mapping table contains information what node ids locally match
each side of element.
"""
const element_mapping = Dict(
    :Tet4 => Dict(
        :S1 => (:Tri3, [1, 3, 2]),
        :S2 => (:Tri3, [1, 2, 4]),
        :S3 => (:Tri3, [2, 3, 4]),
        :S4 => (:Tri3, [1, 4, 3])),
    :Tet10 => Dict(
        :S1 => (:Tri6, [1, 3, 2, 7, 6, 5]),
        :S2 => (:Tri6, [1, 2, 4, 5, 9, 8]),
        :S3 => (:Tri6, [2, 3, 4, 6, 10, 9]),
        :S4 => (:Tri6, [1, 4, 3, 8, 10, 7])),
    :Hex8 => Dict(
        :S1 => (:Quad4, [1, 2, 3, 4]),
        :S2 => (:Quad4, [5, 8, 7, 6]),
        :S3 => (:Quad4, [1, 5, 6, 2]),
        :S4 => (:Quad4, [2, 6, 7, 3]),
        :S5 => (:Quad4, [3, 7, 8, 4]),
        :S6 => (:Quad4, [4, 8, 5, 1])))

""" Given element code, element side and global connectivity, determine boundary
element. E.g. for Tet4 we have 4 sides S1..S4 and boundary element is of type Tri3.
"""
function create_surface_element(element_type::Symbol, element_side::Symbol,
    element_connectivity::Vector{Int})

    if !haskey(element_mapping, element_type)
        error("Unable to find surface element for element of type ",
              "$element_type for side $element_side, update element ",
              "mapping table.")
    end

    if !haskey(element_mapping[element_type], element_side)
        error("Unable to find child element side mapping for element ",
              "of type $element_type for side $element_side, update ",
              "element mapping table.")
    end

    surfel, surfel_lconn = element_mapping[element_type][element_side]
    surfel_gconn = element_connectivity[surfel_lconn]
    return surfel, surfel_gconn
end

"""
    create_surface_elements(mesh, surface_name)

Create surface elements for `surface` using mesh `mesh`.
Mesh can be obtained by using `abaqus_read_mesh`.
"""
function create_surface_elements(mesh::Dict, surface_name::String)
    surface = mesh["surface_sets"][surface_name]
    elements = mesh["elements"]
    eltypes = mesh["element_types"]
    result = Tuple{Symbol, Vector{Int}}[]
    for (elid, side) in surface
        surface_element = create_surface_element(eltypes[elid], side, elements[elid])
        push!(result, surface_element)
    end
    return result
end
