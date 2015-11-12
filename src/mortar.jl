# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar projection integration

abstract MortarEquation <: Equation

"""
Parameters
----------
node_csys
    coordinate system in node, normal + tangent + "binormal"
element_pairs
    m x s matrix of boolean values, indicating elements sharing
    common surface. s is number of slave elements and m is number
    of master elements.
"""
type MortarProblem <: BoundaryProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{MortarEquation}
    element_mapping :: Dict{Element, MortarEquation}
    master_elements :: Vector{Element}  # mortar surface
    node_csys :: Dict{Int, Matrix{Float64}}
    element_pairs :: Matrix{Bool}
end

function MortarProblem(dimension::Int=1, equations=[], master_elements=[])
    element_mapping = Dict(
        Seg2 => MBC2D2,
    )
    MortarProblem("reaction force", dimension, equations, element_mapping, master_elements, Dict(), zeros(0,0))
end

""" Mortar boundary condition element for 2-dimensional problem, 2 node line segment. """
type MBC2D2 <: MortarEquation
    element :: Seg2  # == non-mortar surface element
    integration_points :: Vector{IntegrationPoint}
end
function MBC2D2(element::Seg2)
    integration_points = default_integration_points(element)
    if !haskey(element, "reaction force")
        element["reaction force"] = zeros(1, 2)
    end
    MBC2D2(element, integration_points)
end
Base.size(equation::MBC2D2) = (1, 2)

function find_master_elements(slave_element, problem)
    # find slave element "position" in element pairs matrix
    all_elements = map((equation) -> get_element(equation), problem.equations)
    seid = findfirst(slave_element, all_elements)
    info("slave element id = $seid")
    # find master element "positions" in element pairs matrix
    meids = find(problem.element_pairs[:, seid])
    info("master element ids = $meids")
    # master elements
    master_elements = problem.master_elements[meids]
    return master_elements
end

function calculate_local_assembly!(assembly::LocalAssembly, equation::MortarEquation, unknown_field_name::ASCIIString, time::Number=0.0, problem=nothing)
    # slave element = non-mortar element where integration happens
    # master element = mortar element projected to non-mortar side
    isa(problem, Void) && error("Cannot create projection without problem")
    initialize_local_assembly!(assembly, equation)
    slave_element = get_element(equation)
    basis = get_basis(slave_element)
    detJ = det(basis)
    master_elements = find_master_elements(equation, problem)
    for master_element in master_elements
        for ip in get_integration_points(slave_element)
            mortar_basis = 0 # ...
            assembly.stiffness_matrix += w*basis'*basis
            assembly.force_vector += w*N'*gn
        end
    end
end


