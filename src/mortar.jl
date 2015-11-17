# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar equations

abstract MortarEquation <: Equation

function get_unknown_field_name(equation::MortarEquation)
    return "reaction force"
end

""" Mortar boundary condition element for 2-dimensional problem, 2 node line segment. """
type MBC2D2 <: MortarEquation
    element :: MSeg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::MBC2D2)
    return (1, 2)
end

function Base.convert(::Type{MortarEquation}, element::MSeg2)
    return MBC2D2(element, get_default_integration_points(element))
end

# Mortar problem

"""
Parameters
----------
node_csys
    coordinate system in node, normal + tangent + "binormal"
    in 3d 3x3 matrix, in 2d 2x2 matrix, respectively
"""
type MortarProblem <: BoundaryProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{MortarEquation}
end

function MortarProblem(dimension::Int=1, equations=[])
    MortarProblem("reaction force", dimension, equations)
end

# Mortar projection calculation

""" Find master or "mortar" elements for this slave element. """
function get_master_elements(element::MortarElement)
    return element.master_elements
end

function assemble!(assembly::Assembly, equation::MortarEquation, time::Number=0.0, problem=nothing)
    slave_element = get_element(equation)
    master_elements = get_master_elements(slave_element)
    slave_basis = get_basis(slave_element)
    detJ = det(slave_basis)
    dim = size(equation, 1) # number of nodes
    slave_dofs = get_gdofs(slave_element, dim)
    for master_element in master_elements
        master_dofs = get_gdofs(master_element, dim)
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0])
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0])
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*(xi1[2]-xi1[1])
        if abs(l) < 1.0e-6
            warn("No contribution")
            continue # no contribution
        end
        master_basis = get_basis(master_element)
        for ip in get_integration_points(equation)
            w = ip.weight*detJ(ip)*l

            # integration point on slave side segment
            xi_gauss = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point
            xi_projected = project_from_slave_to_master(slave_element, master_element, xi_gauss)

            # add contribution to left hand side 
            N1 = slave_basis(xi_gauss, time)
            N2 = master_basis(xi_projected, time)
            add!(assembly.lhs, slave_dofs, slave_dofs, w*N1'*N1)
            add!(assembly.lhs, slave_dofs, master_dofs, -w*N1'*N2)

        end
    end
end


