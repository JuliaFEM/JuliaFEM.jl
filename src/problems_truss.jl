# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# This is copied from the files problem_elasticity.jl
"""
This truss fromulation is from
"""
type Truss <: FieldProblem
    # these are found from problem.properties for type Problem{Elasticity}
    formulation :: Symbol
    # Maybe we need more here
end
function Truss()
    # formulations: plane_stress, plane_strain, continuum
    return Truss(:plane_stress)
end

function get_unknown_field_name(problem::Problem{Truss})
    return "displacement"
end

function get_formulation_type(problem::Problem{Truss})
    return :incremental
end

"""
    assemble!(assembly:Assembly, problem::Problem{Elasticity}, elements, time)

Start finite element assembly procedure for Elasticity problem.

Function groups elements to arrays by their type and assembles one element type
at time. This makes it possible to pre-allocate matrices common to same type
of elements.
"""
import JuliaFEM:assemble!


function assemble!(assembly::Assembly, problem::Problem{Truss},
                   element::Element, time=0.0)
    formulation = Val{problem.properties.formulation}
    assemble!(assembly, problem, element, time, formulation)
end

function assemble!(assembly::Assembly, problem::Problem{Truss},
                   element::Element, time, ::Type{Val{:continuum}})
    error("3d elements not implemented in this example")
end

function assemble!(assembly::Assembly, problem::Problem{Truss},
                   element::Element{Seg2}, time, ::Type{Val{:plane_stress}})
    #Require that the number of nodes = 2 ?
    for ip in get_integration_points(element)
        dN = element(ip, time, Val{:Grad})
        detJ = element(ip, time, Val{:detJ})
        A = element("cross section area", ip, time)
        E = element("youngs modulus", ip, time)
        K += ip.weight*E*A*dN'*dN*detJ
    end
    # How do we transform to the global system for 2d/3D
    gdofs = get_gdofs(problem, element)
    add!(assembly.K, gdofs, gdofs, K)
    #add!(assembly.f, gdofs, f)
end
