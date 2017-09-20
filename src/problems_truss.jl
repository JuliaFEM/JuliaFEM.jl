# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# This is copied from the files problem_elasticity.jl
"""
This truss fromulation is from
"""
type Truss <: FieldProblem
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

function assemble!(assembly::Assembly, problem::Problem{Truss},
                   element::Element{Seg2}, time)
    #Require that the number of nodes = 2 ?
    nnodes = length(element)
    ndim = get_unknown_field_dimension(problem)
    ndofs = nnodes*ndim
    K = zeros(ndofs,ndofs)
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
