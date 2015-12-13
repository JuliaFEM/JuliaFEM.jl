# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

abstract HeatProblem <: AbstractProblem

function HeatProblem(dim::Int=1, elements=[])
    return Problem{HeatProblem}(dim, elements)
end

function get_unknown_field_name{P<:HeatProblem}(::Type{P})
    return "temperature"
end

function get_unknown_field_type{P<:HeatProblem}(::Type{P})
    # scalar field
    return Float64
end

""" Heat equations.

Formulation
-----------

Field equation is:

    ρc∂u/∂t = ∇⋅(k∇u) + f

Weak form is: find u∈U such that ∀v in V

    ∫k∇u∇v dx = ∫fv dx + ∫gv ds,

where

    k = temperature thermal conductivity    defined on volume
    f = temperature load                    defined on volume
    g = temperature flux                    defined on boundary

References
----------
https://en.wikipedia.org/wiki/Heat_equation

"""
function assemble!(assembly::Assembly, problem::Problem{HeatProblem}, element::Element, time::Number)

    gdofs = get_gdofs(element, problem.dim)
    for ip in get_integration_points(element)
        w = ip.weight
        J = get_jacobian(element, ip, time)
        N = element(ip, time)
        if haskey(element, "density")
            rho = element("density", ip, time)
            add!(assembly.mass_matrix, gdofs, gdofs, w*rho*N'*N*det(J))
        end
        if haskey(element, "temperature thermal conductivity")
            dN = element(ip, time, Val{:grad})
            k = element("temperature thermal conductivity", ip, time)
            add!(assembly.stiffness_matrix, gdofs, gdofs, w*k*dN'*dN*det(J))
        end
        if haskey(element, "temperature load")
            f = element("temperature load", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*f*det(J))
        end
        if haskey(element, "temperature flux")
            g = element("temperature flux", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*g*norm(J))
        end
    end
end

