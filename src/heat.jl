# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

type Heat <: FieldProblem
end

function get_unknown_field_name(problem::Problem{Heat})
    return "temperature"
end

function get_unknown_field_type(problem::Problem{Heat})
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
function assemble!(assembly::Assembly, problem::Problem{Heat}, element::Element, time=0.0)

    gdofs = get_gdofs(problem, element)

    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ

        N = element(ip, time)
        if haskey(element, "density")
            rho = element("density", ip, time)
            add!(assembly.M, gdofs, gdofs, w*rho*N'*N)
        end
        if haskey(element, "temperature thermal conductivity")
            dN = element(ip, time, Val{:Grad})
            k = element("temperature thermal conductivity", ip, time)
            add!(assembly.K, gdofs, gdofs, w*k*dN'*dN)
        end
        if haskey(element, "temperature load")
            f = element("temperature load", ip, time)
            add!(assembly.f, gdofs, w*N'*f)
        end
        if haskey(element, "temperature flux")
            g = element("temperature flux", ip, time)
            add!(assembly.f, gdofs, w*N'*g)
        end
    end
end

