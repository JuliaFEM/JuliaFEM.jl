# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Heat problems

""" Heat equations.

Formulation
-----------

Field equation is:

    ρc∂u/∂t = ∇⋅(k∇u) + f

Weak form is: find u∈U such that ∀v in V

    ∫k∇u∇v dx = ∫fv dx + ∫gv ds,

where

    k = temperature thermal conductivity    defined on volume elements
    f = temperature load                    defined on volume elements
    g = temperature flux                    defined on boundary elements

References
----------
https://en.wikipedia.org/wiki/Heat_equation

"""
type Heat <: FieldProblem
end

function get_unknown_field_name(problem::Problem{Heat})
    return "temperature"
end

function get_unknown_field_type(problem::Problem{Heat})
    return Float64
end

function assemble!(assembly::Assembly, problem::Problem{Heat}, element::Element, time=0.0)
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    K = zeros(nnodes, nnodes)
    fq = zeros(nnodes)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "$field_name thermal conductivity")
            dN = element(ip, time, Val{:Grad})
            k = element("$field_name thermal conductivity", ip, time)
            K += w*k*dN'*dN
        end
        if haskey(element, "$field_name load")
            f = element("$field_name load", ip, time)
            fq += w*N'*f
        end
        if haskey(element, "$field_name flux")
            g = element("$field_name flux", ip, time)
            fq += w*N'*g
        end
    end
    T = vec(element[field_name](time))
    fq -= K*T
    add!(assembly.K, gdofs, gdofs, K)
    add!(assembly.f, gdofs, fq)
end

