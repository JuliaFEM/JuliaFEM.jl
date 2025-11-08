# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE
#
# Gaussian-Legendre quadrature rules consolidated from FEMQuad.jl

# Quadrature data
include("quadrature/quaddata.jl")

# Element-specific quadrature rules
include("quadrature/glquad.jl")  # 2D quadrilaterals
include("quadrature/gltri.jl")   # 2D triangles
include("quadrature/gltet.jl")   # 3D tetrahedrons
include("quadrature/glwed.jl")   # 3D wedges
include("quadrature/glpyr.jl")   # 3D pyramids

"""
    get_rule(order::Int, rules::Symbol...)

Get the first quadrature rule that meets the required order.
"""
function get_rule(order::Int, rules::Vararg{Symbol})
    for rule in rules
        if get_order(Val{rule}) >= order
            return rule
        end
    end
    @warn("No accurate rule enough found, picking last.", order, rules)
    return rules[end]
end

"""
    integrate_1d(f::Function, rule::Symbol)

Integrate a 1D function using the specified quadrature rule.
"""
function integrate_1d(f::Function, rule::Symbol)
    points = get_quadrature_points(Val{rule})
    result = sum(w * f(ip) for (w, ip) in points)
    return result
end

"""
    integrate_2d(f::Function, rule::Symbol)

Integrate a 2D function using the specified quadrature rule.
"""
function integrate_2d(f::Function, rule::Symbol)
    points = get_quadrature_points(Val{rule})
    result = sum(w * f(ip) for (w, ip) in points)
    return result
end

"""
    integrate_3d(f::Function, rule::Symbol)

Integrate a 3D function using the specified quadrature rule.
"""
function integrate_3d(f::Function, rule::Symbol)
    points = get_quadrature_points(Val{rule})
    result = sum(w * f(ip) for (w, ip) in points)
    return result
end
