# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE
#
# Gaussian-Legendre quadrature rules consolidated from FEMQuad.jl

# Core API types and abstract interfaces
include("quadrature/api.jl")

# Quadrature data
include("quadrature/quaddata.jl")

# Gauss-Legendre quadrature rules by element topology
include("quadrature/gl_tensor_product.jl")  # Segments, quadrilaterals, hexahedra (tensor products)
include("quadrature/gl_triangles.jl")       # 2D triangular elements
include("quadrature/gl_tetrahedra.jl")      # 3D tetrahedral elements
include("quadrature/gl_wedges.jl")          # 3D wedge/prism elements
include("quadrature/gl_pyramids.jl")        # 3D pyramid elements

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
