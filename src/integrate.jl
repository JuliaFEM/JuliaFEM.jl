# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Let's drop here all integration schemes and some defaults for different element types maybe parse from txt file ..?

### Gauss quadrature rules for one dimension

function get_integration_points(::Type{Val{1}})
    return [2.0], [0.0]
end

function get_integration_points(::Type{Val{2}})
    return [1.0, 1.0], sqrt(1.0/3.0)*[-1.0, 1.0]
end

function get_integration_points(::Type{Val{3}})
    return 1.0/9.0*[5.0, 8.0, 5.0], sqrt(3.0/5.0)*[-1.0, 0.0, 1.0]
end

function get_integration_points(::Type{Val{4}})
    weights = 1.0/36.0*[
        18.0-sqrt(30.0),
        18.0+sqrt(30.0),
        18.0+sqrt(30.0),
        18.0-sqrt(30.0)]
    points = [
        -sqrt(3.0/7.0 + 2.0/7.0*sqrt(6.0/5.0)),
        -sqrt(3.0/7.0 - 2.0/7.0*sqrt(6.0/5.0)),
         sqrt(3.0/7.0 - 2.0/7.0*sqrt(6.0/5.0)),
         sqrt(3.0/7.0 + 2.0/7.0*sqrt(6.0/5.0))]
    return weights, points
end

function get_integration_points(::Type{Val{5}})
    weights = [
        1.0/900.0*(322.0-13.0*sqrt(70.0)),
        1.0/900.0*(322.0+13.0*sqrt(70.0)),
        128.0/225.0,
        1.0/900.0*(322.0+13.0*sqrt(70.0)),
        1.0/900.0*(322.0-13.0*sqrt(70.0))]
    points = [
        -1.0/3.0*sqrt(5.0 + 2.0*sqrt(10.0/7.0)),
        -1.0/3.0*sqrt(5.0 - 2.0*sqrt(10.0/7.0)),
         0.0,
         1.0/3.0*sqrt(5.0 - 2.0*sqrt(10.0/7.0)),
         1.0/3.0*sqrt(5.0 + 2.0*sqrt(10.0/7.0))]
    return weights, points
end

function get_integration_points(order::Int64)
    if order <= 5
        return get_integration_points(Val{order})
    else
        points, weights = Base.QuadGK.gauss(Float64, order)
        return weights, points
    end
end

### "cartesian" elements, integration rules comes from tensor product

### 0d elements

function get_integration_points(element::Poi1)
    [ (1.0, [] ) ]
end

### 1d elements

typealias CartesianLineElement Union{Seg2, Seg3, NSeg}
typealias CartesianSurfaceElement Union{Quad4, Quad8, Quad9, NSurf}
typealias CartesianVolumeElement Union{Hex8, Hex20, Hex27, NSolid}

function get_integration_points(element::CartesianLineElement, order::Int64)
    w, xi = get_integration_points(order)
    [ (w[i], [xi[i]]) for i=1:order ]
end

function get_integration_points(element::CartesianSurfaceElement, order::Int64)
    w, xi = get_integration_points(order)
    [ (w[i]*w[j], [xi[i], xi[j]]) for i=1:order, j=1:order ]
end

function get_integration_points(element::CartesianVolumeElement, order::Int64)
    w, xi = get_integration_points(order)
    [ (w[i]*w[j]*w[k], [xi[i], xi[j], xi[k]]) for i=1:order, j=1:order, k=1:order ]
end

### triangular and tetrahedral elements

# http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
# http://libmesh.github.io/doxygen/quadrature__gauss__2D_8C_source.html

typealias TriangularElement Union{Tri3, Tri6}

function get_integration_points(element::TriangularElement, ::Type{Val{1}})
    weights = [0.5]
    points = Vector{Float64}[1.0/3.0*[1.0, 1.0]]
    return zip(weights, points)
end

function get_integration_points(element::TriangularElement, ::Type{Val{2}})
    weights = 1.0/6.0*[1.0, 1.0, 1.0]
    points = Vector{Float64}[
        [2.0/3.0, 1.0/6.0],
        [1.0/6.0, 2.0/3.0],
        [1.0/6.0, 1.0/6.0]]
    return zip(weights, points)
end

function get_integration_points(element::TriangularElement, ::Type{Val{3}})
    weights = 0.5*[
        -0.5625,
         0.5208333333333333,
         0.5208333333333333,
         0.5208333333333333]
    points = Vector{Float64}[
        [1.0/3.0, 1.0/3.0],
        [0.2, 0.2],
        [0.2, 0.6],
        [0.6, 0.2]]
    return zip(weights, points)
end

function get_integration_points(element::TriangularElement, ::Type{Val{4}})
    weights = 0.5*[
        0.22338158967801,
        0.22338158967801,
        0.22338158967801,
        0.10995174365532,
        0.10995174365532,
        0.10995174365532]
    points = Vector{Float64}[
        [0.44594849091597, 0.44594849091597],        
        [0.44594849091597, 0.10810301816807],
        [0.10810301816807, 0.44594849091597],
        [0.09157621350977, 0.09157621350977],
        [0.09157621350977, 0.81684757298046],
        [0.81684757298046, 0.09157621350977]]
    return zip(weights, points)
end

function get_integration_points(element::TriangularElement, ::Type{Val{5}})
    weights = 0.5*[
        0.22500000000000,
        0.13239415278851,
        0.13239415278851,
        0.13239415278851,
        0.12593918054483,
        0.12593918054483,
        0.12593918054483]
    points = Vector{Float64}[
        [0.33333333333333, 0.33333333333333], 
        [0.47014206410511, 0.47014206410511], 
        [0.47014206410511, 0.05971587178977],
        [0.05971587178977, 0.47014206410511],
        [0.10128650732346, 0.10128650732346],
        [0.10128650732346, 0.79742698535309],
        [0.79742698535309, 0.10128650732346]]
    return zip(weights, points)
end

### 3d elements

typealias TetrahedralElement Union{Tet4, Tet10}

function get_integration_points(element::TetrahedralElement, ::Type{Val{1}})
    weights = 1.0/6.0*[1.0]
    points = Vector{Float64}[
        1.0/4.0*[1.0, 1.0, 1.0]]
    return zip(weights, points)
end

function get_integration_points(element::TetrahedralElement, ::Type{Val{2}})
    a = (5.0+3.0*sqrt(5.0))/20.0
    b = (5.0-sqrt(5.0))/20.0
    w = 1.0/24.0
    weights = [w, w, w, w]
    points = Vector{Float64}[
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [b, b, b]]
    return zip(weights, points)
end

function get_integration_points(element::TetrahedralElement, ::Type{Val{3}})
    a = 1.0/4.0
    b = 1.0/6.0
    c = 1.0/2.0
    weights = [-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0]
    points = Vector{Float64}[
        [a, a, a],
        [b, b, b],
        [b, b, c],
        [b, c, b],
        [c, b, b]]
    return zip(weights, points)
end

function get_integration_points(element::TetrahedralElement, ::Type{Val{4}})
    a = 0.25
    b1 = 1.0/34.0*(7.0 + sqrt(15.0))
    b2 = 1.0/34.0*(7.0 - sqrt(15.0))
    c1 = 1.0/34.0*(13.0 - 3.0*sqrt(15.0))
    c2 = 1.0/34.0*(13.0 + 3.0*sqrt(15.0))
    d = 1.0/20.0*(5.0 - sqrt(15.0))
    e = 1.0/20.0*(5.0 + sqrt(15.0))
    w1 = 8.0/405.0
    w2 = (2665.0 - 14.0*sqrt(15.0))/226800.0
    w3 = (2665.0 + 14.0*sqrt(15.0))/226800.0
    w4 = 5.0/567.0
    weights = [w1, w2, w2, w2, w2, w3, w3, w3, w3, w4, w4, w4, w4, w4]
    points = Vector{Float64}[
        [a, a, a],
        [b1, b1, b1],
        [b1, b1, c1],
        [b1, c1, b1],
        [c1, b1, b1],
        [b2, b2, b2],
        [b2, b2, c2],
        [b2, c2, b2],
        [c2, b2, b2],
        [d, d, e],
        [d, e, d],
        [e, d, d],
        [d, e, e],
        [e, d, e],
        [e, e, d]]
    return zip(weights, points)
end

function get_integration_points(element::Union{TriangularElement, TetrahedralElement}, order::Int64)
    return get_integration_points(element, Val{order})
end

### default number of integration points for each element
### 2 for linear elements, 3 for quadratic

typealias LinearElement Union{Seg2, Tri3, Quad4, Tet4, Hex8}

typealias QuadraticElement Union{Seg3, Tri6, Tet10, Quad8, Quad9, Hex20, Hex27}

function get_integration_order(element::LinearElement)
    return 2
end

function get_integration_order(element::QuadraticElement)
    return 3
end

function get_integration_points(element::LinearElement)
    order = get_integration_order(element)
    get_integration_points(element, order)
end

function get_integration_points(element::QuadraticElement)
    order = get_integration_order(element)
    get_integration_points(element, order)
end

