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

function get_integration_points(element::Poi1)
    [ (1.0, [] ) ]
end

const CartesianLineElement = Union{Seg2,Seg3,NSeg}
const CartesianSurfaceElement = Union{Quad4,Quad8,Quad9,NSurf}
const CartesianVolumeElement = Union{Hex8,Hex20,Hex27,NSolid}

function get_integration_points(element::CartesianLineElement, order::Int64)
    w, xi = get_integration_points(order)
    [ (w[i], [xi[i]]) for i=1:order ]
end

function get_integration_points(element::CartesianSurfaceElement, order::Int64)
    w, xi = get_integration_points(order)
    vec([(w[i]*w[j], [xi[i], xi[j]]) for i=1:order, j=1:order])
end

function get_integration_points(element::CartesianVolumeElement, order::Int64)
    w, xi = get_integration_points(order)
    vec([(w[i]*w[j]*w[k], [xi[i], xi[j], xi[k]]) for i=1:order, j=1:order, k=1:order])
end

### triangular and tetrahedral elements

# http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
# http://libmesh.github.io/doxygen/quadrature__gauss__2D_8C_source.html

const TriangularElement = Union{Tri3,Tri6,Tri7}

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

#=
""" Note, this rule is having negative weight. """
function get_integration_points(element::TriangularElement, ::Type{Val{3}})
    weights = [-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0]
    points = Vector{Float64}[
        [1.0/3.0, 1.0/3.0],
        [1.0/5.0, 1.0/5.0],
        [1.0/5.0, 3.0/5.0],
        [3.0/5.0, 1.0/5.0]]
    return zip(weights, points)
end
=#

function get_integration_points(element::TriangularElement, ::Type{Val{3}})
    weights = [
        1.5902069087198858469718450103758e-01,
        9.0979309128011415302815498962418e-02,
        1.5902069087198858469718450103758e-01,
        9.0979309128011415302815498962418e-02]
    points = Vector{Float64}[
        [1.5505102572168219018027159252941e-01, 1.7855872826361642311703513337422e-01],
        [6.4494897427831780981972840747059e-01, 7.5031110222608118177475598324603e-02],
        [1.5505102572168219018027159252941e-01, 6.6639024601470138670269327409637e-01],
        [6.4494897427831780981972840747059e-01, 2.8001991549907407200279599420481e-01]]
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

""" 7 point integration rule for triangular elements.

References
----------
Code Aster documentation, http://code-aster.org/doc/default/fr/man_r/r3/r3.01.01.pdf
"""
function get_integration_points(element::TriangularElement, ::Type{Val{5}})
    A = 0.470142064105115
    B = 0.101286507323456
    P1 = 0.066197076394253
    P2 = 0.062969590272413
    weights = [9/80, P1, P1, P1, P2, P2, P2]
    points = Vector{Float64}[
        [1/3, 1/3], 
        [A, A],
        [1-2A, A],
        [A, 1-2A],
        [B, B],
        [1-2B, B],
        [B, 1-2B]]
    return zip(weights, points)
end

""" 12 point integration fule for triangular elements.
References
----------
Code Aster documentation, http://code-aster.org/doc/default/fr/man_r/r3/r3.01.01.pdf
"""
function get_integration_points{E<:TriangularElement}(element::Element{E}, ::Type{Val{:FPG12}})
    A = 0.063089014491502
    B = 0.249286745170910
    C = 0.310352451033785
    D = 0.053145049844816
    P1 = 0.025422453185103
    P2 = 0.058393137863189
    P3 = 0.041425537809187
    weights = [P1, P1, P1, P2, P2, P2, P3, P3, P3, P3, P3, P3]
    points = Vector{Float64}[
        [A, A],
        [1-2A, A],
        [A, 1-2A],
        [B, B],
        [1-2B, B],
        [B, 1-2B],
        [C, D],
        [D, C],
        [1-C-D, C],
        [1-C,D, D],
        [C, 1-C-D],
        [D, 1-C-D]]
    return zip(weights, points)
end

### 3d elements

const TetrahedralElement = Union{Tet4,Tet10}

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
    weights = [w1, w2, w2, w2, w2, w3, w3, w3, w3, w4, w4, w4, w4, w4, w4]
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

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch12.d/AFEM.Ch12.pdf

const PyramidalElement = Union{Pyr5,}

function get_integration_points(element::PyramidalElement, ::Type{Val{2}})
    g1 =  0.5842373946721771876874344
    g2 = -2.0/3.0
    g3 =  2.0/5.0
    w1 =  81.0/100.0
    w2 =  125.0/27.0
    weights = [w1, w1, w1, w1, w2]
    points = Vector{Float64}[
        [-g1, -g1, g2],
        [ g1, -g1, g2],
        [ g1,  g1, g2],
        [-g1,  g1, g2],
        [0.0, 0.0, g3],
        ]
    return zip(weights, points)
end

const PrismaticElement = Union{Wedge6,Wedge15}

function get_integration_points(element::PrismaticElement, ::Type{Val{2}})
    weights = 1/6*[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    points = Vector{Float64}[
        [0.5, 0.0, -1.0/sqrt(3)],
        [0.0, 0.5, -1.0/sqrt(3)],
        [0.5, 0.5, -1.0/sqrt(3)],
        [0.5, 0.0,  1.0/sqrt(3)],
        [0.0, 0.5,  1.0/sqrt(3)],
        [0.5, 0.5,  1.0/sqrt(3)]]
    return zip(weights, points)
end

# tensor product of triangular element + segment element
#=
function get_integration_points(element::PrismaticElement, ::Type{Val{2}})
    weights = 1.0/6.0*[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    points = Vector{Float64}[
        [2.0/3.0, 1.0/6.0, -1.0/sqrt(3.0)],
        [1.0/6.0, 2.0/3.0, -1.0/sqrt(3.0)],
        [1.0/6.0, 1.0/6.0, -1.0/sqrt(3.0)],
        [2.0/3.0, 1.0/6.0, +1.0/sqrt(3.0)],
        [1.0/6.0, 2.0/3.0, +1.0/sqrt(3.0)],
        [1.0/6.0, 1.0/6.0, +1.0/sqrt(3.0)],
        ]
    return zip(weights, points)
end
=#

# tensor product of triangular element + segment element
#=
function get_integration_points(element::PrismaticElement, ::Type{Val{3}})
    weights = [
        5.0/9.0*1.5902069087198858469718450103758e-01,
        5.0/9.0*9.0979309128011415302815498962418e-02,
        5.0/9.0*1.5902069087198858469718450103758e-01,
        5.0/9.0*9.0979309128011415302815498962418e-02,
        8.0/9.0*1.5902069087198858469718450103758e-01,
        8.0/9.0*9.0979309128011415302815498962418e-02,
        8.0/9.0*1.5902069087198858469718450103758e-01,
        8.0/9.0*9.0979309128011415302815498962418e-02,
        5.0/9.0*1.5902069087198858469718450103758e-01,
        5.0/9.0*9.0979309128011415302815498962418e-02,
        5.0/9.0*1.5902069087198858469718450103758e-01,
        5.0/9.0*9.0979309128011415302815498962418e-02]
    points = Vector{Float64}[
        [1.5505102572168219018027159252941e-01, 1.7855872826361642311703513337422e-01, -sqrt(3.0/5.0)],
        [6.4494897427831780981972840747059e-01, 7.5031110222608118177475598324603e-02, -sqrt(3.0/5.0)],
        [1.5505102572168219018027159252941e-01, 6.6639024601470138670269327409637e-01, -sqrt(3.0/5.0)],
        [6.4494897427831780981972840747059e-01, 2.8001991549907407200279599420481e-01, -sqrt(3.0/5.0)],
        [1.5505102572168219018027159252941e-01, 1.7855872826361642311703513337422e-01, 0.0],
        [6.4494897427831780981972840747059e-01, 7.5031110222608118177475598324603e-02, 0.0],
        [1.5505102572168219018027159252941e-01, 6.6639024601470138670269327409637e-01, 0.0],
        [6.4494897427831780981972840747059e-01, 2.8001991549907407200279599420481e-01, 0.0],
        [1.5505102572168219018027159252941e-01, 1.7855872826361642311703513337422e-01, sqrt(3.0/5.0)],
        [6.4494897427831780981972840747059e-01, 7.5031110222608118177475598324603e-02, sqrt(3.0/5.0)],
        [1.5505102572168219018027159252941e-01, 6.6639024601470138670269327409637e-01, sqrt(3.0/5.0)],
        [6.4494897427831780981972840747059e-01, 2.8001991549907407200279599420481e-01, sqrt(3.0/5.0)],
        ]
    return zip(weights, points)
end
=#

#=
function get_integration_points(element::PrismaticElement, ::Type{Val{2}})
    weights = 1.0/96.0*[-27.0, 25.0, 25.0, 25.0, -27.0, 25.0, 25.0, 25.0]
    a = 1.0/sqrt(3.0)
    points = Vector{Float64}[
        [1/3, 1/3, -a],
        [0.6, 0.2, -a],
        [0.2, 0.6, -a],
        [0.2, 0.2, -a],
        [1/3, 1/3,  a],
        [0.6, 0.2,  a],
        [0.2, 0.6,  a],
        [0.2, 0.2,  a]]
    return zip(weights, points)
end
=#

function get_integration_points(element::PrismaticElement, ::Type{Val{3}})
    alpha = sqrt(3/5)
    c1 = 5/9
    c2 = 8/9
    a = (6+sqrt(15))/21
    b = (6-sqrt(15))/21
    weights = [
        c1*9/80,
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c2*9/80,
        c2*((155+sqrt(15))/2400),
        c2*((155+sqrt(15))/2400),
        c2*((155+sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c2*((155-sqrt(15))/2400),
        c1*9/80,
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155+sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
        c1*((155-sqrt(15))/2400),
    ]
    points = Vector{Float64}[
        [1/3, 1/3, -alpha],
        [a, a, -alpha],
        [1-2a, a, -alpha],
        [a, 1-2a, -alpha],
        [b, b, -alpha],
        [1-2b, b, -alpha],
        [b, 1-2b, -alpha],
        [1/3, 1/3, 0],
        [a, a, 0],
        [1-2a, a, 0],
        [a, 1-2a, 0],
        [b, b, 0],
        [1-2b, b, 0],
        [b, 1-2b, 0],
        [1/3, 1/3, alpha],
        [a, a, alpha],
        [1-2a, a, alpha],
        [a, 1-2a, alpha],
        [b, b, alpha],
        [1-2b, b, alpha],
        [b, 1-2b, alpha],
        ]
    return zip(weights, points)
end

function get_integration_points(element::Union{TriangularElement,
            TetrahedralElement, PyramidalElement, PrismaticElement}, order::Int64)
    return get_integration_points(element, Val{order})
end

### default number of integration points for each element
### 2 for linear elements, 3 for quadratic

const LinearElement = Union{Seg2, Tri3, Quad4, Tet4, Pyr5, Wedge6, Hex8}

const QuadraticElement = Union{Seg3,Tri6,Tri7,Tet10,Quad8,Quad9,Wedge15,Hex20,Hex27}

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

