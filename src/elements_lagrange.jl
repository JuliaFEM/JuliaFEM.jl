# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

### 0d element

type Poi1 <: AbstractElement
end

function description(::Type{Poi1})
    "1 node point"
end

function size(element::Element{Poi1})
    return (0, 1)
end

function length(element::Element{Poi1})
    return 1
end

function get_basis(element::Element{Poi1}, ip, time)
    return [1]
end

function get_dbasis(element::Element{Poi1}, ip, time)
    return [0]
end

function (element::Element{Poi1})(ip, time::Float64, ::Type{Val{:detJ}})
    return 1.0
end

function get_integration_order(element::Poi1)
    return 1
end

function get_integration_points(element::Poi1, order::Int64)
    return [ (1.0, [] ) ]
end

function get_reference_coordinates(::Type{Poi1})
    Vector{Float64}[[0.0]]
end

### 1d elements

type Seg2 <: AbstractElement
end

function description(::Type{Seg2})
    "2 node segment"
end

function size(element::Element{Seg2})
    return (1, 2)
end

function length(element::Element{Seg2})
    return 2
end

function get_reference_coordinates(::Type{Seg2})
    Vector{Float64}[
        [-1.0], # N1
        [ 1.0]] # N2
end

function get_interpolation_polynomial(::Type{Seg2}, xi)
    [1.0 xi[1]]
end

function get_interpolation_polynomial(::Type{Seg2}, xi, ::Type{Val{:partial_derivatives}})
    [0.0 1.0]
end

#

type Seg3 <: AbstractElement
end

function description(::Type{Seg3})
    "3 node segment"
end

function size(element::Element{Seg3})
    return (1, 3)
end

function length(element::Element{Seg3})
    return 3
end

function get_reference_coordinates(::Type{Seg3})
    Vector{Float64}[
        [-1.0], # N1
        [ 1.0], # N2
        [ 0.0]] # N3
end

function get_interpolation_polynomial(::Type{Seg3}, xi)
    [1.0 xi[1] xi[1]^2]
end

function get_interpolation_polynomial(::Type{Seg3}, xi, ::Type{Val{:partial_derivatives}})
    [0.0 1.0 2.0*xi[1]]
end

### 2d elements

type Tri3 <: AbstractElement
end

function description(::Type{Tri3})
    "3 node triangle"
end

function size(element::Element{Tri3})
    return (2, 3)
end

function length(element::Element{Tri3})
    return 3
end

function get_reference_coordinates(::Type{Tri3})
    Vector{Float64}[
        [0.0, 0.0], # N1
        [1.0, 0.0], # N2
        [0.0, 1.0]] # N3
end

function get_interpolation_polynomial(::Type{Tri3}, xi)
    [
        1 xi[1] xi[2]
    ]
end

function get_interpolation_polynomial(::Type{Tri3}, xi, ::Type{Val{:partial_derivatives}})
    [
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]
end

#

type Tri6 <: AbstractElement
end

function description(::Type{Tri6})
    "6 node triangle"
end

function size(element::Element{Tri6})
    return (2, 6)
end

function length(element::Element{Tri6})
    return 6
end

function get_reference_coordinates(::Type{Tri6})
    Vector{Float64}[
        [0.0, 0.0], # N1
        [1.0, 0.0], # N2
        [0.0, 1.0], # N3
        [0.5, 0.0], # N4
        [0.5, 0.5], # N5
        [0.0, 0.5]] # N6
end

function get_interpolation_polynomial(::Type{Tri6}, xi)
    [
        1 xi[1] xi[2] xi[1]^2 xi[1]*xi[2] xi[2]^2
    ]
end

function get_interpolation_polynomial(::Type{Tri6}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  1  0  2*xi[1]  xi[2]  0
        0  0  1  0        xi[1]  2*xi[2]
    ]
end

#

type Tri7 <: AbstractElement
end

function description(::Type{Tri7})
    "7 node triangle"
end

function size(element::Element{Tri7})
    return (2, 7)
end

function length(element::Element{Tri7})
    return 7
end

function get_reference_coordinates(::Type{Tri7})
    Vector{Float64}[
        [0.0, 0.0], # N1
        [1.0, 0.0], # N2
        [0.0, 1.0], # N3
        [0.5, 0.0], # N4
        [0.5, 0.5], # N5
        [0.0, 0.5], # N6
        [1/3, 1/3]] # N7
end

function get_interpolation_polynomial(::Type{Tri7}, xi)
    [
        1 xi[1] xi[2] xi[1]^2 xi[1]*xi[2] xi[2]^2 xi[1]^2*xi[2]^2
    ]
end

function get_interpolation_polynomial(::Type{Tri7}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  1  0  2*xi[1]  xi[2]  0        2*xi[1]*xi[2]^2
        0  0  1  0        xi[1]  2*xi[2]  2*xi[1]^2*xi[2]
    ]
end

#

type Quad4 <: AbstractElement
end

function description(::Type{Quad4})
    "4 node quadrangle"
end

function size(element::Element{Quad4})
    return (2, 4)
end

function length(element::Element{Quad4})
    return 4
end

function get_reference_coordinates(::Type{Quad4})
    Vector{Float64}[
        [-1.0, -1.0], # N1
        [ 1.0, -1.0], # N2
        [ 1.0,  1.0], # N3
        [-1.0,  1.0]] # N4
end

function get_interpolation_polynomial(::Type{Quad4}, xi)
    [
        1.0 xi[1] xi[2] xi[1]*xi[2]
    ]
end

function get_interpolation_polynomial(::Type{Quad4}, xi, ::Type{Val{:partial_derivatives}})
    [
        0 1 0 xi[2]
        0 0 1 xi[1]      
    ]
end

#

type Quad8 <: AbstractElement
end

function description(::Type{Quad8})
    "8 node Serendip quadrangle"
end

function size(element::Element{Quad8})
    return (2, 8)
end

function length(element::Element{Quad8})
    return 8
end

function get_reference_coordinates(::Type{Quad8})
    Vector{Float64}[
        [-1.0, -1.0], # N1
        [ 1.0, -1.0], # N2
        [ 1.0,  1.0], # N3
        [-1.0,  1.0], # N4
        [ 0.0, -1.0], # N5
        [ 1.0,  0.0], # N6
        [ 0.0,  1.0], # N7
        [-1.0,  0.0]] # N8
end

function get_interpolation_polynomial(::Type{Quad8}, xi)
    [
        1 xi[2] xi[1] xi[2]^2 xi[1]*xi[2] xi[1]^2 xi[1]*xi[2]^2 xi[1]^2*xi[2]
    ]
end

function get_interpolation_polynomial(::Type{Quad8}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  1  0        xi[2]  2*xi[1]  xi[2]^2        2*xi[1]*xi[2]
        0  1  0  2*xi[2]  xi[1]  0        2*xi[1]*xi[2]  xi[1]^2
    ]
end

#

type Quad9 <: AbstractElement
end

function description(::Type{Quad9})
    "9 node quadrangle"
end

function size(element::Element{Quad9})
    return (2, 9)
end

function length(element::Element{Quad9})
    return 9
end

function get_reference_coordinates(::Type{Quad9})
    Vector{Float64}[
        [-1.0, -1.0], # N1
        [ 1.0, -1.0], # N2
        [ 1.0,  1.0], # N3
        [-1.0,  1.0], # N4
        [ 0.0, -1.0], # N5
        [ 1.0,  0.0], # N6
        [ 0.0,  1.0], # N7
        [-1.0,  0.0], # N8
        [ 0.0,  0.0]] # N9
end

function get_interpolation_polynomial(::Type{Quad9}, xi)
    [
        1 xi[2] xi[1] xi[2]^2 xi[1]*xi[2] xi[1]^2 xi[1]*xi[2]^2 xi[1]^2*xi[2] xi[1]^2*xi[2]^2
    ]
end

function get_interpolation_polynomial(::Type{Quad9}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  1  0        xi[2]  2*xi[1]  xi[2]^2        2*xi[1]*xi[2]  2*xi[1]*xi[2]^2
        0  1  0  2*xi[2]  xi[1]  0        2*xi[1]*xi[2]  xi[1]^2        2*xi[1]^2*xi[2]
    ]
end

### 3d elements

type Tet4 <: AbstractElement
end

function description(::Type{Tet4})
    "4 node tetrahedral element"
end

function size(element::Element{Tet4})
    return (3, 4)
end

function length(element::Element{Tet4})
    return 4
end

function get_reference_coordinates(::Type{Tet4})
    Vector{Float64}[
        [0.0, 0.0, 0.0], # N1
        [1.0, 0.0, 0.0], # N2
        [0.0, 1.0, 0.0], # N3
        [0.0, 0.0, 1.0]] # N4
end

function get_interpolation_polynomial(::Type{Tet4}, xi)
    [
        1.0 xi[1] xi[2] xi[3]
    ]
end

function get_interpolation_polynomial(::Type{Tet4}, xi, ::Type{Val{:partial_derivatives}})
    [
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
end

#

type Tet10 <: AbstractElement
end

function description(::Type{Tet10})
    "10 node tetrahedral element"
end

function size(element::Element{Tet10})
    return (3, 10)
end

function length(element::Element{Tet10})
    return 10
end

function get_reference_coordinates(::Type{Tet10})
    Vector{Float64}[
        [0.0, 0.0, 0.0], # N1
        [1.0, 0.0, 0.0], # N2
        [0.0, 1.0, 0.0], # N3
        [0.0, 0.0, 1.0], # N4
        [0.5, 0.0, 0.0], # N5
        [0.5, 0.5, 0.0], # N6
        [0.0, 0.5, 0.0], # N7
        [0.0, 0.0, 0.5], # N8
        [0.5, 0.0, 0.5], # N9
        [0.0, 0.5, 0.5]] # N10
end

function get_interpolation_polynomial(::Type{Tet10}, xi)
    [
        1.0 xi[3] xi[2] xi[1] xi[3]^2 xi[2]*xi[3] xi[2]^2 xi[1]*xi[3] xi[1]*xi[2] xi[1]^2
    ]
end

function get_interpolation_polynomial(::Type{Tet10}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  0  1  0        0      0        xi[3]  xi[2]  2*xi[1]
        0  0  1  0  0        xi[3]  2*xi[2]  0      xi[1]  0      
        0  1  0  0  2*xi[3]  xi[2]  0        xi[1]  0      0          
    ]
end

#

type Wedge6 <: AbstractElement
end

function description(::Type{Wedge6})
    "6 node prismatic element (wedge)"
end

function size(element::Element{Wedge6})
    return (3, 6)
end

function length(element::Element{Wedge6})
    return 6
end

function get_reference_coordinates(::Type{Wedge6})
    Vector{Float64}[
        [0.0, 0.0, -1.0], # N1
        [1.0, 0.0, -1.0], # N2
        [0.0, 1.0, -1.0], # N3
        [0.0, 0.0,  1.0], # N4
        [1.0, 0.0,  1.0], # N5
        [0.0, 1.0,  1.0]] # N6
end

function get_interpolation_polynomial(::Type{Wedge6}, x)
    [
        1 x[1] x[2] x[3] x[1]*x[3] x[2]*x[3]
    ]
end

function get_interpolation_polynomial(::Type{Wedge6}, x, ::Type{Val{:partial_derivatives}})
    [
        0  1  0  0  x[3]  0   
        0  0  1  0  0     x[3]
        0  0  0  1  x[1]  x[2]
    ]
end

#

type Wedge15 <: AbstractElement
end

function description(::Type{Wedge15})
    "15 node prismatic element (wedge)"
end

function size(element::Element{Wedge15})
    return (3, 15)
end

function length(element::Element{Wedge15})
    return 15
end

function get_reference_coordinates(::Type{Wedge15})
    Vector{Float64}[
        [0.0, 0.0, -1.0], # N1
        [1.0, 0.0, -1.0], # N2
        [0.0, 1.0, -1.0], # N3
        [0.0, 0.0,  1.0], # N4
        [1.0, 0.0,  1.0], # N5
        [0.0, 1.0,  1.0], # N6
        [0.5, 0.0, -1.0], # N7
        [0.5, 0.5, -1.0], # N8
        [0.0, 0.5, -1.0], # N9
        [0.5, 0.0,  1.0], # N10
        [0.5, 0.5,  1.0], # N11
        [0.0, 0.5,  1.0], # N12
        [0.0, 0.0,  0.0], # N13
        [1.0, 0.0,  0.0], # N14
        [0.0, 1.0,  0.0]] # N15
end

function get_interpolation_polynomial(::Type{Wedge15}, x)
    [
        1 x[1] x[1]^2 x[2] x[1]*x[2] x[2]^2 x[3] x[1]*x[3] x[1]^2*x[3] x[2]*x[3] x[1]*x[2]*x[3] x[2]^2*x[3] x[3]^2 x[1]*x[3]^2 x[2]*x[3]^2
    ]
end

function get_interpolation_polynomial(::Type{Wedge15}, x, ::Type{Val{:partial_derivatives}})
    [
        0 1 2*x[1] 0 x[2] 0 0 x[3] 2*x[1]*x[3] 0 x[2]*x[3] 0 0 x[3]^2 0
        0 0 0 1 x[1] 2*x[2] 0 0 0 x[3] x[1]*x[3] 2*x[2]*x[3] 0 0 x[3]^2
        0 0 0 0 0 0 1 x[1] x[1]^2 x[2] x[1]*x[2] x[2]^2 2*x[3] 2*x[1]*x[3] 2*x[2]*x[3]
    ]
end

#

type Hex8 <: AbstractElement
end

function description(::Type{Hex8})
    "8 node hexahedral element"
end

function size(element::Element{Hex8})
    return (3, 8)
end

function length(element::Element{Hex8})
    return 8
end

function get_reference_coordinates(::Type{Hex8})
    Vector{Float64}[
        [-1.0, -1.0, -1.0], # N1
        [ 1.0, -1.0, -1.0], # N2
        [ 1.0,  1.0, -1.0], # N3
        [-1.0,  1.0, -1.0], # N4
        [-1.0, -1.0,  1.0], # N5
        [ 1.0, -1.0,  1.0], # N6
        [ 1.0,  1.0,  1.0], # N7
        [-1.0,  1.0,  1.0]] # N8
end

function get_interpolation_polynomial(::Type{Hex8}, xi)
    [
        1 xi[3] xi[2] xi[1] xi[2]*xi[3] xi[1]*xi[3] xi[1]*xi[2] xi[1]*xi[2]*xi[3]
    ]
end

function get_interpolation_polynomial(::Type{Hex8}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  0  1  0      xi[3]  xi[2]  xi[2]*xi[3]
        0  0  1  0  xi[3]  0      xi[1]  xi[1]*xi[3]
        0  1  0  0  xi[2]  xi[1]  0      xi[1]*xi[2]
    ]
end

#

type Hex20 <: AbstractElement
end

function description(::Type{Hex20})
    "20 node hexahedral element"
end

function size(element::Element{Hex20})
    return (3, 20)
end

function length(element::Element{Hex20})
    return 20
end

function get_reference_coordinates(::Type{Hex20})
    Vector{Float64}[
        [-1.0, -1.0, -1.0], # N1
        [ 1.0, -1.0, -1.0], # N2
        [ 1.0,  1.0, -1.0], # N3
        [-1.0,  1.0, -1.0], # N4
        [-1.0, -1.0,  1.0], # N5
        [ 1.0, -1.0,  1.0], # N6
        [ 1.0,  1.0,  1.0], # N7
        [-1.0,  1.0,  1.0], # N8
        [ 0.0, -1.0, -1.0], # N9
        [ 1.0,  0.0, -1.0], # N10
        [ 0.0,  1.0, -1.0], # N11
        [-1.0,  0.0, -1.0], # N12
        [-1.0, -1.0,  0.0], # N13
        [ 1.0, -1.0,  0.0], # N14
        [ 1.0,  1.0,  0.0], # N15
        [-1.0,  1.0,  0.0], # N16
        [ 0.0, -1.0,  1.0], # N17
        [ 1.0,  0.0,  1.0], # N18
        [ 0.0,  1.0,  1.0], # N19
        [-1.0,  0.0,  1.0]] # N20
end

function get_interpolation_polynomial(::Type{Hex20}, xi)
    [
        1 xi[3] xi[2] xi[1] xi[2]*xi[3] xi[1]*xi[3] xi[1]*xi[2] xi[1]*xi[2]*xi[3] xi[3]^2 xi[2]^2 xi[1]^2 xi[2]*xi[3]^2 xi[2]^2*xi[3] xi[1]*xi[3]^2 xi[1]*xi[2]^2 xi[1]^2*xi[3] xi[1]^2*xi[2] xi[1]*xi[2]*xi[3]^2 xi[1]*xi[2]^2*xi[3] xi[1]^2*xi[2]*xi[3]
    ]
end

function get_interpolation_polynomial(::Type{Hex20}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  0  1  0      xi[3]  xi[2]  xi[2]*xi[3]  0        0        2*xi[1]  0              0              xi[3]^2        xi[2]^2        2*xi[1]*xi[3]  2*xi[1]*xi[2]  xi[2]*xi[3]^2        xi[2]^2*xi[3]        2*xi[1]*xi[2]*xi[3]
        0  0  1  0  xi[3]  0      xi[1]  xi[1]*xi[3]  0        2*xi[2]  0        xi[3]^2        2*xi[2]*xi[3]  0              2*xi[1]*xi[2]  0              xi[1]^2        xi[1]*xi[3]^2        2*xi[1]*xi[2]*xi[3]  xi[1]^2*xi[3]
        0  1  0  0  xi[2]  xi[1]  0      xi[1]*xi[2]  2*xi[3]  0        0        2*xi[2]*xi[3]  xi[2]^2        2*xi[1]*xi[3]  0              xi[1]^2        0              2*xi[1]*xi[2]*xi[3]  xi[1]*xi[2]^2        xi[1]^2*xi[2]
    ]
end

###

type Hex27 <: AbstractElement
end

function description(::Type{Hex27})
    "27 node hexahedral element"
end

function size(element::Element{Hex27})
    return (3, 27)
end

function length(element::Element{Hex27})
    return 27
end

function get_reference_coordinates(::Type{Hex27})
    Vector{Float64}[
        [-1.0, -1.0, -1.0], # N1
        [ 1.0, -1.0, -1.0], # N2
        [ 1.0,  1.0, -1.0], # N3
        [-1.0,  1.0, -1.0], # N4
        [-1.0, -1.0,  1.0], # N5
        [ 1.0, -1.0,  1.0], # N6
        [ 1.0,  1.0,  1.0], # N7
        [-1.0,  1.0,  1.0], # N8
        [ 0.0, -1.0, -1.0], # N9
        [ 1.0,  0.0, -1.0], # N10
        [ 0.0,  1.0, -1.0], # N11
        [-1.0,  0.0, -1.0], # N12
        [-1.0, -1.0,  0.0], # N13
        [ 1.0, -1.0,  0.0], # N14
        [ 1.0,  1.0,  0.0], # N15
        [-1.0,  1.0,  0.0], # N16
        [ 0.0, -1.0,  1.0], # N17
        [ 1.0,  0.0,  1.0], # N18
        [ 0.0,  1.0,  1.0], # N19
        [-1.0,  0.0,  1.0], # N20
        [ 0.0,  0.0, -1.0], # N21
        [ 0.0, -1.0,  0.0], # N22
        [ 1.0,  0.0,  0.0], # N23
        [ 0.0,  1.0,  0.0], # N24
        [-1.0,  0.0,  0.0], # N25
        [ 0.0,  0.0,  1.0], # N26
        [ 0.0,  0.0,  0.0]] # N27
end

function get_interpolation_polynomial(::Type{Hex27}, xi)
    [
        1 xi[3] xi[2] xi[1] xi[2]*xi[3] xi[1]*xi[3] xi[1]*xi[2] xi[1]*xi[2]*xi[3] xi[3]^2 xi[2]^2 xi[1]^2 xi[2]*xi[3]^2 xi[2]^2*xi[3] xi[1]*xi[3]^2 xi[1]*xi[2]^2 xi[1]^2*xi[3] xi[1]^2*xi[2] xi[2]^2*xi[3]^2 xi[1]*xi[2]*xi[3]^2 xi[1]*xi[2]^2*xi[3] xi[1]^2*xi[3]^2 xi[1]^2*xi[2]*xi[3] xi[1]^2*xi[2]^2 xi[1]*xi[2]^2*xi[3]^2 xi[1]^2*xi[2]*xi[3]^2 xi[1]^2*xi[2]^2*xi[3] xi[1]^2*xi[2]^2*xi[3]^2
    ]
end

function get_interpolation_polynomial(::Type{Hex27}, xi, ::Type{Val{:partial_derivatives}})
    [
        0  0  0  1  0      xi[3]  xi[2]  xi[2]*xi[3]  0        0        2*xi[1]  0              0              xi[3]^2        xi[2]^2        2*xi[1]*xi[3]  2*xi[1]*xi[2]  0                xi[2]*xi[3]^2        xi[2]^2*xi[3]        2*xi[1]*xi[3]^2  2*xi[1]*xi[2]*xi[3]  2*xi[1]*xi[2]^2  xi[2]^2*xi[3]^2        2*xi[1]*xi[2]*xi[3]^2  2*xi[1]*xi[2]^2*xi[3]  2*xi[1]*xi[2]^2*xi[3]^2
        0  0  1  0  xi[3]  0      xi[1]  xi[1]*xi[3]  0        2*xi[2]  0        xi[3]^2        2*xi[2]*xi[3]  0              2*xi[1]*xi[2]  0              xi[1]^2        2*xi[2]*xi[3]^2  xi[1]*xi[3]^2        2*xi[1]*xi[2]*xi[3]  0                xi[1]^2*xi[3]        2*xi[1]^2*xi[2]  2*xi[1]*xi[2]*xi[3]^2  xi[1]^2*xi[3]^2        2*xi[1]^2*xi[2]*xi[3]  2*xi[1]^2*xi[2]*xi[3]^2
        0  1  0  0  xi[2]  xi[1]  0      xi[1]*xi[2]  2*xi[3]  0        0        2*xi[2]*xi[3]  xi[2]^2        2*xi[1]*xi[3]  0              xi[1]^2        0              2*xi[2]^2*xi[3]  2*xi[1]*xi[2]*xi[3]  xi[1]*xi[2]^2        2*xi[1]^2*xi[3]  xi[1]^2*xi[2]        0                2*xi[1]*xi[2]^2*xi[3]  2*xi[1]^2*xi[2]*xi[3]  xi[1]^2*xi[2]^2        2*xi[1]^2*xi[2]^2*xi[3]
    ]
end

###

macro create_basis(T)
    quote
        T = $T
        global get_basis, get_dbasis, length, size
        X = get_reference_coordinates(T)
        nbasis = length(X)
        A = zeros(nbasis, nbasis)
        for i=1:nbasis
            A[i,:] = get_interpolation_polynomial(T, X[i])
        end
        invA = inv(A)
        function get_basis(element::Element{$T}, ip, time)
            return get_interpolation_polynomial($T, ip)*invA
        end
        function get_dbasis(element::Element{$T}, ip, time)
            return get_interpolation_polynomial($T, ip, Val{:partial_derivatives})*invA
        end
    end
end

@create_basis Seg2
@create_basis Seg3
@create_basis Tri3
@create_basis Tri6
@create_basis Tri7
@create_basis Quad4
@create_basis Quad8
@create_basis Quad9
@create_basis Tet4
@create_basis Tet10
@create_basis Wedge6
@create_basis Wedge15
@create_basis Hex8
@create_basis Hex20
@create_basis Hex27

function inside(::Union{Type{Seg2}, Type{Seg3}, Type{Quad4}, Type{Quad8},
                        Type{Quad9}, Type{Hex8}, Type{Hex20}, Type{Hex27}}, xi)
    return all(-1.0 .<= xi .<= 1.0)
end

function inside(::Union{Type{Tri3}, Type{Tri6}, Type{Tri7}, Type{Tet4}, Type{Tet10}}, xi)
    return all(xi .>= 0.0) && (sum(xi) <= 1.0)
end

function get_reference_coordinates{E}(element::Element{E})
    get_reference_coordinates(E)
end

