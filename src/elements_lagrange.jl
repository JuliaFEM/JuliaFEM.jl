# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FEMBasis

#    "Poi1" => (0, 1),
"1 node discrete point element",
type Poi1 <: AbstractBasis
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

function size(::Type{Poi1})
    return (0, 1)
end

function length(::Type{Poi1})
    return 1
end

function FEMBasis.get_reference_element_coordinates(::Type{Poi1})
    Vector{Float64}[[0.0]]
end

function get_basis{B}(element::Element{B}, ip, time)
    T = typeof(first(ip))
    N = zeros(T, 1, length(B))
    eval_basis!(B, N, tuple(ip...))
    return N
end

function get_dbasis{B}(element::Element{B}, ip, time)
    T = typeof(first(ip))
    dN = zeros(T, size(B)...)
    eval_dbasis!(B, dN, tuple(ip...))
    return dN
end

function inside(::Union{Type{Seg2}, Type{Seg3}, Type{Quad4}, Type{Quad8},
                        Type{Quad9}, Type{Pyr5}, Type{Hex8}, Type{Hex20},
                        Type{Hex27}}, xi)
    return all(-1.0 .<= xi .<= 1.0)
end

function inside(::Union{Type{Tri3}, Type{Tri6}, Type{Tri7}, Type{Tet4}, Type{Tet10}}, xi)
    return all(xi .>= 0.0) && (sum(xi) <= 1.0)
end

function get_reference_coordinates{B}(element::Element{B})
    return get_reference_element_coordinates(B)
end

