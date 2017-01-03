# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
using JuliaFEM: description

ALL_ELEMENTS = [
    Seg2, Seg3,
    Tri3, Tri6, Tri7,
    Quad4, Quad8, Quad9,
    Tet4, Tet10,
    Wedge6,
    Hex8, Hex20, Hex27
]

ALL_ELEMENTS_NODES = [
    [1,2], [1,2,3],
    [1,2,3], [1,2,3,4,5,6], [1,2,3,4,5,6,7],
    [1,2,3,4], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9],
    [1,2,3,4], [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6],
    [1,2,3,4,5,6,7,8],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15,17,18,19,20],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15,17,18,19,20,
    21,22,23,24,25,26,27]
]

@testset "Evaluating basis" begin
    for (T, nod) in zip(ALL_ELEMENTS,ALL_ELEMENTS_NODES)
        el = Element(T,nod)
        nnodes = length(el)
        for (i, X) in enumerate(get_reference_coordinates(T))
            Ni = vec(el(X))
            expected = zeros(nnodes)
            expected[i] = 1.0
            @test isapprox(Ni, expected)
        end
    end
end

function get_volume{T<:AbstractElement}(::Type{T},nodes)
    X = get_reference_coordinates(T)
    element = Element(T,nodes)
    update!(element, "geometry", X)
    V = 0.0
    for ip in get_integration_points(element)
        V += ip.weight*element(ip, 0.0, Val{:detJ})
    end
    return V
end

RESULTS = [2.0, 2.0, 0.5, 0.5, 0.5, 2.0^2, 2.0^2,
           2.0^2, 1/6, 1/6, 1.0, 2.0^3, 2.0^3, 2.0^3,]

@testset "Calculate reference element length/area/volume" begin
    for (T, nod, res) in zip(ALL_ELEMENTS,ALL_ELEMENTS_NODES,
                             RESULTS)
        @test isapprox(get_volume(T,nod),  res)
    end
end

SIZES = [(1,2), (1,3), (2,3), (2,6), (2,7), (2,4),
         (2,8), (2,9), (3,4), (3,10), (3,6)]

@testset "element size" begin
    for (T, nod, res) in zip(ALL_ELEMENTS,ALL_ELEMENTS_NODES, SIZES)
        @test size(Element(T,nod)) == res
    end
end

@testset "element length" begin
    for i in 1:length(ALL_ELEMENTS)
        typ = ALL_ELEMENTS[i]
        vec = ALL_ELEMENTS_NODES[i]
        el = Element(typ,vec)
        @test length(el) == length(vec)
    end
end

DESC = ["2 node segment", "3 node segment", "3 node triangle",
        "6 node triangle", "7 node triangle", "4 node quadrangle",
        "8 node Serendip quadrangle", "9 node quadrangle",
        "4 node tetrahedral element", "10 node tetrahedral element",
        "6 node prismatic element (wedge)",
        "8 node hexahedral element", "20 node hexahedral element",
        "27 node hexahedral element"]

@testset "element description" begin
    for (T, res) in zip(ALL_ELEMENTS, DESC)
        @test description(Type(T)) == res
    end
end
