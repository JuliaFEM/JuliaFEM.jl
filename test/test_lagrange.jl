# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

ALL_ELEMENTS = [
    Seg2, Seg3,
    Tri3, Tri6, Tri7,
    Quad4, Quad8, Quad9,
    Tet4, Tet10,
    Wedge6,
    Hex8, Hex20, Hex27
]

@testset "Evaluating basis" begin
    for T in ALL_ELEMENTS
        el = Element(T)
        nnodes = length(el)
        for (i, X) in enumerate(get_reference_coordinates(T))
            Ni = vec(el(X))
            expected = zeros(nnodes)
            expected[i] = 1.0
            @test isapprox(Ni, expected)
        end
    end
end

function get_volume{T<:AbstractElement}(::Type{T})
    X = get_reference_coordinates(T)
    element = Element(T)
    update!(element, "geometry", X)
    V = 0.0
    for ip in get_integration_points(element)
        V += ip.weight*element(ip, 0.0, Val{:detJ})
    end
    return V
end

@testset "Calculate reference element length/area/volume" begin
    @test isapprox(get_volume(Seg2),  2.0)
    @test isapprox(get_volume(Seg3),  2.0)
    @test isapprox(get_volume(Tri3),  0.5)
    @test isapprox(get_volume(Tri6),  0.5)
    @test isapprox(get_volume(Tri7),  0.5)
    @test isapprox(get_volume(Quad4), 2.0^2)
    @test isapprox(get_volume(Quad8), 2.0^2)
    @test isapprox(get_volume(Quad9), 2.0^2)
    @test isapprox(get_volume(Tet4),  1/6)
    @test isapprox(get_volume(Tet10), 1/6)
    @test isapprox(get_volume(Wedge6), 1.0)
    @test isapprox(get_volume(Hex8),  2.0^3)
    @test isapprox(get_volume(Hex20), 2.0^3)
    @test isapprox(get_volume(Hex27), 2.0^3)
end

