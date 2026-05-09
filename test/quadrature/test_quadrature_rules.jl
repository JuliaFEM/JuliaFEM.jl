# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors
function _sum_weights(topo, rule)
    pts = get_quadrature_points(topo, rule)
    return sum(p.weight for p in pts)
end

@testset "Quadrature rules and API" begin
    @test default_quadrature(2) isa GaussLegendre{3}
    @test default_quadrature(Triangle{3}) isa GaussLegendre
    @test default_quadrature(Hexahedron{8}) isa GaussLegendre

    @test JuliaFEM.npoints(Triangle, GaussLegendre{2}()) == 3
    @test JuliaFEM.npoints(Segment, GaussLegendre{3}()) == 3

    @test integration_points(Triangle{3}()) isa NTuple
    @test JuliaFEM.npoints(Triangle{3}()) ≥ 1

    # Segment tensor rules (exercise gl_tensor_product.jl)
    for N in 1:5
        rule = GaussLegendre{N}()
        sw = _sum_weights(Segment, rule)
        @test sw ≈ 2.0 rtol = 1e-14
    end

    # Quadrilateral tensor rules
    for N in 1:5
        rule = GaussLegendre{N}()
        sw = _sum_weights(Quadrilateral, rule)
        @test sw ≈ 4.0 rtol = 1e-14
    end

    # Hexahedron tensor rules
    for N in 1:5
        rule = GaussLegendre{N}()
        sw = _sum_weights(Hexahedron, rule)
        @test sw ≈ 8.0 rtol = 1e-14
    end

    # Triangle specialised rules (variants + higher order)
    @test _sum_weights(Triangle, GaussLegendre{1}()) ≈ 0.5
    @test _sum_weights(Triangle, GaussLegendre{2,:B}()) ≈ 0.5
    @test _sum_weights(Triangle, GaussLegendre{3,:B}()) ≈ 0.5 rtol = 1e-12
    for N in 4:5
        @test _sum_weights(Triangle, GaussLegendre{N}()) ≈ 0.5 rtol = 1e-8
    end
    @test _sum_weights(Triangle, GaussLegendre{6}()) ≈ 0.5 rtol = 1e-8

    # Tetrahedra (gl_tetrahedra.jl)
    for N in 1:4
        @test _sum_weights(Tetrahedron, GaussLegendre{N}()) ≈ 1 / 6 rtol = 1e-12
    end

    # Wedges / pyramids (non-tensor rules; weight sums match reference volume)
    @test sum(p.weight for p in get_quadrature_points(Wedge, GaussLegendre{2}())) ≈ 1.0 rtol = 1e-14
    @test sum(p.weight for p in get_quadrature_points(Wedge, GaussLegendre{2,:B}())) ≈ 1.0 rtol = 1e-14
    @test sum(p.weight for p in get_quadrature_points(Wedge, GaussLegendre{5}())) ≈ 1.0 rtol = 1e-14
    @test sum(p.weight for p in get_quadrature_points(Pyramid, GaussLegendre{2,:default}())) ≈ 7.86962962962963 rtol = 1e-12
    @test sum(p.weight for p in get_quadrature_points(Pyramid, GaussLegendre{2,:B}())) ≈ 2 / 3 rtol = 1e-14

    qp = QuadraturePoint(Vec{2}((0.25, 0.25)), 0.5)
    @test qp.coords isa Vec{2,Float64}
    @test QuadraturePoint(Vec{1}((0.3,)), 1.1).weight == 1.1
end
