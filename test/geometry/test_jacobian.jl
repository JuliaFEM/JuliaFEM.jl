# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "Jacobian helpers" begin
    @testset "compute_jacobian tuple (triangle)" begin
        X = (Vec{2}((0.0, 0.0)), Vec{2}((2.0, 0.0)), Vec{2}((0.0, 1.5)))
        xi = Vec{2}((1 / 3, 1 / 3))
        dN = Tuple(get_basis_derivatives(Triangle{3}(), Lagrange{1}(), xi))
        J = compute_jacobian(X, dN)
        @test J ≈ Tensor{2,2}((2.0, 0.0, 0.0, 1.5))
        @test det(J) ≈ 3.0
    end

    @testset "compute_jacobian AbstractVector" begin
        Xv = [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((0.0, 1.0))]
        xi = Vec{2}((0.2, 0.2))
        dNsv = get_basis_derivatives(Triangle{3}(), Lagrange{1}(), xi)
        dNv = collect(dNsv)
        J1 = compute_jacobian(Xv, dNv)
        J2 = compute_jacobian((Xv...,), Tuple(dNsv))
        @test isapprox(J1, J2; rtol=1e-14)
    end

    @testset "physical_derivatives" begin
        X = (Vec{2}((0.0, 0.0)), Vec{2}((2.0, 0.0)), Vec{2}((0.0, 1.5)))
        xi = Vec{2}((1 / 3, 1 / 3))
        dN = Tuple(get_basis_derivatives(Triangle{3}(), Lagrange{1}(), xi))
        J = compute_jacobian(X, dN)
        dNdx_t = physical_derivatives(J, dN)
        dNdx_v = physical_derivatives(J, collect(dN))
        @test length(dNdx_t) == 3
        @test length(dNdx_v) == 3
        for i in 1:3
            @test dNdx_t[i] ≈ dNdx_v[i]
        end
        @test sum(dNdx_t) ≈ Vec{2}((0.0, 0.0))
    end
end
