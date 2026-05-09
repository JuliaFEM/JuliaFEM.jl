# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using SparseArrays
using LinearAlgebra

@testset "SparseMatrixCOO" begin
    @testset "constructors and empty" begin
        A = SparseMatrixCOO()
        @test isempty(A)
        @test size(A) == (0, 0)
        v = SparseVectorCOO([1, 3], [2.0, 4.0])
        @test v.J == [1, 1]
        @test Vector(v) ≈ [2.0, 0.0, 4.0]
        @test Vector(v, 5) ≈ [2.0, 0.0, 4.0, 0.0, 0.0]
    end

    @testset "convert and sparse" begin
        C = sparse([1, 2], [2, 1], [3.0, 4.0], 3, 3)
        A = convert(SparseMatrixCOO, C)
        @test sparse(A, 3, 3) ≈ C
        @test Matrix(sparse(A, 3, 3)) ≈ Matrix(C)

        M = [0.0 2.0; 3.0 0.0]
        B = convert(SparseMatrixCOO, M)
        @test Matrix(B) ≈ M

        b = [0.0, 5.0, 0.0, 6.0]
        bv = convert(SparseMatrixCOO, b)
        @test sparse(bv, 4, 1) ≈ sparsevec(b)

        sv = sparsevec([2, 4], [7.0, 8.0], 5)
        cv = convert(SparseVectorCOO, sv)
        @test sparsevec(cv, 5) ≈ sv
    end

    @testset "add!, append!, size, isapprox" begin
        K = SparseMatrixCOO()
        JuliaFEM.add!(K, 2, 3, 1.5)
        JuliaFEM.add!(K, 4, 9.0)
        @test size(K, 1) == 4
        @test size(K, 2) == 3

        ke = Float64[1 2; 3 4]
        JuliaFEM.add!(K, [1, 2], [10, 11], ke)

        B = SparseMatrixCOO([3], [3], [1.0])
        append!(K, B)
        @test !isempty(K)

        Bcsc = sparse([1, 1], [1, 2], [0.1, 0.2], 2, 2)
        JuliaFEM.add!(K, Bcsc)

        a2 = sparsevec([1], [3.0], 2)
        w = SparseVectorCOO()
        JuliaFEM.add!(w, a2)
        @test Vector(w, 2) ≈ [3.0, 0.0]

        JuliaFEM.add!(K, [1, 2], [0.5, 1.5], 1)
        Kcopy = SparseMatrixCOO(copy(K.I), copy(K.J), copy(K.V))
        n, m = size(K)
        @test sparse(Kcopy, n, m) ≈ sparse(K, n, m)
        JuliaFEM.empty!(Kcopy)
        @test isempty(Kcopy)
    end

    @testset "add! error on vector block mismatch" begin
        K = SparseMatrixCOO()
        @test_throws ErrorException JuliaFEM.add!(K, [1, 2], [1.0], 1)
    end
end
