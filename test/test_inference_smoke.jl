# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Curated `@inferred` checks on small, stable API entry points.

These catch accidental type-instability regressions on hot metadata paths
without running a whole-package abstract interpretation tool.
"""

using JuliaFEM
using Tensors: SymmetricTensor, Tensor
using Test

function run_inference_smoke_tests()
    @testset "dof_size inferrable" begin
        @test (@inferred dof_size(Float64)) === 1
        @test (@inferred dof_size(Vec{2})) === 2
        @test (@inferred dof_size(Vec{3})) === 3
        @test (@inferred dof_size(Tensor{2, 3})) === 9
        @test (@inferred dof_size(SymmetricTensor{2, 3})) === 6
    end

    @testset "topology counts inferrable" begin
        @test (@inferred nnodes(Hex8)) === 8
        @test (@inferred nnodes(Tet4)) === 4
        @test (@inferred dim(Hex8())) === 3
        @test (@inferred dim(Tet4())) === 3
    end

    @testset "DOF set metadata inferrable" begin
        S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
        @test (@inferred field_count(S)) === 1
        @test (@inferred is_single_field(S)) === true
        Sm = @DOFSet{
            T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex},
        }
        @test (@inferred field_count(Sm)) === 2
        @test (@inferred is_single_field(Sm)) === false
    end

    return nothing
end
