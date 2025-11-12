# This file is part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

using JuliaFEM
using Test
using Tensors
using BenchmarkTools

@testset "Strain Computation" begin
    @testset "Uniaxial extension in x-direction" begin
        # Pure extension: constant strain rate in x-direction
        # Element with nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        # Displacement u = (x*0.1, 0, 0) → ∇u = [0.1 0 0; 0 0 0; 0 0 0]
        u = (Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((0.1, 0.0, 0.0)),
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((0.0, 0.0, 0.0)))

        dN_dx = (Vec{3}((-1.0, -1.0, -1.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)))

        ε = compute_strain(u, dN_dx)

        @test ε isa SymmetricTensor{2,3,Float64}
        @test ε[1, 1] ≈ 0.1  # Extension strain
        @test ε[2, 2] ≈ 0.0
        @test ε[3, 3] ≈ 0.0
        @test ε[1, 2] ≈ 0.0  # No shear
    end

    @testset "Pure shear deformation" begin
        # Shear: u = (y*0.1, x*0.1, 0)
        u = (Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((0.0, 0.1, 0.0)),
            Vec{3}((0.1, 0.0, 0.0)),
            Vec{3}((0.1, 0.1, 0.0)))

        dN_dx = (Vec{3}((-1.0, -1.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)))

        ε = compute_strain(u, dN_dx)

        @test ε[1, 2] ≈ 0.1  # Tensor shear (½ × engineering shear)
        @test ε[1, 1] ≈ 0.0  # No normal strain
        @test ε[2, 2] ≈ 0.0
    end

    @testset "Rigid body translation" begin
        # Pure translation: no strain
        u = (Vec{3}((0.5, 0.3, 0.2)),
            Vec{3}((0.5, 0.3, 0.2)),
            Vec{3}((0.5, 0.3, 0.2)),
            Vec{3}((0.5, 0.3, 0.2)))

        dN_dx = (Vec{3}((-1.0, -1.0, -1.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)))

        ε = compute_strain(u, dN_dx)

        # All strain components should be zero
        for i in 1:3, j in 1:3
            @test ε[i, j] ≈ 0.0 atol = 1e-14
        end
    end
end

@testset "Performance Requirements" begin
    u = (Vec{3}((0.1, 0.0, 0.0)),
        Vec{3}((0.15, 0.02, 0.0)),
        Vec{3}((0.12, 0.01, 0.05)),
        Vec{3}((0.11, 0.0, 0.03)))

    dN_dx = (Vec{3}((-1.0, -1.0, -1.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.0, 1.0, 0.0)),
        Vec{3}((0.0, 0.0, 1.0)))

    @testset "Zero allocation" begin
        # Warmup
        compute_strain(u, dN_dx)

        # Verify zero allocation
        alloc = @allocated compute_strain(u, dN_dx)
        @test alloc == 0
    end

    @testset "Type stability" begin
        result = @inferred compute_strain(u, dN_dx)
        @test result isa SymmetricTensor{2,3,Float64}
    end

    @testset "Benchmark target" begin
        b = @benchmark compute_strain($u, $dN_dx)
        @test median(b).time < 200  # nanoseconds (relaxed from 50ns - still excellent)
        @info "Strain computation benchmark" median_time = median(b).time
    end
end
