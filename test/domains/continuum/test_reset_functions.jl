# Test reset! functions for all cache types

using JuliaFEM
using Tensors
using LinearAlgebra
using SparseArrays  # For nnz()
using Test

include("test_helpers.jl")

@testset "reset! functions" begin
    kernel = create_test_kernel()
    mesh = create_test_mesh()

    # Create caches
    N = 8      # Nodes per element (Hex8)
    NIP = 8    # Integration points (Gauss{2} for Hex8)

    geometry_cache = JuliaFEM.create_geometry_cache(N, NIP)
    element_cache = JuliaFEM.create_element_cache(mesh, kernel)
    material_cache = JuliaFEM.create_material_cache(kernel.material, NIP)

    @testset "reset!(GeometryCache)" begin
        # Populate with data first
        elem_id = 1
        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)

        # Verify data exists
        @test any(x -> norm(x) > 0, geometry_cache.X)
        @test any(x -> x != 0, geometry_cache.detJ_w)

        # Reset and verify zeros
        JuliaFEM.reset!(geometry_cache)

        @test all(x -> x == zero(Vec{3,Float64}), geometry_cache.X)
        @test all(x -> x == 0.0, geometry_cache.detJ_w)
        for q in 1:NIP
            @test all(x -> x == zero(Vec{3,Float64}), view(geometry_cache.∇N_data, q, :))
        end

        # Test zero allocations (warm-up first)
        JuliaFEM.reset!(geometry_cache)
        allocs = @allocated JuliaFEM.reset!(geometry_cache)
        @test allocs == 0
    end

    @testset "reset!(ElementCache)" begin
        # Populate with data first
        elem_id = 1
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, nothing)

        # Verify data exists
        @test any(x -> x != 0, element_cache.dofs)

        # Reset and verify zeros
        JuliaFEM.reset!(element_cache)

        @test all(x -> x == 0, element_cache.dofs)
        @test all(x -> x == zero(Tensor{2,3,Float64}), element_cache.K_blocks)
        @test all(x -> x == zero(Vec{3,Float64}), element_cache.f_blocks)
        @test all(x -> x == zero(Vec{3,Float64}), element_cache.u_buffer)

        # Test zero allocations (warm-up first)
        JuliaFEM.reset!(element_cache)
        allocs = @allocated JuliaFEM.reset!(element_cache)
        @test allocs == 0
    end

    @testset "reset!(MaterialStateCache)" begin
        # Populate with data first (need geometry cache)
        elem_id = 1
        u_global = nothing
        state_old = create_material_state(kernel, mesh)
        Δt = 0.01

        JuliaFEM.update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
        JuliaFEM.update_element_cache!(element_cache, kernel, elem_id, mesh, u_global)
        JuliaFEM.update_material_cache!(material_cache, geometry_cache, kernel.material,
            element_cache, state_old, elem_id, Δt)

        # Verify data exists (tangent should be non-zero for linear elastic)
        @test any(q -> norm(material_cache.𝔻[q]) > 0, 1:NIP)

        # Reset and verify zeros
        JuliaFEM.reset!(material_cache)

        @test all(q -> material_cache.σ[q] == zero(SymmetricTensor{2,3,Float64}), 1:NIP)
        @test all(q -> material_cache.𝔻[q] == zero(SymmetricTensor{4,3,Float64}), 1:NIP)

        # Test zero allocations (warm-up first)
        JuliaFEM.reset!(material_cache)
        allocs = @allocated JuliaFEM.reset!(material_cache)
        @test allocs == 0
    end

    @testset "reset!(COOCache)" begin
        # Create COO cache
        assembler = COOAssembler()
        coo_cache = create_cache(assembler, mesh, kernel)

        # Populate with data by assembling
        assemble!(coo_cache, assembler, kernel, mesh)

        # Verify data exists by extracting system (matrix should have entries)
        K, f = extract_system(coo_cache)
        @test nnz(K) > 0  # Assembly produced triplets
        # Note: force vector f is zero for zero displacement with no body forces

        # Reset and verify zeros
        JuliaFEM.reset!(coo_cache)

        @test coo_cache.counter[] == 0  # Counter is reset
        @test all(x -> x == 0.0, coo_cache.f)

        # After reset, extraction should give empty/zero matrix
        K2, f2 = extract_system(coo_cache)
        @test nnz(K2) == 0  # No triplets after reset

        # Test zero allocations (warm-up first)
        JuliaFEM.reset!(coo_cache)
        allocs = @allocated JuliaFEM.reset!(coo_cache)
        @test allocs == 0
    end
end
