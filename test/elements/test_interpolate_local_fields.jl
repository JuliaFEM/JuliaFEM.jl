# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using Tensors

# Helper to create UInt tuples
uint_tuple(n::Int) = tuple([UInt(i) for i in 1:n]...)

@testset "interpolate_local_fields" begin
    @testset "Single displacement field (quasi-static)" begin
        # Create element with displacement field
        S = @DOFSet{u::DOF{Displacement{3},Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        # Quasi-static: small deformation increment
        # Load step from u_old to u_new
        u_old = zeros(12)  # Initial config
        u_new = Float64[
            0.001, 0.0, 0.0,  # Node 1: small displacement in x
            0.0, 0.0, 0.0,     # Node 2
            0.0, 0.0, 0.0,     # Node 3
            0.0, 0.0, 0.0      # Node 4
        ]
        u_rate = zeros(12)  # Quasi-static: no velocity

        Δt = 1.0
        ξ = Vec((0.25, 0.25, 0.25))  # Tetrahedral center

        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Check structure
        @test haskey(local_fields, :u)
        @test local_fields.u isa LocalField

        # Check that rate is zero (quasi-static)
        @test local_fields.u.rate == zero(Vec{3})

        # Check that gradient_rate is NOT zero (computed from increment)
        @test local_fields.u.gradient_rate != zero(Tensor{2,3})

        # Check value interpolation (average of nodes weighted by basis functions)
        @test local_fields.u.value isa Vec{3}

        # Check gradient interpolation
        @test local_fields.u.gradient isa Tensor{2,3}

        # Type stability
        @inferred interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)
    end

    @testset "Single displacement field (dynamic)" begin
        # Create element
        S = @DOFSet{u::DOF{Displacement{3},Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        # Dynamic: with actual velocity
        u_old = zeros(12)
        u_new = Float64[
            0.001, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]
        u_rate = Float64[  # Actual velocity DOFs
            0.01, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]

        Δt = 0.1
        ξ = Vec((0.25, 0.25, 0.25))

        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Check that rate is NOT zero (dynamic)
        @test local_fields.u.rate != zero(Vec{3})
        @test local_fields.u.rate isa Vec{3}

        # Check that gradient_rate is computed from increment (not from ∇(u_rate))
        @test local_fields.u.gradient_rate isa Tensor{2,3}
    end

    @testset "Multi-field (thermoelasticity)" begin
        # Create element with displacement and temperature
        S = @DOFSet{
            u::DOF{Displacement{3},Vertex},
            T::DOF{Temperature,Vertex}
        }
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 16}(UInt(1), uint_tuple(16))

        # Setup fields
        u_old = zeros(16)
        u_new = zeros(16)
        u_new[1] = 0.001  # Small displacement
        u_new[13] = 300.0  # Temperature at node 1
        u_new[14] = 310.0  # Temperature at node 2
        u_new[15] = 305.0  # Temperature at node 3
        u_new[16] = 308.0  # Temperature at node 4

        u_rate = zeros(16)  # Quasi-static
        Δt = 1.0
        ξ = Vec((0.25, 0.25, 0.25))

        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Check both fields exist
        @test haskey(local_fields, :u)
        @test haskey(local_fields, :T)

        # Check displacement field
        @test local_fields.u isa LocalField
        @test local_fields.u.value isa Vec{3}
        @test local_fields.u.gradient isa Tensor{2,3}
        @test local_fields.u.rate isa Vec{3}
        @test local_fields.u.gradient_rate isa Tensor{2,3}

        # Check temperature field
        @test local_fields.T isa LocalField
        @test local_fields.T.value isa Float64
        @test local_fields.T.gradient isa Vec{3}
        @test local_fields.T.rate isa Float64
        @test local_fields.T.gradient_rate isa Vec{3}

        # Temperature should be interpolated (average of nodes)
        @test 300.0 <= local_fields.T.value <= 310.0
    end

    @testset "Integration with strain extraction" begin
        # Test complete workflow: Element → LocalField → Strain
        S = @DOFSet{u::DOF{Displacement{3},Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        # Setup deformation
        u_old = zeros(12)
        u_new = Float64[
            0.01, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]
        u_rate = zeros(12)
        Δt = 1.0
        ξ = Vec((0.25, 0.25, 0.25))

        # Interpolate to LocalField
        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Extract strain and strain rate
        ε = extract_strain(local_fields.u.gradient)
        ε̇ = extract_strain_rate(local_fields.u.gradient_rate)

        # Verify types
        @test ε isa SymmetricTensor{2,3}
        @test ε̇ isa SymmetricTensor{2,3}

        # Strain rate should not be zero (from increment)
        @test ε̇ != zero(SymmetricTensor{2,3})
    end

    @testset "Zero allocations" begin
        # Test that interpolation is zero-allocation
        S = @DOFSet{u::DOF{Displacement{3},Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        u_old = zeros(12)
        u_new = rand(12)
        u_rate = zeros(12)
        Δt = 1.0
        ξ = Vec((0.25, 0.25, 0.25))

        # Warmup
        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Check allocations
        allocs = @allocated interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)
        @test allocs == 0
    end

    @testset "Gradient rate from increments" begin
        # Verify that gradient_rate is computed from increments
        S = @DOFSet{u::DOF{Displacement{3},Vertex}}
        elem = Element{Tetrahedron{4}, Lagrange{1}, S, 12}(UInt(1), uint_tuple(12))

        # Two configurations
        u_old = zeros(12)
        u_new = Float64[
            0.01, 0.0, 0.0,
            0.0, 0.02, 0.0,
            0.0, 0.0, 0.03,
            0.0, 0.0, 0.0
        ]
        u_rate = zeros(12)
        Δt = 2.0
        ξ = Vec((0.25, 0.25, 0.25))

        local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)

        # Manually compute gradient rate from interpolate_fields
        fields_new = interpolate_fields(elem, u_new, ξ)
        fields_old = interpolate_fields(elem, u_old, ξ)
        ∇u_rate_manual = (fields_new.∇u - fields_old.∇u) / Δt

        # Should match
        @test local_fields.u.gradient_rate ≈ ∇u_rate_manual
    end
end
