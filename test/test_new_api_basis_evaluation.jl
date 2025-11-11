# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Test basis function evaluation using NEW API

This demonstrates the correct usage:
1. Topology defines geometry (Tetrahedron, Triangle, etc.)
2. Basis defines interpolation (Lagrange{Topology, Degree})
3. evaluate_basis() bridges topology + basis + integration point
"""

using Test
using JuliaFEM
using StaticArrays

# Mock implementation for demonstration (to be implemented in src/basis/evaluation.jl)
struct BasisValues{N,D}
    N::SVector{N,Float64}                    # Shape function values
    dN_dξ::SVector{N,SVector{D,Float64}}    # Derivatives w.r.t. parametric coords
end

"""
Mock evaluate_basis for testing (SIMPLIFIED - real implementation more complex)
"""
function evaluate_basis_mock(
    ::Tetrahedron,
    ::Lagrange{Tetrahedron,1},
    ip::IntegrationPoint{3}
)
    ξ, η, ζ = ip.ξ

    # Linear tetrahedral shape functions (P1)
    # N1 = 1 - ξ - η - ζ
    # N2 = ξ
    # N3 = η
    # N4 = ζ
    N = SVector(1 - ξ - η - ζ, ξ, η, ζ)

    # Derivatives w.r.t. parametric coordinates
    # dN/dξ = [dN1/dξ, dN1/dη, dN1/dζ]
    dN_dξ = SVector(
        SVector(-1.0, -1.0, -1.0),  # ∇N1 in parametric space
        SVector(1.0, 0.0, 0.0),  # ∇N2
        SVector(0.0, 1.0, 0.0),  # ∇N3
        SVector(0.0, 0.0, 1.0)   # ∇N4
    )

    return BasisValues(N, dN_dξ)
end

function evaluate_basis_mock(
    ::Triangle,
    ::Lagrange{Triangle,1},
    ip::IntegrationPoint{2}
)
    ξ, η = ip.ξ

    # Linear triangle shape functions (P1)
    # N1 = 1 - ξ - η
    # N2 = ξ
    # N3 = η
    N = SVector(1 - ξ - η, ξ, η)

    # Derivatives
    dN_dξ = SVector(
        SVector(-1.0, -1.0),  # ∇N1
        SVector(1.0, 0.0),  # ∇N2
        SVector(0.0, 1.0)   # ∇N3
    )

    return BasisValues(N, dN_dξ)
end

@testset "New API: Basis Function Evaluation" begin

    @testset "Linear Tetrahedron (P1, 4 nodes)" begin
        topology = Tetrahedron()
        basis = Lagrange{Tetrahedron,1}()

        @test dim(topology) == 3
        @test nnodes(basis) == 4

        # Evaluate at element center (ξ=η=ζ=0.25)
        ip_center = IntegrationPoint((0.25, 0.25, 0.25), 1.0)
        bv = evaluate_basis_mock(topology, basis, ip_center)

        # Check partition of unity
        @test sum(bv.N) ≈ 1.0

        # At center, all shape functions should be equal
        @test all(n -> isapprox(n, 0.25, atol=1e-14), bv.N)

        # Check derivatives (constant for linear elements)
        @test bv.dN_dξ[1] == SVector(-1.0, -1.0, -1.0)
        @test bv.dN_dξ[2] == SVector(1.0, 0.0, 0.0)
        @test bv.dN_dξ[3] == SVector(0.0, 1.0, 0.0)
        @test bv.dN_dξ[4] == SVector(0.0, 0.0, 1.0)

        # Evaluate at corner nodes
        # Node 1: (0,0,0) → N1=1, others=0
        ip_n1 = IntegrationPoint((0.0, 0.0, 0.0), 1.0)
        bv_n1 = evaluate_basis_mock(topology, basis, ip_n1)
        @test bv_n1.N[1] ≈ 1.0
        @test bv_n1.N[2] ≈ 0.0
        @test bv_n1.N[3] ≈ 0.0
        @test bv_n1.N[4] ≈ 0.0

        # Node 2: (1,0,0) → N2=1, others=0
        ip_n2 = IntegrationPoint((1.0, 0.0, 0.0), 1.0)
        bv_n2 = evaluate_basis_mock(topology, basis, ip_n2)
        @test bv_n2.N[1] ≈ 0.0
        @test bv_n2.N[2] ≈ 1.0
        @test bv_n2.N[3] ≈ 0.0
        @test bv_n2.N[4] ≈ 0.0

        # Node 3: (0,1,0) → N3=1
        ip_n3 = IntegrationPoint((0.0, 1.0, 0.0), 1.0)
        bv_n3 = evaluate_basis_mock(topology, basis, ip_n3)
        @test bv_n3.N[3] ≈ 1.0
        @test sum(bv_n3.N) - bv_n3.N[3] ≈ 0.0 atol = 1e-14

        # Node 4: (0,0,1) → N4=1
        ip_n4 = IntegrationPoint((0.0, 0.0, 1.0), 1.0)
        bv_n4 = evaluate_basis_mock(topology, basis, ip_n4)
        @test bv_n4.N[4] ≈ 1.0
        @test sum(bv_n4.N) - bv_n4.N[4] ≈ 0.0 atol = 1e-14
    end

    @testset "Linear Triangle (P1, 3 nodes)" begin
        topology = Triangle()
        basis = Lagrange{Triangle,1}()

        @test dim(topology) == 2
        @test nnodes(basis) == 3

        # Evaluate at triangle center (ξ=η=1/3)
        ip_center = IntegrationPoint((1 / 3, 1 / 3), 0.5)
        bv = evaluate_basis_mock(topology, basis, ip_center)

        # Partition of unity
        @test sum(bv.N) ≈ 1.0

        # At center, all should be equal
        @test all(n -> isapprox(n, 1 / 3, atol=1e-14), bv.N)

        # Check derivatives
        @test bv.dN_dξ[1] == SVector(-1.0, -1.0)
        @test bv.dN_dξ[2] == SVector(1.0, 0.0)
        @test bv.dN_dξ[3] == SVector(0.0, 1.0)

        # Corner nodes
        # Node 1: (0,0)
        ip_n1 = IntegrationPoint((0.0, 0.0), 0.5)
        bv_n1 = evaluate_basis_mock(topology, basis, ip_n1)
        @test bv_n1.N[1] ≈ 1.0
        @test bv_n1.N[2] ≈ 0.0
        @test bv_n1.N[3] ≈ 0.0

        # Node 2: (1,0)
        ip_n2 = IntegrationPoint((1.0, 0.0), 0.5)
        bv_n2 = evaluate_basis_mock(topology, basis, ip_n2)
        @test bv_n2.N[2] ≈ 1.0

        # Node 3: (0,1)
        ip_n3 = IntegrationPoint((0.0, 1.0), 0.5)
        bv_n3 = evaluate_basis_mock(topology, basis, ip_n3)
        @test bv_n3.N[3] ≈ 1.0
    end

    @testset "Integration with Gauss Points" begin
        # Real workflow: evaluate basis at all integration points

        topology = Tetrahedron()
        basis = Lagrange{Tetrahedron,1}()
        scheme = Gauss{1}()  # 1-point rule for tetrahedron

        # Get integration points
        ips = integration_points(scheme, topology)
        @test length(ips) > 0

        # Evaluate basis at each integration point
        basis_values = map(ips) do ip
            evaluate_basis_mock(topology, basis, ip)
        end

        @test length(basis_values) == length(ips)

        # Each should satisfy partition of unity
        for bv in basis_values
            @test sum(bv.N) ≈ 1.0
        end
    end

    @testset "Type Stability" begin
        # Check that return types are fully inferred

        topology = Tetrahedron()
        basis = Lagrange{Tetrahedron,1}()
        ip = IntegrationPoint((0.25, 0.25, 0.25), 1.0)

        bv = evaluate_basis_mock(topology, basis, ip)

        # Type should be concrete
        @test isconcretetype(typeof(bv))
        @test isconcretetype(typeof(bv.N))
        @test isconcretetype(typeof(bv.dN_dξ))

        # SVector ensures stack allocation (no heap allocation)
        @test bv.N isa SVector{4,Float64}
        @test bv.dN_dξ isa SVector{4,SVector{3,Float64}}
    end

    @testset "Zero Allocations" begin
        # Evaluation should not allocate

        topology = Tetrahedron()
        basis = Lagrange{Tetrahedron,1}()
        ip = IntegrationPoint((0.25, 0.25, 0.25), 1.0)

        # First call (compilation)
        _ = evaluate_basis_mock(topology, basis, ip)

        # Subsequent calls should be zero-allocation
        allocs = @allocated evaluate_basis_mock(topology, basis, ip)
        @test allocs == 0
    end
end

@testset "New API: Element + Basis Workflow" begin

    @testset "Complete FEM Workflow Mockup" begin
        # 1. Define element
        topology = Triangle()
        basis = Lagrange{Triangle,1}()
        scheme = Gauss{2}()
        conn = (UInt(1), UInt(2), UInt(3))

        # 2. Get integration points
        ips = integration_points(scheme, topology)

        # 3. Create element
        element = Element(UInt(1), conn, ips, (), basis)

        # 4. Evaluate basis at all integration points
        basis_at_ips = map(ips) do ip
            evaluate_basis_mock(topology, basis, ip)
        end

        @test length(basis_at_ips) == length(ips)
        @test all(bv -> sum(bv.N) ≈ 1.0, basis_at_ips)

        # This demonstrates the data flow:
        # Topology → Integration Points → Basis Values → Element Matrices
    end

    @testset "Multiple Element Types from Same Topology" begin
        # Same topology, different basis degrees

        topology = Tetrahedron()
        scheme = Gauss{2}()

        # Linear element (P1, 4 nodes)
        basis_p1 = Lagrange{Tetrahedron,1}()
        conn_p1 = tuple(UInt.(1:4)...)

        ips = integration_points(scheme, topology)
        element_p1 = Element(UInt(1), conn_p1, ips, (), basis_p1)

        @test nnodes(element_p1.basis) == 4

        # Quadratic element (P2, 10 nodes)
        basis_p2 = Lagrange{Tetrahedron,2}()
        conn_p2 = tuple(UInt.(1:10)...)

        element_p2 = Element(UInt(2), conn_p2, ips, (), basis_p2)

        @test nnodes(element_p2.basis) == 10

        # Same topology, same integration points, different basis!
        @test element_p1.integration_points == element_p2.integration_points
    end
end

println("✅ All New API basis evaluation tests passed!")
println("\nKey API Pattern:")
println("  topology = Tetrahedron()  # Geometry")
println("  basis = Lagrange{Tetrahedron, 1}()  # Interpolation (4 nodes)")
println("  ip = IntegrationPoint(ξ, w)  # Quadrature point")
println("  bv = evaluate_basis(topology, basis, ip)  # Get N and ∇N")
println("\nThis separates concerns: Topology ≠ Basis ≠ Integration!")
