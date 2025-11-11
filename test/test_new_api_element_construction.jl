# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM

@testset "New API: Element Construction with Topology + Basis" begin

    @testset "2D Triangle Elements" begin
        # Linear triangle (P1, 3 nodes)
        topology = Triangle()
        basis = Lagrange{Triangle,1}()

        @test dim(topology) == 2
        @test nnodes(basis) == 3

        # Element construction (new way)
        element = Element(basis, (UInt(1), UInt(2), UInt(3)))

        @test element.connectivity == (UInt(1), UInt(2), UInt(3))
        @test element.basis == basis

        # Quadratic triangle (P2, 6 nodes)
        basis_p2 = Lagrange{Triangle,2}()
        @test nnodes(basis_p2) == 6

        element_p2 = Element(basis_p2,
            (UInt(1), UInt(2), UInt(3), UInt(4), UInt(5), UInt(6)))
        @test length(element_p2.connectivity) == 6
    end

    @testset "3D Tetrahedral Elements" begin
        # Linear tetrahedron (P1, 4 nodes)
        topology = Tetrahedron()
        basis = Lagrange{Tetrahedron,1}()

        @test dim(topology) == 3
        @test nnodes(basis) == 4

        # Element construction
        conn = (UInt(1), UInt(2), UInt(3), UInt(4))
        element = Element(basis, conn)

        @test element.connectivity == conn
        @test element.basis == basis

        # Quadratic tetrahedron (P2, 10 nodes)
        basis_p2 = Lagrange{Tetrahedron,2}()
        @test nnodes(basis_p2) == 10

        conn_p2 = tuple(UInt.(1:10)...)
        element_p2 = Element(basis_p2, conn_p2)
        @test length(element_p2.connectivity) == 10
    end

    @testset "2D Quadrilateral Elements" begin
        # Linear quad (Q1, 4 nodes)
        topology = Quadrilateral()
        basis = Lagrange{Quadrilateral,1}()

        @test dim(topology) == 2
        @test nnodes(basis) == 4

        conn = (UInt(1), UInt(2), UInt(3), UInt(4))
        element = Element(basis, conn)

        @test element.connectivity == conn
        @test element.basis == basis

        # Quadratic quad (Q2, 9 nodes)
        basis_p2 = Lagrange{Quadrilateral,2}()
        @test nnodes(basis_p2) == 9

        conn_p2 = tuple(UInt.(1:9)...)
        element_p2 = Element(basis_p2, conn_p2)
        @test length(element_p2.connectivity) == 9
    end

    @testset "3D Hexahedral Elements" begin
        # Linear hex (Q1, 8 nodes)
        topology = Hexahedron()
        basis = Lagrange{Hexahedron,1}()

        @test dim(topology) == 3
        @test nnodes(basis) == 8

        conn = tuple(UInt.(1:8)...)
        element = Element(basis, conn)

        @test element.connectivity == conn
        @test element.basis == basis

        # Quadratic hex (Q2, 27 nodes)
        basis_p2 = Lagrange{Hexahedron,2}()
        @test nnodes(basis_p2) == 27

        conn_p2 = tuple(UInt.(1:27)...)
        element_p2 = Element(basis_p2, conn_p2)
        @test length(element_p2.connectivity) == 27
    end

    @testset "Backward Compatibility Aliases" begin
        # Old aliases still work (deprecated but functional)

        # Tet4 is alias for Tetrahedron
        @test Tetrahedron() isa Tetrahedron
        @test Tet4() isa Tetrahedron
        @test Tet4 === Tetrahedron

        # Tri3 is alias for Triangle
        @test Triangle() isa Triangle
        @test Tri3() isa Triangle
        @test Tri3 === Triangle

        # Quad4 is alias for Quadrilateral  
        @test Quadrilateral() isa Quadrilateral
        @test Quad4() isa Quadrilateral
        @test Quad4 === Quadrilateral

        # Hex8 is alias for Hexahedron
        @test Hexahedron() isa Hexahedron
        @test Hex8() isa Hexahedron
        @test Hex8 === Hexahedron
    end

    @testset "Topology Properties Independent of Basis" begin
        # Topology describes geometry only
        topology = Tetrahedron()

        # Geometric properties don't depend on basis
        @test dim(topology) == 3

        ref_coords = reference_coordinates(topology)
        @test length(ref_coords) == 4  # 4 corner nodes
        @test ref_coords[1] == (0.0, 0.0, 0.0)
        @test ref_coords[2] == (1.0, 0.0, 0.0)
        @test ref_coords[3] == (0.0, 1.0, 0.0)
        @test ref_coords[4] == (0.0, 0.0, 1.0)

        # Edges (6 for tetrahedron)
        edge_list = edges(topology)
        @test length(edge_list) == 6

        # Faces (4 triangular faces)
        face_list = faces(topology)
        @test length(face_list) == 4

        # These are SAME regardless of basis degree!
        basis_p1 = Lagrange{Tetrahedron,1}()
        basis_p2 = Lagrange{Tetrahedron,2}()

        @test nnodes(basis_p1) == 4   # Different node counts
        @test nnodes(basis_p2) == 10

        # But topology properties are identical
        @test dim(topology) == 3  # Same for both
        @test length(edges(topology)) == 6  # Same
        @test length(faces(topology)) == 4  # Same
    end

    @testset "Element with Fields (Type-Stable)" begin
        # New API: Type-stable fields using NamedTuple
        basis = Lagrange{Triangle,1}()
        conn = (UInt(1), UInt(2), UInt(3))

        # Material properties as NamedTuple (type-stable!)
        fields = (E=210e9, ν=0.3, thickness=0.01)

        element = Element(UInt(42), conn, (), fields, basis)

        @test element.fields.E == 210e9
        @test element.fields.ν == 0.3
        @test element.fields.thickness == 0.01
        @test element.id == UInt(42)

        # Type is known at compile time
        @test typeof(element.fields) <: NamedTuple
        @test fieldnames(typeof(element.fields)) == (:E, :ν, :thickness)
    end

    @testset "Integration Points with New API" begin
        # Integration points are element property
        topology = Triangle()
        basis = Lagrange{Triangle,1}()
        scheme = Gauss{2}()

        # Get integration points for this topology
        ips = integration_points(scheme, topology)

        @test length(ips) > 0
        @test all(ip -> ip isa IntegrationPoint, ips)

        # Create element with integration points
        conn = (UInt(1), UInt(2), UInt(3))
        element = Element(UInt(1), conn, ips, (), basis)

        @test length(element.integration_points) == length(ips)
        @test element.integration_points == ips
    end
end

@testset "New API: Separation of Concerns" begin

    @testset "Topology = Geometry Only" begin
        # Topology describes ONLY the reference element shape
        topo_tet = Tetrahedron()
        topo_tri = Triangle()
        topo_quad = Quadrilateral()
        topo_hex = Hexahedron()

        # These have NO information about:
        # - Number of nodes (depends on basis degree)
        # - Shape functions (comes from basis)
        # - Integration points (comes from integration scheme)
        # - Material properties (comes from fields)
        # - Physical coordinates (comes from mesh)

        @test topo_tet isa AbstractTopology
        @test topo_tri isa AbstractTopology
        @test topo_quad isa AbstractTopology
        @test topo_hex isa AbstractTopology
    end

    @testset "Basis = Interpolation Scheme" begin
        # Basis describes HOW to interpolate fields

        # Linear bases (P1)
        basis_tri_p1 = Lagrange{Triangle,1}()
        basis_tet_p1 = Lagrange{Tetrahedron,1}()

        # Quadratic bases (P2)
        basis_tri_p2 = Lagrange{Triangle,2}()
        basis_tet_p2 = Lagrange{Tetrahedron,2}()

        # Node count determined by basis + topology
        @test nnodes(basis_tri_p1) == 3
        @test nnodes(basis_tri_p2) == 6
        @test nnodes(basis_tet_p1) == 4
        @test nnodes(basis_tet_p2) == 10

        # All are basis functions
        @test basis_tri_p1 isa AbstractBasis
        @test basis_tet_p2 isa AbstractBasis
    end

    @testset "Integration = Quadrature Rule" begin
        # Integration scheme is independent choice
        scheme_1pt = Gauss{1}()
        scheme_2pt = Gauss{2}()
        scheme_3pt = Gauss{3}()

        @test scheme_1pt isa AbstractIntegration
        @test scheme_2pt isa AbstractIntegration
        @test scheme_3pt isa AbstractIntegration

        # Same topology, different integration rules
        topo = Triangle()

        ips_1 = integration_points(scheme_1pt, topo)
        ips_2 = integration_points(scheme_2pt, topo)
        ips_3 = integration_points(scheme_3pt, topo)

        # Different number of integration points
        @test length(ips_1) < length(ips_2) < length(ips_3)
    end

    @testset "Element = Topology + Basis + Integration + Fields" begin
        # Element combines all pieces

        topology = Triangle()           # Geometry
        basis = Lagrange{Triangle,1}() # Interpolation (3 nodes)
        scheme = Gauss{2}()             # Integration rule
        conn = (UInt(1), UInt(2), UInt(3))  # Node IDs
        fields = (E=210e9, ν=0.3)   # Material properties

        ips = integration_points(scheme, topology)
        element = Element(UInt(1), conn, ips, fields, basis)

        # Element has all information needed for FEM
        @test element.connectivity == conn
        @test element.integration_points == ips
        @test element.fields == fields
        @test element.basis == basis

        # Can query properties
        @test nnodes(element.basis) == 3
        @test length(element.integration_points) > 0
        @test element.fields.E == 210e9
    end
end

@testset "New API: Type Stability Benefits" begin

    @testset "Compile-Time Known Sizes" begin
        # All sizes known at compile time for optimization

        basis = Lagrange{Tetrahedron,1}()
        conn = (UInt(1), UInt(2), UInt(3), UInt(4))
        fields = (E=210e3, ν=0.3)

        element = Element(UInt(1), conn, (), fields, basis)

        # Type parameters encode sizes
        @test element isa Element{4,0,typeof(fields),typeof(basis)}

        # Connectivity is NTuple (stack-allocated, zero-cost)
        @test element.connectivity isa NTuple{4,UInt}

        # Fields are NamedTuple (type-stable, fast access)
        @test element.fields isa NamedTuple

        # Compiler knows exact types → can optimize aggressively
        E_val = element.fields.E
        @test E_val isa Float64  # Exact type known
    end

    @testset "No Allocations in Hot Paths" begin
        # Test that element access doesn't allocate

        basis = Lagrange{Triangle,1}()
        conn = (UInt(1), UInt(2), UInt(3))
        fields = (E=210e3, ν=0.3, ρ=7850.0)

        element = Element(UInt(1), conn, (), fields, basis)

        # Access should not allocate
        allocs = @allocated begin
            _ = element.connectivity
            _ = element.fields.E
            _ = element.fields.ν
            _ = element.basis
        end

        @test allocs == 0  # Zero allocations!
    end
end

println("✅ All New API element construction tests passed!")
println("\nKey Takeaways:")
println("  • Topology = Geometry (Tetrahedron, Triangle, etc.)")
println("  • Basis = Interpolation (Lagrange{Topology, Degree})")
println("  • Element = Topology + Basis + Integration + Fields")
println("  • Old names (Tet4, Tri3) are aliases for backward compatibility")
println("  • Node count comes from BASIS, not topology!")
