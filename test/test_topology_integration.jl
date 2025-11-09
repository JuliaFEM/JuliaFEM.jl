# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM

@testset "Topology and Integration: Complete test suite" begin

    # ========================================================================
    # TOPOLOGY: 1D SEGMENTS
    # ========================================================================
    @testset "Seg2 topology" begin
        topo = Seg2()
        @test nnodes(topo) == 2
        @test dim(topo) == 1

        coords = reference_coordinates(topo)
        @test coords isa NTuple{2,NTuple{1,Float64}}
        @test coords[1] == (-1.0,)
        @test coords[2] == (1.0,)

        e = edges(topo)
        @test e isa NTuple{1,Tuple{Int,Int}}
        @test e[1] == (1, 2)

        @test faces(topo) == ()
    end

    @testset "Seg3 topology" begin
        topo = Seg3()
        @test nnodes(topo) == 3
        @test dim(topo) == 1

        coords = reference_coordinates(topo)
        @test coords[1] == (-1.0,)
        @test coords[2] == (1.0,)
        @test coords[3] == (0.0,)
    end

    # ========================================================================
    # TOPOLOGY: 2D TRIANGLES
    # ========================================================================
    @testset "Tri3 topology" begin
        topo = Tri3()
        @test nnodes(topo) == 3
        @test dim(topo) == 2

        coords = reference_coordinates(topo)
        @test coords isa NTuple{3,NTuple{2,Float64}}
        @test coords[1] == (0.0, 0.0)
        @test coords[2] == (1.0, 0.0)
        @test coords[3] == (0.0, 1.0)

        e = edges(topo)
        @test e isa NTuple{3,Tuple{Int,Int}}
        @test length(e) == 3

        f = faces(topo)
        @test f isa NTuple{1,NTuple{3,Int}}
        @test f[1] == (1, 2, 3)
    end

    @testset "Tri6 topology" begin
        topo = Tri6()
        @test nnodes(topo) == 6
        @test dim(topo) == 2

        coords = reference_coordinates(topo)
        @test coords[4] == (0.5, 0.0)  # Edge node
        @test coords[5] == (0.5, 0.5)  # Edge node
        @test coords[6] == (0.0, 0.5)  # Edge node
    end

    @testset "Tri7 topology" begin
        topo = Tri7()
        @test nnodes(topo) == 7
        @test coords = reference_coordinates(topo)
        @test coords[7] ≈ (1 / 3, 1 / 3)  # Center node
    end

    # ========================================================================
    # TOPOLOGY: 2D QUADRILATERALS
    # ========================================================================
    @testset "Quad4 topology" begin
        topo = Quad4()
        @test nnodes(topo) == 4
        @test dim(topo) == 2

        coords = reference_coordinates(topo)
        @test coords isa NTuple{4,NTuple{2,Float64}}
        @test coords[1] == (-1.0, -1.0)
        @test coords[2] == (1.0, -1.0)
        @test coords[3] == (1.0, 1.0)
        @test coords[4] == (-1.0, 1.0)

        e = edges(topo)
        @test length(e) == 4

        f = faces(topo)
        @test f[1] == (1, 2, 3, 4)
    end

    @testset "Quad8 topology" begin
        topo = Quad8()
        @test nnodes(topo) == 8
        @test dim(topo) == 2

        coords = reference_coordinates(topo)
        @test coords[5] == (0.0, -1.0)  # Edge node
        @test coords[8] == (-1.0, 0.0)  # Edge node
    end

    @testset "Quad9 topology" begin
        topo = Quad9()
        @test nnodes(topo) == 9
        coords = reference_coordinates(topo)
        @test coords[9] == (0.0, 0.0)  # Center node
    end

    # ========================================================================
    # TOPOLOGY: 3D TETRAHEDRA
    # ========================================================================
    @testset "Tet4 topology" begin
        topo = Tet4()
        @test nnodes(topo) == 4
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords isa NTuple{4,NTuple{3,Float64}}
        @test coords[1] == (0.0, 0.0, 0.0)
        @test coords[2] == (1.0, 0.0, 0.0)
        @test coords[3] == (0.0, 1.0, 0.0)
        @test coords[4] == (0.0, 0.0, 1.0)

        e = edges(topo)
        @test length(e) == 6  # Tet has 6 edges

        f = faces(topo)
        @test length(f) == 4  # Tet has 4 triangular faces
    end

    @testset "Tet10 topology" begin
        topo = Tet10()
        @test nnodes(topo) == 10
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords[5] == (0.5, 0.0, 0.0)  # Edge node
    end

    # ========================================================================
    # TOPOLOGY: 3D HEXAHEDRA
    # ========================================================================
    @testset "Hex8 topology" begin
        topo = Hex8()
        @test nnodes(topo) == 8
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords isa NTuple{8,NTuple{3,Float64}}
        @test coords[1] == (-1.0, -1.0, -1.0)
        @test coords[7] == (1.0, 1.0, 1.0)

        e = edges(topo)
        @test length(e) == 12  # Hex has 12 edges

        f = faces(topo)
        @test length(f) == 6  # Hex has 6 quadrilateral faces
    end

    @testset "Hex20 topology" begin
        topo = Hex20()
        @test nnodes(topo) == 20
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords[9] == (0.0, -1.0, -1.0)  # Edge node
    end

    @testset "Hex27 topology" begin
        topo = Hex27()
        @test nnodes(topo) == 27
        coords = reference_coordinates(topo)
        @test coords[27] == (0.0, 0.0, 0.0)  # Volume center node
    end

    # ========================================================================
    # TOPOLOGY: 3D PYRAMIDS
    # ========================================================================
    @testset "Pyr5 topology" begin
        topo = Pyr5()
        @test nnodes(topo) == 5
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords[5] == (0.0, 0.0, 1.0)  # Apex

        e = edges(topo)
        @test length(e) == 8  # 4 base + 4 to apex

        f = faces(topo)
        @test length(f) == 5  # 1 quad base + 4 triangular
    end

    # ========================================================================
    # TOPOLOGY: 3D WEDGES
    # ========================================================================
    @testset "Wedge6 topology" begin
        topo = Wedge6()
        @test nnodes(topo) == 6
        @test dim(topo) == 3

        coords = reference_coordinates(topo)
        @test coords[1] == (0.0, 0.0, -1.0)  # Bottom triangle
        @test coords[4] == (0.0, 0.0, 1.0)   # Top triangle

        e = edges(topo)
        @test length(e) == 9  # 3 bottom + 3 top + 3 vertical

        f = faces(topo)
        @test length(f) == 5  # 2 triangular + 3 quadrilateral
    end

    @testset "Wedge15 topology" begin
        topo = Wedge15()
        @test nnodes(topo) == 15
        @test dim(topo) == 3
    end

    # ========================================================================
    # INTEGRATION: GAUSS QUADRATURE
    # ========================================================================
    @testset "Integration points structure" begin
        ip = IntegrationPoint{2}((0.5, 0.5), 1.0)
        @test ip.ξ == (0.5, 0.5)
        @test ip.weight == 1.0
        @test ip.ξ isa NTuple{2,Float64}
    end

    @testset "Gauss quadrature for Seg2" begin
        ips = integration_points(Gauss{2}(), Seg2())
        @test ips isa Tuple
        @test length(ips) == 2  # 2-point Gauss rule
        @test all(ip -> ip isa IntegrationPoint{1}, ips)

        # Check weights sum correctly
        total_weight = sum(ip.weight for ip in ips)
        @test total_weight ≈ 2.0  # Domain [-1,1] has length 2
    end

    @testset "Gauss quadrature for Tri3" begin
        ips1 = integration_points(Gauss{1}(), Tri3())
        @test length(ips1) == 1  # 1-point rule
        @test ips1[1].ξ ≈ (1 / 3, 1 / 3)  # Centroid
        @test ips1[1].weight ≈ 0.5  # Triangle area

        ips3 = integration_points(Gauss{3}(), Tri3())
        @test length(ips3) == 3  # 3-point rule

        # Check weights sum to triangle area
        total_weight = sum(ip.weight for ip in ips3)
        @test total_weight ≈ 0.5
    end

    @testset "Gauss quadrature for Quad4" begin
        ips1 = integration_points(Gauss{1}(), Quad4())
        @test length(ips1) == 1  # 1-point rule

        ips2 = integration_points(Gauss{2}(), Quad4())
        @test length(ips2) == 4  # 2² = 4 points

        ips3 = integration_points(Gauss{3}(), Quad4())
        @test length(ips3) == 9  # 3² = 9 points

        # Check weights sum to square area
        total_weight = sum(ip.weight for ip in ips2)
        @test total_weight ≈ 4.0  # Domain [-1,1]² has area 4
    end

    @testset "Gauss quadrature for Tet4" begin
        ips = integration_points(Gauss{1}(), Tet4())
        @test length(ips) == 1
        @test all(ip -> ip isa IntegrationPoint{3}, ips)
    end

    @testset "Gauss quadrature for Hex8" begin
        ips1 = integration_points(Gauss{1}(), Hex8())
        @test length(ips1) == 1  # 1-point rule

        ips2 = integration_points(Gauss{2}(), Hex8())
        @test length(ips2) == 8  # 2³ = 8 points

        ips3 = integration_points(Gauss{3}(), Hex8())
        @test length(ips3) == 27  # 3³ = 27 points

        # Check weights sum to cube volume
        total_weight = sum(ip.weight for ip in ips2)
        @test total_weight ≈ 8.0  # Domain [-1,1]³ has volume 8
    end

    @testset "Gauss quadrature for Wedge6" begin
        ips = integration_points(Gauss{6}(), Wedge6())
        @test length(ips) == 6
        @test all(ip -> ip isa IntegrationPoint{3}, ips)
    end

    @testset "Gauss quadrature for Pyr5" begin
        ips = integration_points(Gauss{5}(), Pyr5())
        @test length(ips) == 5
        @test all(ip -> ip isa IntegrationPoint{3}, ips)
    end

    # ========================================================================
    # INTEGRATION: HIGHER ORDER ELEMENTS
    # ========================================================================
    @testset "Quadratic elements use same quadrature" begin
        # Tri3 and Tri6 can use same rules
        ips_tri3 = integration_points(Gauss{3}(), Tri3())
        ips_tri6 = integration_points(Gauss{3}(), Tri6())
        @test length(ips_tri3) == length(ips_tri6)

        # Quad4 and Quad9 can use same rules
        ips_quad4 = integration_points(Gauss{2}(), Quad4())
        ips_quad9 = integration_points(Gauss{2}(), Quad9())
        @test length(ips_quad4) == length(ips_quad9)

        # Hex8 and Hex27 can use same rules
        ips_hex8 = integration_points(Gauss{2}(), Hex8())
        ips_hex27 = integration_points(Gauss{2}(), Hex27())
        @test length(ips_hex8) == length(ips_hex27)
    end

    # ========================================================================
    # ZERO-ALLOCATION VERIFICATION
    # ========================================================================
    @testset "Zero-allocation design" begin
        # Topology functions return tuples
        @test reference_coordinates(Tri3()) isa NTuple
        @test edges(Quad4()) isa NTuple
        @test faces(Hex8()) isa NTuple

        # Integration points return tuple
        @test integration_points(Gauss{1}(), Tri3()) isa Tuple

        # IntegrationPoint.ξ is tuple
        ip = first(integration_points(Gauss{1}(), Tri3()))
        @test ip.ξ isa NTuple
    end

    # ========================================================================
    # API COMPLETENESS
    # ========================================================================
    @testset "All topology types exported" begin
        @test isdefined(JuliaFEM, :Seg2)
        @test isdefined(JuliaFEM, :Seg3)
        @test isdefined(JuliaFEM, :Tri3)
        @test isdefined(JuliaFEM, :Tri6)
        @test isdefined(JuliaFEM, :Tri7)
        @test isdefined(JuliaFEM, :Quad4)
        @test isdefined(JuliaFEM, :Quad8)
        @test isdefined(JuliaFEM, :Quad9)
        @test isdefined(JuliaFEM, :Tet4)
        @test isdefined(JuliaFEM, :Tet10)
        @test isdefined(JuliaFEM, :Hex8)
        @test isdefined(JuliaFEM, :Hex20)
        @test isdefined(JuliaFEM, :Hex27)
        @test isdefined(JuliaFEM, :Pyr5)
        @test isdefined(JuliaFEM, :Wedge6)
        @test isdefined(JuliaFEM, :Wedge15)
    end

    @testset "Integration types exported" begin
        @test isdefined(JuliaFEM, :Gauss)
        @test isdefined(JuliaFEM, :IntegrationPoint)
        @test isdefined(JuliaFEM, :integration_points)
    end

end
