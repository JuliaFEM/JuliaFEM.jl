# Simple standalone test for topology and integration modules
# Does not load full JuliaFEM to avoid conflicts

push!(LOAD_PATH, "/home/juajukka/dev/JuliaFEM.jl/src")

# Load only what we need
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/topology.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/seg2.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/seg3.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/tri3.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/tri6.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/tri7.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/quad4.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/quad8.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/quad9.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/tet4.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/tet10.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/hex8.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/hex20.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/hex27.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/pyr5.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/wedge6.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/topology/wedge15.jl")

# Need quadrature data
include("/home/juajukka/dev/JuliaFEM.jl/src/quadrature/quaddata.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/quadrature/gltri.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/quadrature/glquad.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/quadrature/gltet.jl")

# Load integration
include("/home/juajukka/dev/JuliaFEM.jl/src/integration/integration.jl")
include("/home/juajukka/dev/JuliaFEM.jl/src/integration/gauss.jl")

using Test

println("="^70)
println("STANDALONE TOPOLOGY & INTEGRATION TEST")
println("="^70)

@testset "Topology standalone" begin
    @testset "Seg2" begin
        t = Seg2()
        @test nnodes(t) == 2
        @test dim(t) == 1
        coords = reference_coordinates(t)
        @test coords[1] == (-1.0,)
        @test coords[2] == (1.0,)
        println("✓ Seg2: 2 nodes, 1D")
    end

    @testset "Tri3" begin
        t = Tri3()
        @test nnodes(t) == 3
        @test dim(t) == 2
        coords = reference_coordinates(t)
        @test coords[1] == (0.0, 0.0)
        @test coords[2] == (1.0, 0.0)
        @test coords[3] == (0.0, 1.0)
        e = edges(t)
        @test length(e) == 3
        println("✓ Tri3: 3 nodes, 2D, 3 edges")
    end

    @testset "Quad4" begin
        t = Quad4()
        @test nnodes(t) == 4
        @test dim(t) == 2
        coords = reference_coordinates(t)
        @test coords[1] == (-1.0, -1.0)
        e = edges(t)
        @test length(e) == 4
        println("✓ Quad4: 4 nodes, 2D, 4 edges")
    end

    @testset "Tet4" begin
        t = Tet4()
        @test nnodes(t) == 4
        @test dim(t) == 3
        coords = reference_coordinates(t)
        @test coords[1] == (0.0, 0.0, 0.0)
        e = edges(t)
        @test length(e) == 6
        f = faces(t)
        @test length(f) == 4
        println("✓ Tet4: 4 nodes, 3D, 6 edges, 4 faces")
    end

    @testset "Hex8" begin
        t = Hex8()
        @test nnodes(t) == 8
        @test dim(t) == 3
        e = edges(t)
        @test length(e) == 12
        f = faces(t)
        @test length(f) == 6
        println("✓ Hex8: 8 nodes, 3D, 12 edges, 6 faces")
    end
end

@testset "Integration standalone" begin
    @testset "IntegrationPoint" begin
        ip = IntegrationPoint{2}((0.5, 0.5), 1.0)
        @test ip.ξ == (0.5, 0.5)
        @test ip.weight == 1.0
        println("✓ IntegrationPoint structure works")
    end

    @testset "Gauss{1} + Tri3" begin
        ips = integration_points(Gauss{1}(), Tri3())
        @test length(ips) == 1
        @test all(ips[1].ξ[i] ≈ (1 / 3, 1 / 3)[i] for i in 1:2)
        @test ips[1].weight ≈ 0.5
        println("✓ Gauss{1} + Tri3: 1 point at centroid")
    end

    @testset "Gauss{3} + Tri3" begin
        ips = integration_points(Gauss{3}(), Tri3())
        @test length(ips) == 3
        total = sum(ip.weight for ip in ips)
        @test total ≈ 0.5
        println("✓ Gauss{3} + Tri3: 3 points, weights sum to 0.5")
    end

    @testset "Gauss{2} + Quad4" begin
        ips = integration_points(Gauss{2}(), Quad4())
        @test length(ips) == 4  # 2² points
        total = sum(ip.weight for ip in ips)
        @test total ≈ 4.0  # Area of [-1,1]²
        println("✓ Gauss{2} + Quad4: 4 points, weights sum to 4.0")
    end

    @testset "Gauss{1} + Tet4" begin
        ips = integration_points(Gauss{1}(), Tet4())
        @test length(ips) == 1
        @test all(ip -> ip isa IntegrationPoint{3}, ips)
        println("✓ Gauss{1} + Tet4: 1 point (3D)")
    end

    @testset "Gauss{2} + Hex8" begin
        ips = integration_points(Gauss{2}(), Hex8())
        @test length(ips) == 8  # 2³ points
        total = sum(ip.weight for ip in ips)
        @test total ≈ 8.0  # Volume of [-1,1]³
        println("✓ Gauss{2} + Hex8: 8 points, weights sum to 8.0")
    end
end

println("\n" * "="^70)
println("ALL TESTS PASSED ✓")
println("="^70)
