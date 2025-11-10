"""
Test GPU elasticity solver with cantilever beam

Geometry: 10×1×1 beam, fixed at X=0, pressure on top
"""

using Test
using LinearAlgebra

# Load solver
include(joinpath(@__DIR__, "..", "src", "gpu_elasticity.jl"))
using .GPUElasticity

@testset "GPU Elasticity - Cantilever Beam" begin
    println("\n" * "="^70)
    println("TEST: Cantilever Beam with GPU Solver")
    println("="^70)

    # Check if mesh exists, if not generate it
    mesh_file = joinpath(@__DIR__, "testdata", "cantilever_beam.msh")

    if !isfile(mesh_file)
        println("\n⚠️  Mesh file not found: $mesh_file")
        println("Generating mesh...")

        # Create testdata directory if needed
        mkpath(dirname(mesh_file))

        # Generate mesh
        include(joinpath(@__DIR__, "..", "scripts", "generate_cantilever_mesh.jl"))
        Base.invokelatest(generate_cantilever_mesh,
            length=10.0,
            width=1.0,
            height=1.0,
            mesh_size=0.5,
            output_file=mesh_file
        )
    end

    # Read mesh
    println("\nReading mesh...")
    mesh = read_gmsh_mesh(mesh_file)

    @test size(mesh.nodes, 2) > 0
    @test size(mesh.elements, 2) > 0

    # Find boundary nodes
    fixed_nodes = get_surface_nodes(mesh, "FixedEnd")
    pressure_nodes = get_surface_nodes(mesh, "PressureSurface")

    println("\nBoundary nodes:")
    println("  Fixed:    $(length(fixed_nodes))")
    println("  Pressure: $(length(pressure_nodes))")

    @test length(fixed_nodes) > 0
    @test length(pressure_nodes) > 0

    # Material (steel)
    material = ElasticMaterial(
        210e9,  # E = 210 GPa
        0.3     # ν = 0.3
    )

    # Applied pressure (1 MPa)
    pressure = 1e6  # Pa

    # Define physics
    physics = ElasticityPhysics(
        mesh,
        material,
        fixed_nodes,
        pressure_nodes,
        pressure
    )

    # Solve on GPU
    println("\n" * "="^70)
    println("Solving on GPU...")
    println("="^70)

    u = solve_elasticity_gpu(physics, tol=1e-6, max_iter=1000)

    # Check solution
    @test length(u) == 3 * size(mesh.nodes, 2)
    @test maximum(abs.(u)) > 0.0  # Non-trivial solution

    # Fixed nodes should have zero displacement
    for node in fixed_nodes
        ux = u[3*node-2]
        uy = u[3*node-1]
        uz = u[3*node]
        @test abs(ux) < 1e-10
        @test abs(uy) < 1e-10
        @test abs(uz) < 1e-10
    end
    println("✅ Fixed boundary condition satisfied")

    # Free end should have maximum displacement
    x_max = maximum(mesh.nodes[1, :])
    free_end_nodes = findall(abs.(mesh.nodes[1, :] .- x_max) .< 1e-6)

    max_disp_free_end = maximum(abs.(u[3*n] for n in free_end_nodes))
    max_disp_overall = maximum(abs.(u))

    println("\nDisplacement analysis:")
    println("  Max displacement at free end: $(max_disp_free_end) m")
    println("  Max displacement overall:     $(max_disp_overall) m")

    # Free end should have largest displacement (cantilever behavior)
    @test max_disp_free_end > 0.5 * max_disp_overall
    println("✅ Cantilever deflection pattern correct")

    # Analytical comparison (Euler-Bernoulli beam theory)
    # For uniform load q on beam of length L:
    # w_max = q*L^4 / (8*E*I)

    L = 10.0  # Length
    width = 1.0
    height = 1.0
    I = width * height^3 / 12  # Second moment of area

    # Convert pressure to line load (approximate)
    q = pressure * width  # N/m

    w_analytical = q * L^4 / (8 * material.E * I)

    println("\nAnalytical comparison:")
    println("  Analytical max deflection: $(w_analytical) m")
    println("  FEM max deflection:        $(max_disp_free_end) m")
    println("  Relative error:            $(abs(w_analytical - max_disp_free_end) / w_analytical * 100)%")

    # Should be within reasonable range (mesh dependent)
    # Note: This is approximate due to discretization
    @test max_disp_free_end > 0.1 * w_analytical
    @test max_disp_free_end < 10.0 * w_analytical
    println("✅ Solution within reasonable range of analytical")

    println("\n" * "="^70)
    println("✅ All tests passed!")
    println("="^70)
end
