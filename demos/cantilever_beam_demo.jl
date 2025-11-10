"""
Demo: Solve cantilever beam problem with GPU

This demonstrates the complete workflow:
1. Generate mesh with Gmsh
2. Read mesh
3. Define material and boundary conditions
4. Solve on GPU
5. Visualize results
"""

# Load the GPU elasticity module
include(joinpath(@__DIR__, "..", "src", "gpu_elasticity.jl"))
using .GPUElasticity
using .GPUElasticity.GmshReader: get_surface_nodes
using Printf

function main()
    println("\n" * "="^70)
    println("Demo: Cantilever Beam on GPU")
    println("="^70)

    # Step 1: Generate mesh (if not exists)
    mesh_file = joinpath(@__DIR__, "..", "test", "testdata", "cantilever_beam.msh")

    if !isfile(mesh_file)
        println("\nGenerating mesh...")
        mkpath(dirname(mesh_file))

        include(joinpath(@__DIR__, "..", "scripts", "generate_cantilever_mesh.jl"))
        Base.invokelatest(generate_cantilever_mesh,
            length=10.0,
            width=1.0,
            height=1.0,
            mesh_size=0.5,
            output_file=mesh_file
        )
    else
        println("\nUsing existing mesh: $mesh_file")
    end

    # Step 2: Read mesh
    println("\nReading mesh...")
    mesh = read_gmsh_mesh(mesh_file)

    # Step 3: Define boundary conditions
    println("\nDefining boundary conditions...")

    # Fixed end (X = 0)
    fixed_nodes = get_surface_nodes(mesh, "FixedEnd")
    println("  Fixed nodes: $(length(fixed_nodes))")

    # Pressure surface (Z = max)
    pressure_nodes = get_surface_nodes(mesh, "PressureSurface")
    println("  Pressure nodes: $(length(pressure_nodes))")

    # Step 4: Define material (steel)
    material = ElasticMaterial(
        210e9,  # E = 210 GPa
        0.3     # ν = 0.3
    )
    println("\nMaterial: Steel")
    println("  E = $(material.E / 1e9) GPa")
    println("  ν = $(material.ν)")

    # Step 5: Define load (1 MPa pressure on top)
    pressure = 1e6  # Pa
    println("\nLoad: Pressure on top surface")
    println("  Magnitude: $(pressure / 1e6) MPa")

    # Step 6: Create physics
    physics = ElasticityPhysics(
        mesh,
        material,
        fixed_nodes,
        pressure_nodes,
        pressure
    )

    # Step 7: Solve on GPU
    println("\n" * "="^70)
    println("Solving on GPU...")
    println("="^70)

    u = solve_elasticity_gpu(physics, tol=1e-6, max_iter=1000)

    # Step 8: Post-process results
    println("\n" * "="^70)
    println("Results:")
    println("="^70)

    n_nodes = size(mesh.nodes, 2)

    # Displacement magnitudes
    disp_mag = zeros(n_nodes)
    for i in 1:n_nodes
        ux = u[3*i-2]
        uy = u[3*i-1]
        uz = u[3*i]
        disp_mag[i] = sqrt(ux^2 + uy^2 + uz^2)
    end

    println("\nDisplacement statistics:")
    @printf("  Max: %.6e m\n", maximum(disp_mag))
    @printf("  Min: %.6e m\n", minimum(disp_mag))
    @printf("  Avg: %.6e m\n", sum(disp_mag) / n_nodes)

    # Find node with max displacement
    max_node = argmax(disp_mag)
    x_max = mesh.nodes[1, max_node]
    y_max = mesh.nodes[2, max_node]
    z_max = mesh.nodes[3, max_node]

    println("\nMax displacement location:")
    @printf("  Node: %d\n", max_node)
    @printf("  Position: (%.3f, %.3f, %.3f)\n", x_max, y_max, z_max)
    @printf("  Displacement: (%.6e, %.6e, %.6e) m\n",
        u[3*max_node-2], u[3*max_node-1], u[3*max_node])

    # Analytical comparison
    L = 10.0
    width = 1.0
    height = 1.0
    I = width * height^3 / 12
    q = pressure * width
    w_analytical = q * L^4 / (8 * material.E * I)

    println("\nComparison with beam theory:")
    @printf("  Analytical: %.6e m\n", w_analytical)
    @printf("  FEM:        %.6e m\n", maximum(disp_mag))
    @printf("  Error:      %.2f%%\n",
        abs(w_analytical - maximum(disp_mag)) / w_analytical * 100)

    println("\n" * "="^70)
    println("✅ Demo complete!")
    println("="^70)
    println("\nNext steps:")
    println("  - Visualize with ParaView (export to VTK)")
    println("  - Try different mesh sizes")
    println("  - Add more complex loading")
    println("  - Test with nonlinear materials")
    println("="^70 * "\n")
end

main()
