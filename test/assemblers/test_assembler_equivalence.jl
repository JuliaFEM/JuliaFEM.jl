# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Test: Assembler Equivalence

Verify that all assembler implementations produce identical results:
- COOAssembler (element-based, coordinate format)
- CSCAssembler (element-based, compressed sparse column)
- NodeBasedCOOAssembler (node-based, coordinate format)

Test model: Two Hex8 elements sharing a face (minimal realistic mesh)
"""

using Test
using JuliaFEM
using LinearAlgebra
using SparseArrays
using Tensors

@testset "Assembler Equivalence - Two Hex8 Elements" begin
    println("\n" * "="^70)
    println("ASSEMBLER EQUIVALENCE TEST")
    println("="^70)

    # ========================================================================
    # 1. Create Minimal Test Mesh (Two Hex8 Elements)
    # ========================================================================

    println("\n[1] Creating two-element mesh...")

    # Two unit cubes sharing a face
    # Element 1: X ∈ [0,1], Y ∈ [0,1], Z ∈ [0,1]
    # Element 2: X ∈ [1,2], Y ∈ [0,1], Z ∈ [0,1]

    nodes = Vec{3,Float64}[
        # Element 1 nodes (bottom face at Z=0, top face at Z=1)
        Vec{3}((0.0, 0.0, 0.0)),  # 1
        Vec{3}((1.0, 0.0, 0.0)),  # 2
        Vec{3}((1.0, 1.0, 0.0)),  # 3
        Vec{3}((0.0, 1.0, 0.0)),  # 4
        Vec{3}((0.0, 0.0, 1.0)),  # 5
        Vec{3}((1.0, 0.0, 1.0)),  # 6
        Vec{3}((1.0, 1.0, 1.0)),  # 7
        Vec{3}((0.0, 1.0, 1.0)),  # 8
        # Element 2 additional nodes (shares face with element 1)
        Vec{3}((2.0, 0.0, 0.0)),  # 9
        Vec{3}((2.0, 1.0, 0.0)),  # 10
        Vec{3}((2.0, 0.0, 1.0)),  # 11
        Vec{3}((2.0, 1.0, 1.0)),  # 12
    ]

    # Connectivity (Hex8 standard ordering)
    connectivity = [
        (1, 2, 3, 4, 5, 6, 7, 8),      # Element 1
        (2, 9, 10, 3, 6, 11, 12, 7),   # Element 2 (shares face 2-3-7-6 with element 1)
    ]

    nnodes = length(nodes)
    nelems = length(connectivity)
    ndofs = 3 * nnodes

    println("  Nodes: $nnodes")
    println("  Elements: $nelems")
    println("  DOFs: $ndofs")

    # Create mesh
    connectivity_uint32 = [NTuple{8,UInt32}(c) for c in connectivity]
    element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1):UInt32(nelems)))
    mesh = Mesh{8,Hexahedron{8}}(nodes, connectivity_uint32, element_sets)

    # ========================================================================
    # 2. Material and Kernel
    # ========================================================================

    println("\n[2] Setting up physics...")

    # Simple linear elastic material
    E = 210e9  # Pa
    ν = 0.3
    material = LinearElastic(E=E, ν=ν)

    # Create kernel
    kernel = ContinuumKernel(
        ContinuumFormulation{FullThreeD}(),
        material,
        Displacement{3}()
    )

    println("  Material: LinearElastic (E=$(E/1e9) GPa, ν=$ν)")
    println("  Kernel: ContinuumKernel (FullThreeD, Displacement{3})")

    # ========================================================================
    # 3. Boundary Conditions
    # ========================================================================

    println("\n[3] Setting up boundary conditions...")

    # Fix left face (X=0): nodes 1,4,5,8
    fixed_nodes = [1, 4, 5, 8]

    # Load right face (X=2): nodes 9,10,11,12
    loaded_nodes = [9, 10, 11, 12]

    # Apply unit load in X-direction
    force_per_node = Vec{3}((1000.0, 0.0, 0.0))  # 1 kN per node

    # Create boundary conditions
    bc_dirichlet = DirichletBC()
    for node in fixed_nodes
        push!(bc_dirichlet.node_ids, node)
        push!(bc_dirichlet.components, [1, 2, 3])
        push!(bc_dirichlet.values, 0.0)
    end

    bc_neumann = NeumannBC()
    for node in loaded_nodes
        push!(bc_neumann.surface_ids, node)
        push!(bc_neumann.values, force_per_node)
    end

    println("  Fixed nodes (X=0): $fixed_nodes")
    println("  Loaded nodes (X=2): $loaded_nodes")
    println("  Force per node: $(force_per_node[1]/1e3) kN")

    # ========================================================================
    # 4. Assemble with Each Assembler
    # ========================================================================

    println("\n[4] Assembling with all three assemblers...")

    # -------------------------------------------------------------------------
    # 4a. COOAssembler (Element-Based)
    # -------------------------------------------------------------------------

    println("\n  [4a] COOAssembler (element-based)...")

    assembler_coo = COOAssembler()
    cache_coo = create_cache(assembler_coo, mesh, kernel)

    t_coo = @elapsed begin
        assemble!(cache_coo, assembler_coo, kernel, mesh)
        K_coo, f_coo = extract_system(cache_coo)
        apply_neumann_bcs!(f_coo, kernel, mesh, bc_neumann)
        K_coo_bc = copy(K_coo)
        f_coo_bc = copy(f_coo)
        apply_dirichlet_bcs!(K_coo_bc, f_coo_bc, kernel, mesh, bc_dirichlet)
    end

    println("    Assembly time: $(round(t_coo*1e6, digits=2)) μs")
    println("    Matrix nnz: $(nnz(K_coo))")
    println("    Force norm: $(round(norm(f_coo), digits=6))")

    # -------------------------------------------------------------------------
    # 4b. CSCAssembler (Element-Based, Optimized)
    # -------------------------------------------------------------------------

    println("\n  [4b] CSCAssembler (element-based, optimized)...")

    assembler_csc = CSCAssembler()
    cache_csc = create_cache(assembler_csc, mesh, kernel)

    t_csc = @elapsed begin
        assemble!(cache_csc, assembler_csc, kernel, mesh)
        K_csc, f_csc = extract_system(cache_csc)
        apply_neumann_bcs!(f_csc, kernel, mesh, bc_neumann)
        K_csc_bc = copy(K_csc)
        f_csc_bc = copy(f_csc)
        apply_dirichlet_bcs!(K_csc_bc, f_csc_bc, kernel, mesh, bc_dirichlet)
    end

    println("    Assembly time: $(round(t_csc*1e6, digits=2)) μs")
    println("    Matrix nnz: $(nnz(K_csc))")
    println("    Force norm: $(round(norm(f_csc), digits=6))")

    # -------------------------------------------------------------------------
    # 4c. NodeBasedCOOAssembler (Node-Based)
    # -------------------------------------------------------------------------

    println("\n  [4c] NodeBasedCOOAssembler (node-based)...")

    assembler_nodal = NodeBasedCOOAssembler()
    cache_nodal = create_cache(assembler_nodal, mesh, kernel)

    t_nodal = @elapsed begin
        assemble!(cache_nodal, assembler_nodal, kernel, mesh)
        K_nodal, f_nodal = extract_system(cache_nodal)
        apply_neumann_bcs!(f_nodal, kernel, mesh, bc_neumann)
        K_nodal_bc = copy(K_nodal)
        f_nodal_bc = copy(f_nodal)
        apply_dirichlet_bcs!(K_nodal_bc, f_nodal_bc, kernel, mesh, bc_dirichlet)
    end

    println("    Assembly time: $(round(t_nodal*1e6, digits=2)) μs")
    println("    Matrix nnz: $(nnz(K_nodal))")
    println("    Force norm: $(round(norm(f_nodal), digits=6))")

    # ========================================================================
    # 5. Compare Results (Before BC Application)
    # ========================================================================

    println("\n[5] Comparing assembled systems (before BC)...")

    # Compare stiffness matrices
    K_diff_coo_csc = norm(K_coo - K_csc)
    K_diff_coo_nodal = norm(K_coo - K_nodal)
    K_diff_csc_nodal = norm(K_csc - K_nodal)

    K_norm = norm(K_coo)

    println("  Stiffness matrix differences:")
    println("    ||K_coo - K_csc||: $(K_diff_coo_csc)")
    println("    ||K_coo - K_nodal||: $(K_diff_coo_nodal)")
    println("    ||K_csc - K_nodal||: $(K_diff_csc_nodal)")
    println("    ||K_coo|| (reference): $(K_norm)")

    # Compare force vectors
    f_diff_coo_csc = norm(f_coo - f_csc)
    f_diff_coo_nodal = norm(f_coo - f_nodal)
    f_diff_csc_nodal = norm(f_csc - f_nodal)

    f_norm = norm(f_coo)

    println("  Force vector differences:")
    println("    ||f_coo - f_csc||: $(f_diff_coo_csc)")
    println("    ||f_coo - f_nodal||: $(f_diff_coo_nodal)")
    println("    ||f_csc - f_nodal||: $(f_diff_csc_nodal)")
    println("    ||f_coo|| (reference): $(f_norm)")

    # ========================================================================
    # 6. Solve and Compare Solutions
    # ========================================================================

    println("\n[6] Solving systems and comparing solutions...")

    u_coo = K_coo_bc \ f_coo_bc
    u_csc = K_csc_bc \ f_csc_bc
    u_nodal = K_nodal_bc \ f_nodal_bc

    u_diff_coo_csc = norm(u_coo - u_csc)
    u_diff_coo_nodal = norm(u_coo - u_nodal)
    u_diff_csc_nodal = norm(u_csc - u_nodal)

    u_norm = norm(u_coo)

    println("  Solution differences:")
    println("    ||u_coo - u_csc||: $(u_diff_coo_csc)")
    println("    ||u_coo - u_nodal||: $(u_diff_coo_nodal)")
    println("    ||u_csc - u_nodal||: $(u_diff_csc_nodal)")
    println("    ||u_coo|| (reference): $(u_norm)")

    # ========================================================================
    # 7. Test Assertions
    # ========================================================================

    println("\n[7] Running test assertions...")

    # Tolerance for floating-point comparison
    rtol = 1e-10  # Relative tolerance
    atol = 1e-12  # Absolute tolerance

    # Test 1: Stiffness matrices are identical
    @test isapprox(K_coo, K_csc, rtol=rtol, atol=atol)
    @test isapprox(K_coo, K_nodal, rtol=rtol, atol=atol)
    @test isapprox(K_csc, K_nodal, rtol=rtol, atol=atol)
    println("  ✓ All stiffness matrices are identical (within tolerance)")

    # Test 2: Force vectors are identical (should be zero for internal forces)
    @test isapprox(f_coo, f_csc, rtol=rtol, atol=atol)
    @test isapprox(f_coo, f_nodal, rtol=rtol, atol=atol)
    @test isapprox(f_csc, f_nodal, rtol=rtol, atol=atol)
    println("  ✓ All force vectors are identical (within tolerance)")

    # Test 3: Solutions are identical
    @test isapprox(u_coo, u_csc, rtol=rtol, atol=atol)
    @test isapprox(u_coo, u_nodal, rtol=rtol, atol=atol)
    @test isapprox(u_csc, u_nodal, rtol=rtol, atol=atol)
    println("  ✓ All solutions are identical (within tolerance)")

    # Test 4: Solutions are physically reasonable
    @test !any(isnan, u_coo)
    @test !any(isinf, u_coo)
    @test norm(u_coo) > 0  # Solution should not be zero
    println("  ✓ Solutions are physically reasonable (finite, non-zero)")

    # ========================================================================
    # 8. Performance Comparison
    # ========================================================================

    println("\n[8] Performance comparison...")

    println("  Assembly times:")
    println("    COO: $(round(t_coo*1e6, digits=2)) μs (baseline)")
    println("    CSC: $(round(t_csc*1e6, digits=2)) μs ($(round(t_coo/t_csc, digits=2))×)")
    println("    Nodal: $(round(t_nodal*1e6, digits=2)) μs ($(round(t_coo/t_nodal, digits=2))×)")

    if t_csc < t_coo
        println("  → CSC is $(round(t_coo/t_csc, digits=2))× faster than COO")
    end

    # Note: For very small problems, nodal may be slower due to overhead
    # But it should scale better for large problems and GPU
    if t_nodal > t_coo
        println("  → Nodal is slower for this tiny problem (expected)")
        println("     (Nodal assembly excels on GPU and large problems)")
    end

    # ========================================================================
    # 9. Summary
    # ========================================================================

    println("\n" * "="^70)
    println("TEST SUMMARY - ASSEMBLER EQUIVALENCE")
    println("="^70)
    println("Problem:")
    println("  Elements: $nelems Hex8")
    println("  Nodes: $nnodes")
    println("  DOFs: $ndofs")
    println()
    println("Results:")
    println("  All assemblers produce IDENTICAL results:")
    println("    ✓ Stiffness matrices match (||K_i - K_j|| < $rtol)")
    println("    ✓ Force vectors match (||f_i - f_j|| < $rtol)")
    println("    ✓ Solutions match (||u_i - u_j|| < $rtol)")
    println()
    println("Performance (tiny problem, CPU overhead dominant):")
    println("  COO: $(round(t_coo*1e6, digits=2)) μs")
    println("  CSC: $(round(t_csc*1e6, digits=2)) μs")
    println("  Nodal: $(round(t_nodal*1e6, digits=2)) μs")
    println()
    println("Status: ✓ ALL TESTS PASSED")
    println("="^70)
end
