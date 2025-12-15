"""
Demonstration: REAL Stokes Flow - Mixed Finite Elements

This demo implements COMPLETE Stokes physics:
1. Velocity DOFs (Vec{2} at vertices) 
2. Pressure DOFs (Float64 at cell centers)
3. REAL viscous term assembly (∫∇u:∇v dx)
4. REAL divergence operator (∫p(∇·v) dx)
5. Solving incompressible Stokes flow

Physics: -μΔu + ∇p = f, ∇·u = 0

Use case: Incompressible flow, MINI element, lid-driven cavity
"""

using JuliaFEM
using Test
using Tensors
using LinearAlgebra
using SparseArrays
using Printf

@testset "Real Stokes Flow: Mixed FEM" begin
    println("\n" * "="^60)
    println("REAL STOKES FLOW: Mixed Finite Elements")
    println("="^60)
    
    # Create mesh: Two triangles forming a square domain
    # Triangle 1: nodes (1, 2, 3)
    # Triangle 2: nodes (2, 4, 3)
    nodes = [
        Vec{3,Float64}((0.0, 0.0, 0.0)),  # Node 1 (bottom-left)
        Vec{3,Float64}((1.0, 0.0, 0.0)),  # Node 2 (bottom-right)
        Vec{3,Float64}((0.0, 1.0, 0.0)),  # Node 3 (top-left)
        Vec{3,Float64}((1.0, 1.0, 0.0)),  # Node 4 (top-right)
    ]
    connectivity = [
        (UInt32(1), UInt32(2), UInt32(3)),  # Triangle 1
        (UInt32(2), UInt32(4), UInt32(3)),  # Triangle 2
    ]
    mesh = Mesh{Triangle{3}}(nodes, connectivity)
    
    println("\nMesh: 2 triangles forming unit square [0,1]×[0,1]")
    println("  4 nodes, 5 edges")
    
    # Create velocity elements (Vec{2} at vertices)
    elements_u, dof_mgr_u = create_elements!(mesh, Element{Triangle{3}, Lagrange{1}, DOF{Vec{2}, Vertex}})
    
    println("\n✓ Created $(length(elements_u)) velocity elements")
    println("  DOF type: Vec{2} at Vertices")
    
    # Create pressure elements (Float64 at cell centers) - NOW WORKS!
    elements_p, dof_mgr_p = create_elements!(mesh, Element{Triangle{3}, Lagrange{1}, DOF{Float64, Cell}})
    
    println("✓ Created $(length(elements_p)) pressure elements")
    println("  DOF type: Float64 at Cell centers")
    
    # Count DOFs
    n_u_dofs = dof_mgr_u.total_dofs
    n_p_dofs = dof_mgr_p.total_dofs
    n_total = n_u_dofs + n_p_dofs
    
    println("\nDOF counts:")
    println("  Velocity DOFs: $n_u_dofs (4 nodes × 2 components)")
    println("  Pressure DOFs: $n_p_dofs (2 cells × 1 scalar)")
    println("  Total DOFs: $n_total")
    
    @test n_u_dofs == 8
    @test n_p_dofs == 2
    
    # Check pressure DOFs are element-local (no sharing)
    p_dof_1 = elements_p[1].dof_indices[1]
    p_dof_2 = elements_p[2].dof_indices[1]
    @test p_dof_1 != p_dof_2
    println("\n✓ Pressure DOFs are element-local (no sharing)")
    
    println("\n" * "="^60)
    println("Assembling REAL Stokes system...")
    println("  Physics: -μΔu + ∇p = f, ∇·u = 0")
    println("="^60)
    
    # Viscosity
    μ = 1.0
    
    # Initialize matrices
    A = spzeros(Float64, n_u_dofs, n_u_dofs)  # Viscous term: ∫μ∇u:∇v
    B = spzeros(Float64, n_p_dofs, n_u_dofs)  # Divergence: ∫p(∇·v)
    f = zeros(Float64, n_u_dofs)              # Body force
    
    # Assemble REAL viscous term for each triangle
    for (elem_idx, elem_u) in enumerate(elements_u)
        # Get triangle nodes
        conn = mesh.connectivity[elem_idx]
        X = [nodes[i] for i in conn]
        
        # Extract 2D coordinates
        x1, y1 = X[1][1], X[1][2]
        x2, y2 = X[2][1], X[2][2]
        x3, y3 = X[3][1], X[3][2]
        
        # Element area
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        
        # Gradients of shape functions (constant per element)
        # ∇N₁ = (1/2A) * [y₂-y₃, x₃-x₂]
        # ∇N₂ = (1/2A) * [y₃-y₁, x₁-x₃]
        # ∇N₃ = (1/2A) * [y₁-y₂, x₂-x₁]
        inv_2A = 1.0 / (2.0 * area)
        
        ∇N = [
            inv_2A * Vec{2,Float64}((y2 - y3, x3 - x2)),
            inv_2A * Vec{2,Float64}((y3 - y1, x1 - x3)),
            inv_2A * Vec{2,Float64}((y1 - y2, x2 - x1))
        ]
        
        # For vector-valued FEM with Vec{2} DOFs:
        # u = [ux₁, uy₁, ux₂, uy₂, ux₃, uy₃]
        # 
        # Viscous stiffness: ∫∇u:∇v dx
        # Since ∇N are constant: K_ij = μ * (∇Nᵢ ⋅ ∇Nⱼ) * area
        # 
        # For Vec{2}: each node has 2 components
        # A_local is 6×6 with 2×2 blocks: A_ij = μ * (∇Nᵢ ⋅ ∇Nⱼ) * area * I₂
        
        A_local = zeros(6, 6)
        
        for i in 1:3  # Test function node
            for j in 1:3  # Trial function node
                # Scalar stiffness contribution
                k_scalar = μ * (∇N[i] ⋅ ∇N[j]) * area
                
                # Fill 2×2 block (diagonal for isotropic viscosity)
                # DOFs: (2i-1, 2i) for node i
                idx_i = [2*i-1, 2*i]
                idx_j = [2*j-1, 2*j]
                
                A_local[idx_i[1], idx_j[1]] += k_scalar  # ux-ux
                A_local[idx_i[2], idx_j[2]] += k_scalar  # uy-uy
            end
        end
        
        # Body force (e.g., gravity in y-direction)
        f_body = Vec{2,Float64}((0.0, -1.0))
        
        # ∫f·v dx = (area/3) * f_body (constant per node for P1)
        f_local = zeros(6)
        for i in 1:3
            f_local[2*i-1] = (area/3) * f_body[1]  # fx
            f_local[2*i] = (area/3) * f_body[2]    # fy
        end
        
        # Scatter velocity stiffness to global
        for i in 1:6
            I = elem_u.dof_indices[i]
            f[I] += f_local[i]
            
            for j in 1:6
                J = elem_u.dof_indices[j]
                A[I, J] += A_local[i, j]
            end
        end
    end
    
    println("  ✓ Viscous term assembled: ∫μ∇u:∇v dx")
    println("    Matrix A size: $(size(A))")
    println("    Non-zeros: $(nnz(A))")
    
    # Assemble REAL divergence operator
    for (elem_idx, elem_p) in enumerate(elements_p)
        elem_u = elements_u[elem_idx]
        
        # Get triangle nodes
        conn = mesh.connectivity[elem_idx]
        X = [nodes[i] for i in conn]
        
        # Extract 2D coordinates
        x1, y1 = X[1][1], X[1][2]
        x2, y2 = X[2][1], X[2][2]
        x3, y3 = X[3][1], X[3][2]
        
        # Element area
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        
        # Gradients of shape functions
        inv_2A = 1.0 / (2.0 * area)
        
        ∇N = [
            inv_2A * Vec{2,Float64}((y2 - y3, x3 - x2)),
            inv_2A * Vec{2,Float64}((y3 - y1, x1 - x3)),
            inv_2A * Vec{2,Float64}((y1 - y2, x2 - x1))
        ]
        
        # Divergence operator: B_ki = ∫p_k(∇·v_i) dx
        # For P0 pressure (constant): p_k = 1 on element
        # ∇·v = ∂vx/∂x + ∂vy/∂y
        # 
        # For Vec{2} DOFs: v = Σᵢ [vxᵢ*Nᵢ, vyᵢ*Nᵢ]
        # ∇·v = Σᵢ [vxᵢ*∂Nᵢ/∂x + vyᵢ*∂Nᵢ/∂y]
        #     = Σᵢ [vxᵢ*(∇Nᵢ)ₓ + vyᵢ*(∇Nᵢ)ᵧ]
        # 
        # B_ki = ∫(∇Nᵢ)ₓ dx = area * (∇Nᵢ)ₓ  [for vxᵢ component]
        # B_ki = ∫(∇Nᵢ)ᵧ dx = area * (∇Nᵢ)ᵧ  [for vyᵢ component]
        
        B_local = zeros(6)  # 6 velocity DOFs per element
        
        for i in 1:3  # Node
            # Divergence contributions
            B_local[2*i-1] = area * ∇N[i][1]  # ∂vx/∂x term
            B_local[2*i]   = area * ∇N[i][2]  # ∂vy/∂y term
        end
        
        # Scatter to global B matrix
        p_dof = elem_p.dof_indices[1]
        
        for i in 1:6
            u_dof = elem_u.dof_indices[i]
            B[p_dof, u_dof] += B_local[i]
        end
    end
    
    println("  ✓ Divergence operator assembled: ∫p(∇·v) dx")
    println("    Matrix B size: $(size(B))")
    println("    Non-zeros: $(nnz(B))")
    
    # Build saddle-point system:
    # [ A   B^T ] [ u ]   [ f ]
    # [ B    0  ] [ p ] = [ 0 ]
    K = [A B'; B spzeros(n_p_dofs, n_p_dofs)]
    F = [f; zeros(n_p_dofs)]
    
    println("\n  Full system size: $(size(K))")
    println("  Saddle-point structure: [A B'; B 0]")
    
    # Apply boundary conditions: no-slip on bottom (y=0)
    # Fix nodes 1 and 2 (y=0)
    bc_nodes = [1, 2]
    bc_dofs = Int[]
    for node in bc_nodes
        push!(bc_dofs, 2*node-1)  # ux
        push!(bc_dofs, 2*node)    # uy
    end
    
    for dof in bc_dofs
        K[dof, :] .= 0.0
        K[:, dof] .= 0.0
        K[dof, dof] = 1.0
        F[dof] = 0.0
    end
    
    println("\n  Applied BC: No-slip on bottom (nodes 1, 2)")
    println("    u(0,0) = (0,0), u(1,0) = (0,0)")
    
    # Solve
    println("\n  Solving...")
    sol = K \ F
    
    u_sol = sol[1:n_u_dofs]
    p_sol = sol[n_u_dofs+1:end]
    
    println("\n" * "="^60)
    println("SOLUTION:")
    println("="^60)
    
    println("\nVelocity field:")
    for node_id in 1:4
        ux = u_sol[2*node_id-1]
        uy = u_sol[2*node_id]
        x, y = nodes[node_id][1], nodes[node_id][2]
        println("  Node $node_id at ($x,$y): u = ($(@sprintf("%.4f", ux)), $(@sprintf("%.4f", uy)))")
    end
    
    println("\nPressure field:")
    for elem_id in 1:2
        p = p_sol[elem_id]
        println("  Element $elem_id: p = $(@sprintf("%.4f", p))")
    end
    
    # Verify solution properties
    @test all(isfinite.(sol))
    
    # Check BCs enforced
    @test abs(u_sol[1]) < 1e-10  # ux at node 1
    @test abs(u_sol[2]) < 1e-10  # uy at node 1
    @test abs(u_sol[3]) < 1e-10  # ux at node 2
    @test abs(u_sol[4]) < 1e-10  # uy at node 2
    
    # Check incompressibility: B*u ≈ 0
    div_residual = B * u_sol
    println("\nIncompressibility check:")
    println("  ∇·u residual: [$(@sprintf("%.6e", div_residual[1])), $(@sprintf("%.6e", div_residual[2]))]")
    @test norm(div_residual) < 1e-10
    
    println("\n✓ REAL Stokes flow solved successfully!")
    println("✓ Incompressibility ∇·u = 0 satisfied to machine precision!")
    
    println("\n" * "="^60)
    println("Key achievements:")
    println("="^60)
    println("  ✅ Cell-entity DOFs IMPLEMENTED!")
    println("  ✅ Mixed DOF types working (Vec{2} + Float64)")
    println("  ✅ REAL viscous stiffness: ∫μ∇u:∇v dx")
    println("  ✅ REAL divergence operator: ∫p(∇·v) dx")
    println("  ✅ Saddle-point system solved")
    println("  ✅ Incompressible Stokes flow")
    println("  ✅ Use case: Microfluidics, creeping flow")
    println("="^60)
end
