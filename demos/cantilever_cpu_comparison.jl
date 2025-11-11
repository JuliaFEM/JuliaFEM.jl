# Cantilever Beam - Assembly Strategy Benchmark (RESEARCH/DEVELOPMENT)
#
# ⚠️  NOTE: This is a LOW-LEVEL benchmark for algorithm research!
# ⚠️  For USER-FACING examples, see:
# ⚠️    - demos/assembly_comparison_simple.jl (uses real Problem API)
# ⚠️    - demos/cantilever_gmsh_gpu.jl (uses Physics API + GPU)
# ⚠️    - examples/linear_static.jl (complete workflow)
#
# This file compares three assembly/solver combinations at the structure level:
# 1. Element assembly + Direct solver (baseline)
# 2. Element assembly + Iterative CG
# 3. Nodal assembly + Matrix-free CG (research)
#
# Uses element_assembly_structures.jl and nodal_assembly_structures.jl directly.
# Not intended as example of user-facing API!

using LinearAlgebra
using SparseArrays
using Tensors
using Printf

# Import our assembly structures
include("../src/element_assembly_structures.jl")
include("../src/nodal_assembly_structures.jl")

println("="^70)
println("Cantilever Beam - CPU Assembly Comparison")
println("="^70)

# ============================================================================
# 1. Generate Mesh with Gmsh
# ============================================================================

println("\n[1] Generating mesh with Gmsh...")

# Simple beam: L=10, W=1, H=1
# Target ~20 Tet4 elements

using Gmsh: gmsh

gmsh.initialize()
gmsh.model.add("cantilever")

# Geometry
lc = 1.5  # Characteristic length (controls mesh density)
L, W, H = 10.0, 1.0, 1.0

# Create box
box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
gmsh.model.occ.synchronize()

# Set mesh size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Extract nodes
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
n_nodes = length(node_tags)
nodes = reshape(node_coords, 3, n_nodes)

println("  Nodes: $n_nodes")

# Extract Tet4 elements (type 4)
elem_types, elem_tags_vec, elem_node_tags_vec = gmsh.model.mesh.getElements(3)
tet4_idx = findfirst(t -> t == 4, elem_types)  # Type 4 = Tet4

if tet4_idx === nothing
    error("No Tet4 elements found!")
end

elem_node_tags = elem_node_tags_vec[tet4_idx]
n_elements = div(length(elem_node_tags), 4)
connectivity = reshape(Int.(elem_node_tags), 4, n_elements)

println("  Elements: $n_elements")
println("  DOFs: $(3 * n_nodes)")

gmsh.finalize()

# ============================================================================
# 2. Material Properties and BCs
# ============================================================================

println("\n[2] Setting up problem...")

# Material (steel)
E = 210e9  # Young's modulus [Pa]
ν = 0.3    # Poisson's ratio

# Boundary conditions
# Fixed: nodes at X=0
fixed_nodes = findall(x -> abs(x) < 1e-10, nodes[1, :])
println("  Fixed nodes: $(length(fixed_nodes))")

# Loaded: nodes at X=L (free end)
loaded_nodes = findall(x -> abs(x - L) < 1e-10, nodes[1, :])
println("  Loaded nodes: $(length(loaded_nodes))")

# Applied force (total 1000 N downward, distributed)
F_total = -1000.0  # Negative Z direction
f_per_node = F_total / length(loaded_nodes)

println("  Force per node: $(f_per_node) N")

# ============================================================================
# 3. Compute Element Stiffness Matrices (Shared by all methods)
# ============================================================================
#
# NOTE: This is a simplified reference implementation for benchmarking.
# For production use, see src/problems_elasticity.jl which includes:
# - Geometric nonlinearity, finite strain
# - Plasticity and advanced material models
# - Surface tractions, body forces
# - Integration with Problem/Element API
#
# This demo focuses on assembly strategy comparison, not material complexity.
# ============================================================================

println("\n[3] Computing element stiffness matrices...")

# Elasticity tensor (isotropic)
λ = E * ν / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))

function compute_tet4_stiffness(X::Matrix{Float64}, E::Float64, ν::Float64)
    # X: 3×4 matrix of node coordinates
    # Returns: 12×12 element stiffness matrix

    # Shape function derivatives in parent element (constant for Tet4)
    dN_dξ = [-1.0 -1.0 -1.0;
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0]

    # Jacobian: J = dX/dξ
    J = X * dN_dξ  # 3×3
    detJ = det(J)

    if detJ <= 0
        error("Negative Jacobian determinant!")
    end

    # Shape function derivatives in physical space
    dN_dx = dN_dξ / J  # 4×3

    # B matrix (strain-displacement): 6×12
    B = zeros(6, 12)
    for i in 1:4
        B[1, 3i-2] = dN_dx[i, 1]      # ∂u/∂x
        B[2, 3i-1] = dN_dx[i, 2]      # ∂v/∂y
        B[3, 3i] = dN_dx[i, 3]      # ∂w/∂z
        B[4, 3i-2] = dN_dx[i, 2]      # ∂u/∂y
        B[4, 3i-1] = dN_dx[i, 1]      # ∂v/∂x
        B[5, 3i-1] = dN_dx[i, 3]      # ∂v/∂z
        B[5, 3i] = dN_dx[i, 2]      # ∂w/∂y
        B[6, 3i-2] = dN_dx[i, 3]      # ∂u/∂z
        B[6, 3i] = dN_dx[i, 1]      # ∂w/∂x
    end

    # Elasticity matrix (Voigt notation)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))

    D = [λ+2μ λ λ 0 0 0;
        λ λ+2μ λ 0 0 0;
        λ λ λ+2μ 0 0 0;
        0 0 0 μ 0 0;
        0 0 0 0 μ 0;
        0 0 0 0 0 μ]

    # Element stiffness: K_e = ∫ B^T D B dV = B^T D B * V
    # For Tet4: V = detJ / 6
    V = abs(detJ) / 6.0
    K_e = (B' * D * B) * V

    return K_e
end

# Compute all element matrices
K_elements = Vector{Matrix{Float64}}(undef, n_elements)
for e in 1:n_elements
    conn = connectivity[:, e]
    X_elem = nodes[:, conn]
    K_elements[e] = compute_tet4_stiffness(X_elem, E, ν)
end

println("  Element stiffness matrices computed")

# ============================================================================
# 4. METHOD 1: Element Assembly + Direct Solver
# ============================================================================

println("\n" * "="^70)
println("METHOD 1: Element Assembly + Direct Solver (LU)")
println("="^70)

t1 = time()

# Assemble global system
n_dofs = 3 * n_nodes
assembly = ElementAssemblyData(n_dofs, Float64)

for e in 1:n_elements
    conn = Tuple(connectivity[:, e])
    gdofs = get_dof_indices(conn, 3)

    contrib = ElementContribution(e, gdofs, K_elements[e],
        zeros(12), zeros(12))
    scatter_to_global!(assembly, contrib)
end

# Apply loads
for node in loaded_nodes
    dof_z = 3 * node  # Z component
    assembly.f_ext_global[dof_z] = f_per_node
end

# Compute residual
compute_residual!(assembly)

# Apply Dirichlet BCs
fixed_dofs = Int[]
for node in fixed_nodes
    append!(fixed_dofs, [3 * node - 2, 3 * node - 1, 3 * node])
end
apply_dirichlet_bc!(assembly, fixed_dofs, zeros(length(fixed_dofs)))

t_assembly_1 = time() - t1
println("Assembly time: $(round(t_assembly_1, digits=4)) s")

# Solve with direct solver (K_global is already CSC)
t_solve_1_start = time()
u1 = assembly.K_global \ assembly.r_global
t_solve_1 = time() - t_solve_1_start

# Compute final residual
r1 = matrix_vector_product(assembly, u1) - assembly.f_ext_global
r1_norm = norm(r1)

t_total_1 = time() - t1

println("Solve time: $(round(t_solve_1, digits=4)) s")
println("Total time: $(round(t_total_1, digits=4)) s")
println("Residual norm: $(r1_norm)")
println("Max displacement: $(maximum(abs.(u1)) * 1000) mm")

# ============================================================================
# 5. METHOD 2: Element Assembly + Iterative Solver (CG)
# ============================================================================

println("\n" * "="^70)
println("METHOD 2: Element Assembly + Iterative Solver (CG)")
println("="^70)

t2 = time()

# Reuse assembly from Method 1
t_assembly_2 = t_assembly_1  # Same assembly

# Conjugate Gradient solver
function cg_solve(A::ElementAssemblyData, b::Vector{Float64};
    tol=1e-8, max_iter=1000)
    n = length(b)
    x = zeros(n)
    r = b - matrix_vector_product(A, x)
    p = copy(r)
    rsold = dot(r, r)

    for iter in 1:max_iter
        Ap = matrix_vector_product(A, p)
        α = rsold / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)

        if sqrt(rsnew) < tol
            return x, iter, sqrt(rsnew)
        end

        β = rsnew / rsold
        p .= r .+ β .* p
        rsold = rsnew
    end

    return x, max_iter, sqrt(rsold)
end

t_solve_2_start = time()
u2, cg_iters_2, cg_res_2 = cg_solve(assembly, assembly.r_global, tol=1e-8)
t_solve_2 = time() - t_solve_2_start

# Compute final residual
r2 = matrix_vector_product(assembly, u2) - assembly.f_ext_global
r2_norm = norm(r2)

t_total_2 = time() - t2

println("Assembly time: $(round(t_assembly_2, digits=4)) s")
println("Solve time: $(round(t_solve_2, digits=4)) s")
println("Total time: $(round(t_total_2, digits=4)) s")
println("CG iterations: $cg_iters_2")
println("CG residual: $(cg_res_2)")
println("Residual norm: $(r2_norm)")
println("Max displacement: $(maximum(abs.(u2)) * 1000) mm")
println("Difference from Method 1: $(norm(u1 - u2))")

# ============================================================================
# 6. METHOD 3: Nodal Assembly + Iterative Solver (Matrix-Free CG)
# ============================================================================

println("\n" * "="^70)
println("METHOD 3: Nodal Assembly + Matrix-Free Iterative Solver")
println("="^70)

t3 = time()

# Build node-to-elements map (convert matrix to vector of tuples)
conn_tuples = [Tuple(connectivity[:, e]) for e in 1:n_elements]
node_map = NodeToElementsMap(conn_tuples)

# For each node, precompute 3×3 stiffness blocks with all coupling nodes
# This is the "spider" pattern

struct NodalAssemblyData
    node_map::NodeToElementsMap
    K_elements::Vector{Matrix{Float64}}
    connectivity::Matrix{Int}
    conn_tuples::Vector{NTuple{4,Int}}  # Store tuple version too
    n_nodes::Int
    n_dofs::Int
end

nodal_data = NodalAssemblyData(node_map, K_elements, connectivity,
    conn_tuples, n_nodes, n_dofs)

# Matrix-vector product using nodal assembly
function nodal_matvec!(w::Vector{Float64}, v::Vector{Float64},
    data::NodalAssemblyData)
    fill!(w, 0.0)

    for node_i in 1:data.n_nodes
        # Get spider nodes (all nodes coupled to node_i)
        spider = get_node_spider(data.node_map, node_i, data.conn_tuples)

        w_local = zeros(3)

        for node_j in spider
            # Sum contributions from all elements containing both nodes
            K_block_ij = zeros(3, 3)

            for elem_info in data.node_map.node_to_elements[node_i]
                elem_idx = elem_info.element_id
                conn = data.conn_tuples[elem_idx]

                # Check if node_j is in this element
                local_j = findfirst(==(node_j), conn)
                if local_j !== nothing
                    local_i = findfirst(==(node_i), conn)
                    K_e = data.K_elements[elem_idx]

                    # Extract 3×3 block
                    for α in 1:3, β in 1:3
                        K_block_ij[α, β] += K_e[3*(local_i-1)+α, 3*(local_j-1)+β]
                    end
                end
            end

            # Apply to displacement
            v_j = v[3*(node_j-1)+1:3*node_j]
            w_local .+= K_block_ij * v_j
        end

        # Write to global
        w[3*(node_i-1)+1:3*node_i] .= w_local
    end

    return w
end

t_assembly_3 = time() - t3
println("Nodal map construction: $(round(t_assembly_3, digits=4)) s")

# Build RHS (same as before)
f_ext = zeros(n_dofs)
for node in loaded_nodes
    dof_z = 3 * node
    f_ext[dof_z] = f_per_node
end

# CG with matrix-free matvec
function cg_solve_nodal(data::NodalAssemblyData, b::Vector{Float64},
    fixed_dofs::Vector{Int};
    tol=1e-8, max_iter=1000)
    n = length(b)
    x = zeros(n)

    # Apply BC to initial guess
    x[fixed_dofs] .= 0.0

    # Compute initial residual
    Ax = zeros(n)
    nodal_matvec!(Ax, x, data)
    Ax[fixed_dofs] .= 0.0  # Zero out fixed DOFs

    r = b - Ax
    r[fixed_dofs] .= 0.0
    p = copy(r)
    rsold = dot(r, r)

    for iter in 1:max_iter
        Ap = zeros(n)
        nodal_matvec!(Ap, p, data)
        Ap[fixed_dofs] .= 0.0

        α = rsold / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)

        if sqrt(rsnew) < tol
            return x, iter, sqrt(rsnew)
        end

        β = rsnew / rsold
        p .= r .+ β .* p
        rsold = rsnew
    end

    return x, max_iter, sqrt(rsold)
end

t_solve_3_start = time()
u3, cg_iters_3, cg_res_3 = cg_solve_nodal(nodal_data, f_ext, fixed_dofs,
    tol=1e-8)
t_solve_3 = time() - t_solve_3_start

# Compute final residual
w3 = zeros(n_dofs)
nodal_matvec!(w3, u3, nodal_data)
r3 = w3 - f_ext
r3_norm = norm(r3)

t_total_3 = time() - t3

println("Solve time: $(round(t_solve_3, digits=4)) s")
println("Total time: $(round(t_total_3, digits=4)) s")
println("CG iterations: $cg_iters_3")
println("CG residual: $(cg_res_3)")
println("Residual norm: $(r3_norm)")
println("Max displacement: $(maximum(abs.(u3)) * 1000) mm")
println("Difference from Method 1: $(norm(u1 - u3))")

# ============================================================================
# 7. Summary Comparison
# ============================================================================

println("\n" * "="^70)
println("SUMMARY COMPARISON")
println("="^70)

println("\nProblem Size:")
println("  Nodes: $n_nodes")
println("  Elements: $n_elements")
println("  DOFs: $n_dofs")
println("  Fixed DOFs: $(length(fixed_dofs))")
println("  Free DOFs: $(n_dofs - length(fixed_dofs))")

println("\n" * "-"^70)
println(@sprintf("%-40s %10s %10s %10s", "Method", "Assembly", "Solve", "Total"))
println("-"^70)
println(@sprintf("%-40s %9.4fs %9.4fs %9.4fs",
    "1. Element + Direct (LU)", t_assembly_1, t_solve_1, t_total_1))
println(@sprintf("%-40s %9.4fs %9.4fs %9.4fs",
    "2. Element + Iterative (CG, $cg_iters_2 iter)",
    t_assembly_2, t_solve_2, t_total_2))
println(@sprintf("%-40s %9.4fs %9.4fs %9.4fs",
    "3. Nodal + Iterative (CG, $cg_iters_3 iter)",
    t_assembly_3, t_solve_3, t_total_3))
println("-"^70)

println("\nAccuracy (vs Method 1):")
println(@sprintf("  Method 2 error: %.3e", norm(u1 - u2)))
println(@sprintf("  Method 3 error: %.3e", norm(u1 - u3)))

println("\nSpeedup vs Method 1:")
println(@sprintf("  Method 2: %.2fx", t_total_1 / t_total_2))
println(@sprintf("  Method 3: %.2fx", t_total_1 / t_total_3))

println("\nMax Displacement:")
println(@sprintf("  Method 1: %.6f mm", maximum(abs.(u1)) * 1000))
println(@sprintf("  Method 2: %.6f mm", maximum(abs.(u2)) * 1000))
println(@sprintf("  Method 3: %.6f mm", maximum(abs.(u3)) * 1000))

println("\n" * "="^70)
println("All methods complete!")
println("="^70)
