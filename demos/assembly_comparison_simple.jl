# Assembly Strategy Comparison - Simple Example
#
# Demonstrates the modern Physics API for solving elasticity problems.
# Uses the CPU backend with element assembly.

using JuliaFEM
using LinearAlgebra
using Printf

println("="^70)
println("Assembly Comparison - Modern Physics API")
println("="^70)

# ============================================================================
# 1. Create Simple Mesh
# ============================================================================

println("\n[1] Creating mesh...")

# Simple 2-element beam (Hex8 elements)
nodes = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [2.0, 0.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [1.0, 1.0, 0.0],
    6 => [2.0, 1.0, 0.0],
    7 => [0.0, 0.0, 1.0],
    8 => [1.0, 0.0, 1.0],
    9 => [2.0, 0.0, 1.0],
    10 => [0.0, 1.0, 1.0],
    11 => [1.0, 1.0, 1.0],
    12 => [2.0, 1.0, 1.0]
)

connectivity_hex = [
    (1, 2, 5, 4, 7, 8, 11, 10),
    (2, 3, 6, 5, 8, 9, 12, 11)
]

n_nodes = length(nodes)
n_elements = length(connectivity_hex)
n_dofs = 3 * n_nodes

println("  Nodes: $n_nodes")
println("  Elements: $n_elements")
println("  DOFs: $n_dofs")

# ============================================================================
# 2. Create Physics Problem
# ============================================================================

println("\n[2] Creating physics problem...")

physics = Physics(Elasticity, "simple beam", 3)
physics.properties.formulation = :continuum
physics.properties.finite_strain = false

# Create elements with new immutable API
elements = Element[]
for conn in connectivity_hex
    # Extract node coordinates
    X = [nodes[i] for i in conn]

    # Create immutable element with all fields
    element = Element(Hex8, conn,
        fields=(geometry=X,
            youngs_modulus=210e9,  # Steel
            poissons_ratio=0.3))

    push!(elements, element)
end

add_elements!(physics, elements)

println("  Elements added: $(length(physics.body_elements))")

# ============================================================================
# 3. Apply Boundary Conditions
# ============================================================================

println("\n[3] Applying boundary conditions...")

# Fix left end (nodes 1, 4, 7, 10)
fixed_nodes = [1, 4, 7, 10]
add_dirichlet!(physics, fixed_nodes, [1, 2, 3], 0.0)

println("  Fixed nodes: $(length(fixed_nodes)) (all DOFs)")
println("  Total Dirichlet BCs: $(length(physics.bc_dirichlet.node_ids))")

# Note: External forces would be applied via Neumann BC or body forces
# For this simple demo, we solve with zero external loading

# ============================================================================
# 4. Solve with CPU Backend
# ============================================================================

println("\n[4] Solving with CPU backend...")

t_solve = @elapsed begin
    sol = solve!(physics; backend=CPU(), tol=1e-6, max_iter=1000)
end

println("  Solve time: $(round(t_solve * 1000, digits=2)) ms")
println("  CG iterations: $(sol.cg_iterations)")
println("  Newton iterations: $(sol.newton_iterations)")
println("  Residual: $(sol.residual)")
println("  Max displacement: $(maximum(abs.(sol.u)) * 1000) mm")

# ============================================================================
# 5. Summary
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("\nProblem:")
println("  Nodes: $n_nodes")
println("  Elements: $n_elements")
println("  DOFs: $n_dofs")
println("  Fixed DOFs: $(3 * length(fixed_nodes))")
println("\nSolution:")
println("  Backend: CPU (element assembly + CG)")
println("  Solve time: $(round(t_solve * 1000, digits=2)) ms")
println("  CG iterations: $(sol.cg_iterations)")
println("  Newton iterations: $(sol.newton_iterations)")
println("  Residual: $(sol.residual)")
println("  Max displacement: $(maximum(abs.(sol.u)) * 1000) mm")
println("\n" * "="^70)
println("✓ Modern Physics API working on CPU!")
println("="^70)
