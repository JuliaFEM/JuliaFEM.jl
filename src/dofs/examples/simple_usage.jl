# Simple Example: DOF Extraction Usage
# ======================================

using JuliaFEM
using StaticArrays
using Tensors

println("="^80)
println("Simple DOF Extraction Example")
println("="^80)
println()

# Create a mock global DOF vector (e.g., from FEM solve)
n_dofs = 100
u_global = rand(n_dofs)

println("Global DOF vector size: $n_dofs")
println()

# ============================================================================
# Example 1: Extract 3D displacement for Tet4 element
# ============================================================================

println("Example 1: Tet4 with 3D displacement")
println("-"^80)

# Tet4 has 4 nodes, each with 3 DOFs (ux, uy, uz) = 12 DOFs total
# Suppose this element's DOFs are at indices:
dof_indices_tet4 = (5, 6, 7,      # Node 1: ux, uy, uz
                    20, 21, 22,    # Node 2: ux, uy, uz
                    35, 36, 37,    # Node 3: ux, uy, uz
                    50, 51, 52)    # Node 4: ux, uy, uz

# Extract DOFs for this element
u_elem = extract_element_dofs(VectorDOF{3}, u_global, dof_indices_tet4)

println("Extracted DOFs type: $(typeof(u_elem))")
println("Number of nodes: $(length(u_elem))")
println()

println("Node displacements:")
for (i, u_node) in enumerate(u_elem)
    println("  Node $i: ux=$(round(u_node[1], digits=4)), " *
            "uy=$(round(u_node[2], digits=4)), " *
            "uz=$(round(u_node[3], digits=4))")
end
println()

# Access individual components
println("Accessing components:")
println("  Node 1, x-displacement: $(u_elem[1][1])")
println("  Node 2, y-displacement: $(u_elem[2][2])")
println("  Node 4, z-displacement: $(u_elem[4][3])")
println()

# ============================================================================
# Example 2: Extract 2D displacement for Tri3 element
# ============================================================================

println("Example 2: Tri3 with 2D displacement")
println("-"^80)

# Tri3 has 3 nodes, each with 2 DOFs (ux, uy) = 6 DOFs total
dof_indices_tri3 = (10, 11,    # Node 1: ux, uy
                    25, 26,    # Node 2: ux, uy
                    40, 41)    # Node 3: ux, uy

u_elem_2d = extract_element_dofs(VectorDOF{2}, u_global, dof_indices_tri3)

println("Extracted DOFs type: $(typeof(u_elem_2d))")
println()

println("Node displacements:")
for (i, u_node) in enumerate(u_elem_2d)
    println("  Node $i: ux=$(round(u_node[1], digits=4)), " *
            "uy=$(round(u_node[2], digits=4))")
end
println()

# ============================================================================
# Example 3: Extract scalar field (e.g., temperature)
# ============================================================================

println("Example 3: Tet4 with temperature field")
println("-"^80)

# Tet4 has 4 nodes, each with 1 DOF (temperature) = 4 DOFs total
dof_indices_temp = (15, 30, 45, 60)

T_elem = extract_element_dofs(ScalarDOF, u_global, dof_indices_temp)

println("Extracted DOFs type: $(typeof(T_elem))")
println()

println("Node temperatures:")
for (i, T_node) in enumerate(T_elem)
    println("  Node $i: T=$(round(T_node, digits=4))")
end
println()

# ============================================================================
# Example 4: Using in element assembly
# ============================================================================

println("Example 4: Usage in element stiffness assembly")
println("-"^80)
println()

println("""
# Typical usage in FEM assembly loop:

for elem in elements
    # Extract current displacements for this element
    u_elem = extract_element_dofs(VectorDOF{3}, u_global, elem.dof_indices)
    
    # Compute element internal forces (uses u_elem)
    f_int_elem = compute_internal_forces(elem, u_elem)
    
    # Assemble to global vector
    assemble!(f_int_global, elem.dof_indices, f_int_elem)
end
""")

println("="^80)
println("Key Benefits:")
println("  • Type-safe: Compiler knows exact return type")
println("  • Zero-cost: Compiles to pure load operations")
println("  • Ergonomic: Natural Vec{3} access (u_node[1], u_node[2], u_node[3])")
println("  • Generic: Works for any D (1D, 2D, 3D, ...)")
println("="^80)
