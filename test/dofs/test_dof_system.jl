using JuliaFEM
using Tensors

# Create simple 2-element mesh (2 hexes sharing a face)
# Hex 1: nodes 1-8
# Hex 2: nodes 5-8, 9-12 (shares face 5-6-7-8 with Hex 1)

nodes = [
    Vec{3}((0.0, 0.0, 0.0)),  # 1
    Vec{3}((1.0, 0.0, 0.0)),  # 2
    Vec{3}((1.0, 1.0, 0.0)),  # 3
    Vec{3}((0.0, 1.0, 0.0)),  # 4
    Vec{3}((0.0, 0.0, 1.0)),  # 5
    Vec{3}((1.0, 0.0, 1.0)),  # 6
    Vec{3}((1.0, 1.0, 1.0)),  # 7
    Vec{3}((0.0, 1.0, 1.0)),  # 8
    Vec{3}((0.0, 0.0, 2.0)),  # 9
    Vec{3}((1.0, 0.0, 2.0)),  # 10
    Vec{3}((1.0, 1.0, 2.0)),  # 11
    Vec{3}((0.0, 1.0, 2.0)),  # 12
]

connectivity = [
    NTuple{8,UInt32}((1, 2, 3, 4, 5, 6, 7, 8)),
    NTuple{8,UInt32}((5, 6, 7, 8, 9, 10, 11, 12)),
]

element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32[1, 2]))

mesh = Mesh{8,Hexahedron{8}}(nodes, connectivity, element_sets)

println("Mesh created:")
println("  Nodes: ", length(nodes))
println("  Elements: ", length(connectivity))

# Create elements with 3D displacement DOFs (Vec{3} at vertices)
println("\nCreating elements with DOF{Vec{3}, Vertex}...")

ElemType = Element{Hexahedron{8}, Lagrange{1}, DOF{Vec{3}, Vertex}}
elements, mgr = create_elements!(mesh, ElemType)

println("Elements created:")
println("  Element count: ", length(elements))
println("  Total DOFs: ", mgr.total_dofs)
println("  Expected DOFs: ", 3 * length(nodes), " (3 per node)")

# Check DOF indices for first element
println("\nElement 1 DOF indices:")
println("  ", elements[1].dof_indices)
println("  Length: ", length(elements[1].dof_indices))
println("  Expected: 24 (8 nodes × 3 DOFs)")

# Check DOF indices for second element (should reuse nodes 5-8)
println("\nElement 2 DOF indices:")
println("  ", elements[2].dof_indices)
println("  Length: ", length(elements[2].dof_indices))
println("  Expected: 24 (8 nodes × 3 DOFs)")

# Verify node-to-DOF mapping
println("\nNode-to-DOF mapping:")
for node_id in [1, 5, 9]
    dofs = get_node_dofs(mgr, node_id)
    println("  Node $node_id: DOFs $dofs")
end

println("\n✓ DOF system test complete!")
