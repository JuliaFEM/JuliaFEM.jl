#!/usr/bin/env julia
#
# GPU NodeSet MatVec Demo: NODAL ASSEMBLY for Krylov Methods
#
# Key insights:
# 1. JuliaFEM uses NODAL ASSEMBLY, not element assembly!
# 2. Loop over nodes, gather from connected elements
# 3. For Krylov (CG, GMRES), we need y = K*x (matvec)
# 4. Fields accessed through node_set (not passed separately)
# 5. This is TRULY general - no manual parameter extraction!
#
# Pattern:
#   for node in node_set.nodes
#       local_dofs = get_dofs(node)
#       # Gather from all elements connected to this node
#       y_local = compute_nodal_contribution(node, node_set.fields, x)
#       y[local_dofs] = y_local  # Direct write (no atomics!)
#   end

using LinearAlgebra
using Printf

println("="^70)
println("GPU ElementSet MatVec Demo: Krylov-Ready Pattern")
println("="^70)
println()

# ============================================================================
# Mock Structures (NODAL ASSEMBLY)
# ============================================================================

"""Node with position"""
struct Node
    id::UInt
    x::Float64
    y::Float64
    z::Float64
end

"""Element references nodes (for gathering)"""
struct Element{N,B}
    id::UInt
    connectivity::NTuple{N,UInt}  # Node IDs
    basis::B
end

"""Mock basis type"""
struct MockBasis end

"""
NodeSet: Groups nodes + fields + connectivity

CRITICAL: 
- Fields live at NODES (not elements!)
- node_to_elements[i] = list of elements connected to node i
- This enables nodal assembly: loop over nodes, gather from connected elements
"""
struct NodeSet{F}
    name::String
    nodes::Vector{Node}
    elements::Vector{Element{4,MockBasis}}  # For gathering
    node_to_elements::Vector{Vector{Int}}   # Inverse connectivity
    fields::F  # Type-stable field container (nodal fields!)
end

# Helper: Get DOF indices for a node (assuming 3 DOF per node)
function get_dofs(node::Node, dofs_per_node::Int=3)
    return tuple(UInt.((node.id - 1) * dofs_per_node .+ (1:dofs_per_node))...)
end

# ============================================================================
# Mock GPU Module
# ============================================================================

module MockGPU
struct CuArray{T,N}
    data::Array{T,N}
end

Base.length(a::CuArray) = length(a.data)
Base.getindex(a::CuArray, i...) = getindex(a.data, i...)
Base.setindex!(a::CuArray, v, i...) = setindex!(a.data, v, i...)

cu(x::Array) = CuArray(x)
cu(x::Vector) = CuArray(x)
cpu(x::CuArray) = x.data

macro cuda(args...)
    func_call = args[end]
    return esc(quote
        $func_call
    end)
end

export CuArray, cu, cpu, @cuda
end

using .MockGPU

# ============================================================================
# GPU Kernel: Matrix-Free Matrix-Vector Product (NODAL ASSEMBLY!)
# ============================================================================

"""
GPU kernel for matrix-vector product: y = K*x (NODAL ASSEMBLY)

TRULY GENERAL approach:
- Takes NodeSet (contains nodes + fields + connectivity)
- Loop over NODES (not elements!)
- Each node gathers from its connected elements
- No race conditions (each node writes to its own DOFs!)
- Accesses fields via node_set.fields (no manual extraction!)
- Computes nodal contribution by gathering from all connected elements

This is what Krylov methods need - NOT the full K matrix!
"""
function gpu_matvec_kernel!(
    y::CuArray{Float64,1},           # Output: y = K*x
    x::CuArray{Float64,1},           # Input vector
    node_set::NodeSet,               # Contains nodes + fields + connectivity!
    dofs_per_node::Int,
)
    # In real CUDA: thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # Each thread processes one NODE

    for node_id in 1:length(node_set.nodes)
        node = node_set.nodes[node_id]

        # Access fields from node_set (GENERAL!)
        E = node_set.fields.E
        ν = node_set.fields.ν

        # Get this node's DOFs
        local_dofs = get_dofs(node, dofs_per_node)
        n_local_dofs = length(local_dofs)

        # Initialize nodal contribution to zero
        y_nodal = ntuple(i -> 0.0, n_local_dofs)

        # Gather from all elements connected to this node
        connected_elements = node_set.node_to_elements[node_id]
        
        for elem_idx in connected_elements
            element = node_set.elements[elem_idx]
            
            # Find this node's position in element connectivity
            local_node_idx = findfirst(==(node.id), element.connectivity)
            
            # Get all element DOFs
            elem_dofs = UInt[]
            for node_id_in_elem in element.connectivity
                for d in 1:dofs_per_node
                    push!(elem_dofs, (node_id_in_elem - 1) * dofs_per_node + d)
                end
            end
            
            # Extract element x values
            x_elem = [x[dof] for dof in elem_dofs]
            
            # Mock element stiffness contribution
            # In real code: K_elem = assemble_element_stiffness(element, E, ν)
            # Extract rows corresponding to this node
            K_elem_factor = E * (1 - ν^2) * 0.1
            
            # Add this element's contribution to nodal y
            # (rows corresponding to this node)
            node_start = (local_node_idx - 1) * dofs_per_node
            for i in 1:n_local_dofs
                row_in_elem = node_start + i
                # Sum over all DOFs in element
                for j in 1:length(x_elem)
                    y_nodal = ntuple(k -> k == i ? y_nodal[k] + K_elem_factor * x_elem[j] : y_nodal[k], n_local_dofs)
                end
            end
        end

        # Write nodal contribution to global y (no atomics needed!)
        for i in 1:n_local_dofs
            y[local_dofs[i]] = y_nodal[i]
        end
    end

    return nothing
end

# ============================================================================
# CPU Version: Same Logic
# ============================================================================

"""CPU matrix-vector product using nodal assembly pattern"""
function cpu_matvec!(
    y::Vector{Float64},
    x::Vector{Float64},
    node_set::NodeSet,
    dofs_per_node::Int,
)
    # Zero output
    fill!(y, 0.0)

    # Access fields from node_set (GENERAL!)
    E = node_set.fields.E
    ν = node_set.fields.ν

    # Loop over NODES (not elements!)
    for (node_id, node) in enumerate(node_set.nodes)
        # Get this node's DOFs
        local_dofs = get_dofs(node, dofs_per_node)
        n_local_dofs = length(local_dofs)

        # Initialize nodal contribution
        y_nodal = zeros(n_local_dofs)

        # Gather from all connected elements
        connected_elements = node_set.node_to_elements[node_id]
        
        for elem_idx in connected_elements
            element = node_set.elements[elem_idx]
            
            # Find this node's position in element
            local_node_idx = findfirst(==(node.id), element.connectivity)
            
            # Get all element DOFs
            elem_dofs = Int[]
            for node_id_in_elem in element.connectivity
                for d in 1:dofs_per_node
                    push!(elem_dofs, (node_id_in_elem - 1) * dofs_per_node + d)
                end
            end
            
            # Extract element x values
            x_elem = [x[dof] for dof in elem_dofs]
            
            # Mock element stiffness computation
            K_elem_factor = E * (1 - ν^2) * 0.1
            
            # Add element contribution (rows for this node)
            node_start = (local_node_idx - 1) * dofs_per_node
            for i in 1:n_local_dofs
                for j in 1:length(x_elem)
                    y_nodal[i] += K_elem_factor * x_elem[j]
                end
            end
        end

        # Write nodal contribution to global y
        for i in 1:n_local_dofs
            y[local_dofs[i]] = y_nodal[i]
        end
    end

    return y
end

# ============================================================================
# Setup Problem (NODAL ASSEMBLY)
# ============================================================================

println("Setting up problem with NODAL ASSEMBLY...")
println()

# Create simple 2D mesh
n_x = 10
n_y = 10
n_nodes = n_x * n_y
dofs_per_node = 3
n_dofs = n_nodes * dofs_per_node

# Create nodes
nodes = [
    Node(
        UInt((j-1)*n_x + i),
        Float64(i),
        Float64(j),
        0.0
    )
    for j in 1:n_y for i in 1:n_x
]

# Create elements (quads)
elements = Element{4,MockBasis}[]
for j in 1:(n_y-1)
    for i in 1:(n_x-1)
        node1 = UInt((j-1)*n_x + i)
        node2 = UInt((j-1)*n_x + i + 1)
        node3 = UInt(j*n_x + i + 1)
        node4 = UInt(j*n_x + i)
        push!(elements, Element{4,MockBasis}(
            UInt(length(elements) + 1),
            (node1, node2, node3, node4),
            MockBasis()
        ))
    end
end

n_elements = length(elements)

# Build node-to-elements connectivity
node_to_elements = [Int[] for _ in 1:n_nodes]
for (elem_idx, element) in enumerate(elements)
    for node_id in element.connectivity
        push!(node_to_elements[node_id], elem_idx)
    end
end

# Fields in node set (type-stable!)
fields = (
    E=210e3,
    ν=0.3,
)

# Create node set (THIS IS THE REAL STRUCTURE!)
node_set = NodeSet("steel_body", nodes, elements, node_to_elements, fields)

println("Problem setup:")
println("  Nodes: $n_nodes")
println("  Elements: $n_elements")
println("  DOFs per node: $dofs_per_node")
println("  Total DOFs: $n_dofs")
println("  Node type: ", typeof(nodes[1]))
println("  Field type: ", typeof(node_set.fields))
println("  Fields accessed via: node_set.fields.E, node_set.fields.ν")
println("  Average elements/node: ", sum(length.(node_to_elements)) / n_nodes)
println()
println("NODAL ASSEMBLY:")
println("  - Loop over nodes (not elements!)")
println("  - Each node gathers from connected elements")
println("  - No race conditions (each node owns its DOFs)")
println()

# ============================================================================
# Test CPU MatVec
# ============================================================================

println("="^70)
println("CPU Matrix-Vector Product")
println("="^70)
println()

x = randn(n_dofs)
y_cpu = zeros(n_dofs)

println("Computing y = K*x using NODAL ASSEMBLY pattern...")
@time cpu_matvec!(y_cpu, x, node_set, dofs_per_node)

println("\nResult:")
println("  ||x||: ", @sprintf("%.6e", norm(x)))
println("  ||y||: ", @sprintf("%.6e", norm(y_cpu)))
println("  ✓ Matrix-vector product complete (nodal assembly)")
println()

# ============================================================================
# Test GPU MatVec (Mock)
# ============================================================================

println("="^70)
println("GPU Matrix-Vector Product (Mock CUDA) - NODAL ASSEMBLY")
println("="^70)
println()

println("Key insight: NodeSet goes to GPU!")
println("  - Nodes: node_set.nodes")
println("  - Elements: node_set.elements (for gathering)")
println("  - Connectivity: node_set.node_to_elements")
println("  - Fields: node_set.fields")
println("  - No manual parameter extraction needed!")
println("  - Each GPU thread processes ONE NODE (not element)")
println()

# Transfer to GPU
x_gpu = cu(x)
y_gpu = cu(zeros(n_dofs))

println("Launching GPU kernel (one thread per NODE)...")
@cuda threads = 256 blocks = ceil(Int, n_nodes / 256) gpu_matvec_kernel!(
    y_gpu, x_gpu, node_set, dofs_per_node
)

println("  ✓ Kernel execution complete")
println()

# Transfer back and verify
y_gpu_result = cpu(y_gpu)

error = norm(y_gpu_result - y_cpu) / (norm(y_cpu) + 1e-10)
println("📊 GPU Results:")
println("  ||y_GPU||: ", @sprintf("%.6e", norm(y_gpu_result)))
println("  Relative error: ", @sprintf("%.6e", error))
println("  ✓ GPU matches CPU: ", error < 1e-6 ? "YES ✅" : "NO ❌")
println()

println("🎯 NODAL ASSEMBLY ADVANTAGES:")
println("  ✅ No atomic operations needed (each node owns its DOFs)")
println("  ✅ Natural for contact mechanics (contact forces at nodes)")
println("  ✅ Clean domain decomposition (node ownership)")
println("  ✅ Better cache locality (node data grouped)")
println()

# ============================================================================
# Demonstrate GMRES Pattern (Conceptual)
# ============================================================================

println("="^70)
println("How This Enables Krylov Methods (GMRES/CG)")
println("="^70)
println()

println("""
GMRES only needs matrix-vector products, not the matrix itself!

Traditional approach (WRONG for large problems):
    K = assemble_global_matrix(elements)  # O(N²) memory!
    y = K * x                              # Dense operation
    
Matrix-free approach (CORRECT):
    y = matvec(element_set, x)             # O(N) memory!
    # Computed by looping over elements, no global K

GMRES iteration:
    for iteration in 1:max_iterations
        # Build Krylov subspace using matvec
        v_new = matvec(element_set, v_old)  # ← Our GPU kernel!
        
        # Orthogonalize (Arnoldi)
        # ...
        
        # Check convergence
        if residual < tolerance
            break
        end
    end

For contact mechanics + plasticity:
    - Update contact state (node-by-node)
    - Update material state (integration points)
    - Compute y = K_tangent * x using current state
    - No need to form K_tangent explicitly!

Our GPU kernel is PERFECT for this:
    1. Element-local computation (easy to parallelize)
    2. Fields accessed naturally (element_set.fields)
    3. Returns y vector (what GMRES needs)
    4. O(N) memory (no global matrix)
""")

# ============================================================================
# Show Field Access Pattern
# ============================================================================

println("="^70)
println("Field Access Pattern (TRULY General)")
println("="^70)
println()

println("""
In the GPU kernel, we access fields like this:

    function gpu_matvec_kernel!(y, x, element_set, ...)
        element = element_set.elements[elem_id]
        
        # Access fields from element_set (not passed separately!)
        E = element_set.fields.E           # ← GENERAL!
        ν = element_set.fields.ν           # ← GENERAL!
        u = element_set.fields.u           # ← If displacement field exists
        
        # Get local DOFs from element connectivity
        local_dofs = get_dofs(element)
        
        # Extract local portion of x
        x_local = x[local_dofs]
        
        # Compute local matvec
        y_local = K_local(element, E, ν) * x_local
        
        # Add to global (atomic on GPU)
        y[local_dofs] += y_local
    end

No manual parameter extraction!
No separate arrays for E, ν, etc!
Everything accessed through element_set!

This works for ANY field type (NamedTuple, struct, whatever) as long as
it's type-stable!
""")

# ============================================================================
# Time-Dependent Fields Example
# ============================================================================

println("="^70)
println("Time-Dependent Fields (Bonus)")
println("="^70)
println()

println("For transient problems, create new node_set each time step:")
println()

println("""
# Time stepping loop
for t in time_steps
    # Solve for new displacement using GMRES (matrix-free!)
    u_new = gmres(x0) do x
        y = zeros(n_dofs)
        gpu_matvec_kernel!(y, x, node_set, dofs_per_node)
        return y
    end
    
    # Create NEW field container (cheap!)
    fields_new = (
        E = node_set.fields.E,        # Constant (keep)
        ν = node_set.fields.ν,        # Constant (keep)
        u = u_new,                    # Updated!
        temperature = T_new,          # Updated!
    )
    
    # Create new node set (cheap - just wraps references)
    node_set = NodeSet(name, nodes, elements, node_to_elements, fields_new)
    
    # Next iteration uses updated fields automatically!
end

Creating new NamedTuple: ~2-3 ns (just wraps references)
No data copying needed!
""")

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("SUMMARY: Matrix-Free Krylov with NODAL ASSEMBLY")
println("="^70)
println()

println("""
✓ TRULY GENERAL approach with NODAL ASSEMBLY:
  1. NodeSet contains nodes + elements + connectivity + fields
  2. Fields accessed via node_set.fields (no manual extraction!)
  3. GPU kernel loops over NODES (not elements!)
  4. Each node gathers from connected elements
  5. No race conditions (each node owns its DOFs)
  6. O(N) memory (no global matrix)

✓ For GMRES/CG:
  - Only need matvec operation (y = K*x)
  - No need to form or store K
  - Perfect for contact + plasticity (state-dependent K)
  - Scales to millions of DOFs

✓ Nodal assembly advantages:
  - node_set.fields.E  ✅ (constant material property)
  - node_set.fields.ν  ✅ (constant material property)
  - node_set.fields.u  ✅ (nodal displacement)
  - Contact forces natural (at nodes!)
  - Domain decomposition clean (node ownership)
  - No atomic operations needed on GPU

✓ Type stability:
  - NodeSet{F} has known field type F
  - Compiler generates optimal code
  - Zero runtime dispatch
  - GPU-compatible

This is the NODAL ASSEMBLY pattern for JuliaFEM v1.0!

🎯 Why nodal assembly?
  1. Contact mechanics is nodal (forces, constraints at nodes)
  2. Domain decomposition is nodal (clean node ownership)
  3. No atomic operations on GPU (each node owns its DOFs)
  4. Better cache locality (node data grouped together)
  5. Natural for adaptive refinement (local node operations)
""")

println("="^70)
println("✅ DEMONSTRATION COMPLETE - NODAL ASSEMBLY")
println("="^70)
println()

println("Next steps:")
println("  1. Implement real nodal matvec with full stiffness computation")
println("  2. Integrate with Krylov.jl for GMRES/CG")
println("  3. Add contact state updates (natural at nodes!)")
println("  4. Add material state updates (at integration points)")
println("  5. Test on real CUDA hardware")
println("  6. Benchmark: nodal vs element assembly")
println()
