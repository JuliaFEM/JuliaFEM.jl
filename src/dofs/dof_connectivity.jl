# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
DOF connectivity structures for DOF-based assembly.

Provides inverse mapping: DOF → Elements (which elements touch each DOF).

**GPU-Compatible**: Uses bits types and fixed-size arrays for device transfer.
**Zero-Allocation**: After initialization, all operations are allocation-free.
"""

# Import element and DOF manager interfaces (defined in elements.jl and dof_manager.jl)
using ..JuliaFEM: element_dofs, DOFManager

# ============================================================================
# DOF-ELEMENT CONNECTION (GPU-Compatible Bits Type)
# ============================================================================

"""
    DOFElementConnection

Maps a global DOF to an element and local DOF index.

**GPU-Compatible**: Bits type, can be transferred to GPU arrays.
**Zero-Allocation**: Immutable struct, stack-allocated.

# Fields
- `elem_id::Int32`: Element ID (index in elements array)
- `local_dof_idx::Int16`: Local DOF index within element (1-based)

# Memory
- 6 bytes per connection (Int32 + Int16)
- Aligned to 8 bytes (padding)
"""
struct DOFElementConnection
    elem_id::Int32      # Element ID (max 2^31 elements)
    local_dof_idx::Int16  # Local DOF index (max 2^15 = 32k DOFs per element)
    
    function DOFElementConnection(elem_id::Integer, local_dof_idx::Integer)
        # Validate ranges
        if elem_id < 1 || elem_id > typemax(Int32)
            error("Element ID $elem_id out of range [1, $(typemax(Int32))]")
        end
        if local_dof_idx < 1 || local_dof_idx > typemax(Int16)
            error("Local DOF index $local_dof_idx out of range [1, $(typemax(Int16))]")
        end
        return new(Int32(elem_id), Int16(local_dof_idx))
    end
end

# Convenience constructors
DOFElementConnection(elem_id::Int32, local_dof_idx::Int16) = 
    DOFElementConnection(Int(elem_id), Int(local_dof_idx))

# Accessors
elem_id(conn::DOFElementConnection) = Int(conn.elem_id)
local_dof_idx(conn::DOFElementConnection) = Int(conn.local_dof_idx)

# ============================================================================
# DOF CONNECTIVITY (CPU Version - Variable-Length Vectors)
# ============================================================================

"""
    DOFConnectivity

Inverse mapping: For each global DOF, which elements contain it?

**CPU Version**: Uses `Vector{Vector{DOFElementConnection}}` for variable-length lists.

**Zero-Allocation After Init**: All arrays pre-allocated, no heap allocations during access.

# Fields
- `dof_to_elements::Vector{Vector{DOFElementConnection}}`: For each DOF, list of connections
- `n_total_dofs::Int`: Total number of DOFs in system
- `max_connections::Int`: Maximum number of elements touching any single DOF

# Memory
- ~16 bytes per connection (DOFElementConnection + Vector overhead)
- O(n_elements × avg_dofs_per_element) total memory

# Usage
```julia
connectivity = build_dof_connectivity(elements, dof_manager)

# Access elements touching DOF i (zero-allocation!)
connections = connectivity.dof_to_elements[dof_i]
for conn in connections
    elem = elements[conn.elem_id]
    local_dof = conn.local_dof_idx
    # Process...
end
```
"""
struct DOFConnectivity
    dof_to_elements::Vector{Vector{DOFElementConnection}}
    n_total_dofs::Int
    max_connections::Int
    
    function DOFConnectivity(
        dof_to_elements::Vector{Vector{DOFElementConnection}},
        n_total_dofs::Int
    )
        # Compute max connections
        max_conn = isempty(dof_to_elements) ? 0 : maximum(length, dof_to_elements)
        return new(dof_to_elements, n_total_dofs, max_conn)
    end
end

# Accessors
Base.length(conn::DOFConnectivity) = conn.n_total_dofs
Base.getindex(conn::DOFConnectivity, dof_i::Int) = conn.dof_to_elements[dof_i]

"""
    build_dof_connectivity(
        elements::Vector{<:AbstractElement},
        dof_manager::DOFManager
    ) -> DOFConnectivity

Build DOF-to-element connectivity from elements.

**Complexity**: O(n_elements × avg_dofs_per_element) - optimal single pass.

**Zero-Allocation After Init**: All arrays pre-allocated during build, no allocations during access.

# Algorithm

1. Initialize: `dof_to_elements = [DOFElementConnection[] for _ in 1:n_total_dofs]`
2. Loop over elements:
   - Get element's global DOF indices: `elem.dof_indices`
   - For each (local_idx, global_dof):
     - Push `DOFElementConnection(elem_id, local_idx)` to `dof_to_elements[global_dof]`
3. Return `DOFConnectivity(dof_to_elements, n_total_dofs)`

# Arguments
- `elements`: Vector of elements with assigned DOF indices
- `dof_manager`: DOF manager (for total DOF count)

# Returns
- `DOFConnectivity` with inverse mapping

# Example

```julia
# Create elements with DOF assignment
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ElemType = Element{Tetrahedron{4}, Lagrange{1}, S}
elements, dof_mgr = create_elements!(mesh, ElemType)

# Build connectivity (one-time cost)
connectivity = build_dof_connectivity(elements, dof_mgr)

# Use in DOF-based assembly (zero-allocation access!)
for dof_i in 1:connectivity.n_total_dofs
    connections = connectivity.dof_to_elements[dof_i]  # O(1) access
    for conn in connections  # Zero-allocation iteration
        elem = elements[conn.elem_id]
        local_dof = conn.local_dof_idx
        # Process...
    end
end
```
"""
function build_dof_connectivity(
    elements::Vector,
    dof_manager::DOFManager
)
    n_total_dofs = dof_manager.total_dofs
    
    if n_total_dofs == 0
        return DOFConnectivity(Vector{DOFElementConnection}[], 0)
    end
    
    # Initialize: one vector per DOF
    dof_to_elements = [DOFElementConnection[] for _ in 1:n_total_dofs]
    
    # Pre-allocate capacity estimate (reduce reallocations)
    # Average: each DOF touched by ~6 elements (typical for 3D meshes)
    avg_connections = 6
    for vec in dof_to_elements
        sizehint!(vec, avg_connections)
    end
    
    # Loop over elements
    for (elem_id, elem) in enumerate(elements)
        # Get all global DOF indices for this element
        dofs = element_dofs(elem)  # Returns NTuple{N, UInt64}
        
        # Register element with each DOF
        for (local_idx, global_dof) in enumerate(dofs)
            # Validate DOF index
            if global_dof < 1 || global_dof > n_total_dofs
                error("Invalid global DOF index $global_dof (must be in [1, $n_total_dofs])")
            end
            
            # Create connection and push
            conn = DOFElementConnection(elem_id, local_idx)
            push!(dof_to_elements[global_dof], conn)
        end
    end
    
    return DOFConnectivity(dof_to_elements, n_total_dofs)
end

"""
    build_dof_connectivity(
        elements::Vector{<:AbstractElement},
        n_total_dofs::Int
    ) -> DOFConnectivity

Build DOF connectivity without DOFManager (direct DOF count).

Useful when DOF count is known but DOFManager is not available.
"""
function build_dof_connectivity(
    elements::Vector,
    n_total_dofs::Int
)
    if n_total_dofs == 0
        return DOFConnectivity(Vector{DOFElementConnection}[], 0)
    end
    
    # Initialize
    dof_to_elements = [DOFElementConnection[] for _ in 1:n_total_dofs]
    
    # Pre-allocate capacity
    avg_connections = 6
    for vec in dof_to_elements
        sizehint!(vec, avg_connections)
    end
    
    # Build connectivity
    for (elem_id, elem) in enumerate(elements)
        dofs = element_dofs(elem)
        for (local_idx, global_dof) in enumerate(dofs)
            if global_dof < 1 || global_dof > n_total_dofs
                error("Invalid global DOF index $global_dof (must be in [1, $n_total_dofs])")
            end
            conn = DOFElementConnection(elem_id, local_idx)
            push!(dof_to_elements[global_dof], conn)
        end
    end
    
    return DOFConnectivity(dof_to_elements, n_total_dofs)
end

# ============================================================================
# GPU-COMPATIBLE VERSION (Fixed-Size Arrays)
# ============================================================================

"""
    DOFConnectivityGPU

GPU-compatible DOF connectivity using fixed-size arrays.

**GPU Transfer**: All arrays are `CuArray`-compatible (bits types only).

**Memory**: Fixed-size matrices for predictable GPU memory usage.

# Fields
- `elem_ids::Matrix{Int32}`: Element IDs [max_connections, n_dofs]
- `local_indices::Matrix{Int16}`: Local DOF indices [max_connections, n_dofs]
- `counts::Vector{Int32}`: Actual connection count per DOF [n_dofs]
- `n_total_dofs::Int`: Total DOFs
- `max_connections::Int`: Maximum connections per DOF

# Memory
- ~(8 + 2) × max_connections × n_dofs bytes
- More memory than CPU version, but GPU-friendly

# Usage
```julia
# Build on CPU
conn_gpu = build_dof_connectivity_gpu(elements, dof_manager, max_conn=20)

# Transfer to GPU
using CUDA
elem_ids_gpu = CuArray(conn_gpu.elem_ids)
local_indices_gpu = CuArray(conn_gpu.local_indices)
counts_gpu = CuArray(conn_gpu.counts)

# Use in GPU kernel
@cuda threads=256 blocks=... dof_assembly_kernel!(
    elem_ids_gpu, local_indices_gpu, counts_gpu, ...
)
```
"""
struct DOFConnectivityGPU
    elem_ids::Matrix{Int32}        # [max_connections, n_dofs]
    local_indices::Matrix{Int16}   # [max_connections, n_dofs]
    counts::Vector{Int32}          # [n_dofs] - actual count per DOF
    n_total_dofs::Int
    max_connections::Int
    
    function DOFConnectivityGPU(
        elem_ids::Matrix{Int32},
        local_indices::Matrix{Int16},
        counts::Vector{Int32},
        n_total_dofs::Int,
        max_connections::Int
    )
        # Validate dimensions
        if size(elem_ids) != (max_connections, n_total_dofs)
            error("elem_ids must be [$max_connections, $n_total_dofs]")
        end
        if size(local_indices) != (max_connections, n_total_dofs)
            error("local_indices must be [$max_connections, $n_total_dofs]")
        end
        if length(counts) != n_total_dofs
            error("counts must have length $n_total_dofs")
        end
        return new(elem_ids, local_indices, counts, n_total_dofs, max_connections)
    end
end

"""
    build_dof_connectivity_gpu(
        elements::Vector{<:AbstractElement},
        dof_manager::DOFManager;
        max_connections::Int = 20
    ) -> DOFConnectivityGPU

Build GPU-compatible DOF connectivity with fixed-size arrays.

# Arguments
- `elements`: Vector of elements
- `dof_manager`: DOF manager
- `max_connections`: Maximum elements per DOF (default: 20)

# Returns
- `DOFConnectivityGPU` with fixed-size matrices

# Notes
- If a DOF has more than `max_connections` elements, an error is thrown.
- Choose `max_connections` based on mesh connectivity (typically 6-12 for 3D).
"""
function build_dof_connectivity_gpu(
    elements::Vector,
    dof_manager::DOFManager;
    max_connections::Int = 20
)
    n_total_dofs = dof_manager.total_dofs
    
    if n_total_dofs == 0
        return DOFConnectivityGPU(
            Matrix{Int32}(undef, 0, 0),
            Matrix{Int16}(undef, 0, 0),
            Int32[],
            0,
            0
        )
    end
    
    # Allocate fixed-size matrices
    elem_ids = zeros(Int32, max_connections, n_total_dofs)
    local_indices = zeros(Int16, max_connections, n_total_dofs)
    counts = zeros(Int32, n_total_dofs)
    
    # Build connectivity
    for (elem_id, elem) in enumerate(elements)
        dofs = element_dofs(elem)
        for (local_idx, global_dof) in enumerate(dofs)
            if global_dof < 1 || global_dof > n_total_dofs
                error("Invalid global DOF index $global_dof")
            end
            
            count = counts[global_dof] + 1
            if count > max_connections
                error("DOF $global_dof has more than $max_connections connections " *
                      "(found at least $count). Increase max_connections.")
            end
            
            elem_ids[count, global_dof] = Int32(elem_id)
            local_indices[count, global_dof] = Int16(local_idx)
            counts[global_dof] = count
        end
    end
    
    return DOFConnectivityGPU(elem_ids, local_indices, counts, n_total_dofs, max_connections)
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    connection_count(connectivity::DOFConnectivity, dof_i::Int) -> Int

Get number of elements touching DOF i.

**Zero-allocation**: Direct array length access.
"""
@inline connection_count(connectivity::DOFConnectivity, dof_i::Int) = 
    length(connectivity.dof_to_elements[dof_i])

"""
    connection_count(connectivity::DOFConnectivityGPU, dof_i::Int) -> Int

Get number of elements touching DOF i (GPU version).

**Zero-allocation**: Direct array access.
"""
@inline connection_count(connectivity::DOFConnectivityGPU, dof_i::Int) = 
    Int(connectivity.counts[dof_i])

"""
    is_empty(connectivity::DOFConnectivity, dof_i::Int) -> Bool

Check if DOF i has no connections.

**Zero-allocation**: Direct array length check.
"""
@inline is_empty(connectivity::DOFConnectivity, dof_i::Int) = 
    isempty(connectivity.dof_to_elements[dof_i])

"""
    is_empty(connectivity::DOFConnectivityGPU, dof_i::Int) -> Bool

Check if DOF i has no connections (GPU version).

**Zero-allocation**: Direct array access.
"""
@inline is_empty(connectivity::DOFConnectivityGPU, dof_i::Int) = 
    connectivity.counts[dof_i] == 0

# ============================================================================
# EXPORTS
# ============================================================================

# Export types and functions
export DOFElementConnection, DOFConnectivity, DOFConnectivityGPU
export build_dof_connectivity, build_dof_connectivity_gpu
export connection_count, is_empty
export elem_id, local_dof_idx
