# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    DOFElementConnection

Maps a global DOF to an element and local DOF index.

GPU-Compatible: Bits type, can be transferred to GPU arrays.
Zero-Allocation: Immutable struct, stack-allocated.

# Fields
- `elem_id::Int32`: Element ID (index in elements array)
- `local_dof_idx::Int16`: Local DOF index within element (1-based)

# Memory
- 6 bytes per connection (Int32 + Int16)
- Aligned to 8 bytes (padding)
"""
struct DOFElementConnection
    elem_id::Int32        # Element ID (max 2^31 elements)
    local_dof_idx::Int16  # Local DOF index (max 2^15 = 32k DOFs per element)

    function DOFElementConnection(elem_id::Integer, local_dof_idx::Integer)
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

"""
    DOFConnectivity

Inverse mapping: For each global DOF, which elements contain it?

CPU Version: Uses `Vector{Vector{DOFElementConnection}}` for variable-length lists.

Zero-Allocation After Init: All arrays pre-allocated, no heap allocations during access.

# Fields
- `dof_to_elements::Vector{Vector{DOFElementConnection}}`: For each DOF, list of connections
- `n_total_dofs::Int`: Total number of DOFs in system
- `max_connections::Int`: Maximum number of elements touching any single DOF

# Memory
- ~16 bytes per connection (DOFElementConnection + Vector overhead)
- O(n_elements × avg_dofs_per_element) total memory

# Usage
```julia
connectivity = build_dof_connectivity(elements, dof_handler)

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
        max_conn = isempty(dof_to_elements) ? 0 : maximum(length, dof_to_elements)
        return new(dof_to_elements, n_total_dofs, max_conn)
    end
end

"""
    DOFConnectivity()

Empty placeholder used by `DOFHandler` / `InterfaceDOFHandler` between
constructor and `create_elements!` (no element list yet, so no
DOF→element mapping). Callers detect this state by comparing
`connectivity.n_total_dofs` with `handler.total_dofs`.
"""
DOFConnectivity() = DOFConnectivity(Vector{DOFElementConnection}[], 0)

# Accessors
Base.length(conn::DOFConnectivity) = conn.n_total_dofs
Base.getindex(conn::DOFConnectivity, dof_i::Int) = conn.dof_to_elements[dof_i]
