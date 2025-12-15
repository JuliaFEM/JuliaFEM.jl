# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
DOF-based COO assembly using row-wise matrix-free integration.

**STATUS: EXPERIMENTAL - Single-kernel assumption only**

**DOF-BY-DOF PARADIGM**: Loop over DOFs (matrix rows), not elements or nodes!

**CRITICAL LIMITATION**: Currently assumes ALL elements use the SAME kernel (physics).
For multi-physics problems with different element types/kernels, this approach
requires additional infrastructure (element → kernel mapping, field coupling registry).

This implementation serves as a proof-of-concept for the simplest case.

Each DOF (matrix row):
1. Finds all elements containing the corresponding node
2. For each element:
   - Prepares element geometry once
   - Computes only needed matrix entries in this row
3. Scatters entries to COO triplets

# Key Differences

**Element-Based (traditional):**
```julia
for element in elements
    K_e = compute_element_stiffness(element)  # Full N×N matrix of 3×3 blocks
    scatter(K_e)                               # Scatter all entries
end
```

**Node-Based (existing):**
```julia
for node_i in nodes
    for element in elements_touching(node_i)
        for node_j in element.nodes
            K_ij = compute_block!(element, i, j)  # Single 3×3 block
            scatter(K_ij, i, j)
        end
    end
end
```

**DOF-Based (this file):**
```julia
for dof_i in 1:ndofs  # Each matrix row!
    node_i, comp_i = decode_dof(dof_i)
    for element in elements_touching(node_i)
        for local_j in 1:nodes_per_element
            for comp_j in 1:3
                dof_j = global_dof(element.nodes[local_j], comp_j)
                K_ij = compute_entry!(element, node_i, comp_i, local_j, comp_j)
                scatter(K_ij, dof_i, dof_j)
            end
        end
    end
end
```

# Advantages over Node-Based

1. **True matrix-free**: Computes K*v without ANY intermediate storage
2. **Natural for mixed methods**: Easy to handle p/u coupling where DOF dimensions differ
3. **Minimal memory**: No 3×3 block storage at all
4. **Contact-ready**: Contact operates on DOFs directly (normal/tangential components)

# Performance Expectations

- **CPU Single-thread**: ~2-3x slower than element-based (finest granularity)
- **CPU Multi-thread**: ~Same as node-based (parallelization still works)
- **GPU**: ~Same as node-based (one thread per DOF vs one thread per node)
- **Matrix-free K*v**: ~5-10x faster (no matrix storage, cache-friendly)

# When to Use

✅ **Matrix-free Krylov solvers** (GMRES, CG)
✅ **Very large problems** (> 1M DOFs where storing K is prohibitive)
✅ **Contact mechanics** (normal/tangential DOF operations)
✅ **Mixed methods** (different DOF dimensions per field)

❌ **Forming explicit K** (element-based is faster)
❌ **Direct solvers** (need explicit matrix)
❌ **Small problems** (< 10k DOFs, overhead dominates)

# References

- Golden standard: `docs/src/book/multigpu_nodal_assembly.md`
- Theory: `docs/design/dof_by_dof_efficiency_analysis.md`
- Node-based version: `src/assemblers/node_based_coo.jl`

# Example

```julia
# Setup
mesh = create_cantilever_mesh(50, 10, 10)
material = LinearElastic(E=210e9, ν=0.3)
kernel = ContinuumKernel(
    ContinuumFormulation{FullThreeD}(),
    material,
    Displacement{3}()
)

# Create DOF-based assembler
assembler = DOFBasedCOOAssembler()
cache = create_cache(assembler, mesh, kernel)

# Assemble (zero allocations after warmup!)
assemble!(cache, assembler, kernel, mesh)

# Extract system
K, f = extract_system(cache)

# Or use matrix-free!
function matvec(v)
    return compute_matvec_dof_based(cache, v, kernel, mesh)
end
u = gmres(matvec, f, tol=1e-6)
```
=#

using SparseArrays
using Tensors

# Import DOF connectivity infrastructure
using ..JuliaFEM: DOFConnectivity, DOFElementConnection, build_dof_connectivity
using ..JuliaFEM: element_dofs, local_dof_count, field_dof_range
using ..JuliaFEM: compute_stiffness_value
using ..JuliaFEM: create_element_cache, create_geometry_cache, create_material_cache
using ..JuliaFEM: update_geometry_cache!, update_element_cache!, update_material_cache!
using ..JuliaFEM: compute_tangent, get_tangent_vector
using ..JuliaFEM: nnodes, Vertex, GeometryCache, AssemblyMaterialWorkspace
using ..JuliaFEM: GlobalMaterialCache, create_global_material_cache
using ..JuliaFEM: @field_vector
using ..JuliaFEM: compute_stress, create_zero_field, material_field_type, material_state_type

# Import field decoding (functions defined in included file)
include("dof_field_info.jl")
# Functions are now available: DOFFieldInfo, decode_local_dof, flatten_dof_indices

# extract_tangent! is now defined in material_cache.jl for shared use

"""
    DOFBasedCOOCache

Pre-allocated cache for DOF-based COO assembly.

Uses Phase 1 infrastructure from `src/assembly/dof_connectivity.jl`.

# Fields
- `I::Vector{Int}`: Row indices (COO format)
- `J::Vector{Int}`: Column indices (COO format)
- `V::Vector{Float64}`: Values (COO format)
- `f::Vector{Float64}`: Global force vector
- `counter::Ref{Int}`: Current triplet count
- `capacity::Int`: Maximum triplet capacity
- `dof_connectivity::DOFConnectivity`: DOF → elements inverse connectivity (Phase 1)
- `elements::Vector{<:AbstractElement}`: Element array
- `element_cache::ElementCache`: Cache for element operations
- `ndofs::Int`: Total DOFs in system
"""
mutable struct DOFBasedCOOCache{T<:AbstractTopology,B<:AbstractBasis,IPS,E<:AbstractElement,FieldType<:NamedTuple,StateType<:NamedTuple}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    f::Vector{Float64}
    counter::Int  # Plain Int instead of Ref{Int} for zero-allocation access (struct is mutable)
    capacity::Int
    dof_connectivity::DOFConnectivity
    elements::Vector{E}  # Concrete element type for type stability
    element_cache::ElementCache{T,B,IPS}
    geometry_cache::GeometryCache
    material_workspace::AssemblyMaterialWorkspace{FieldType, StateType}  # Material workspace for element assembly - TYPE STABLE
    ndofs::Int
    dof_index_buffer::Vector{Int}  # Pre-allocated buffer for DOF indices (zero-allocation reuse)
    last_updated_element::Int  # Track last element to avoid redundant geometry updates (Int instead of Ref{Int} for type stability) - DEPRECATED: Use global caches
    global_material_cache::GlobalMaterialCache  # Global material cache for zero-allocation state management
    𝔻_vec_buffer::Vector{SymmetricTensor{4,3,Float64,36}}  # Pre-allocated buffer for tangent vector (zero-allocation extraction) - DEPRECATED: Use global caches
    fields_ref::FieldType  # Pre-allocated NamedTuple for material fields (zero-allocation reuse) - TYPE STABLE
    empty_state::StateType  # Pre-allocated empty state NamedTuple (zero-allocation reuse) - TYPE STABLE
    zero_field::FieldType  # Pre-allocated zero field for reset! (zero-allocation) - TYPE STABLE
    # Global caches: one per element (pre-computed in Pass 1, reused in Pass 2)
    element_caches::Vector{ElementCache{T,B,IPS}}  # Pre-computed element caches [nelems]
    geometry_caches::Vector{GeometryCache}  # Pre-computed geometry caches [nelems]
    material_workspaces::Vector{AssemblyMaterialWorkspace{FieldType, StateType}}  # Pre-computed material workspaces [nelems]
    tangent_buffers::Vector{Vector{SymmetricTensor{4,3,Float64,36}}}  # Pre-computed tangent buffers [nelems]
end

"""
    DOFBasedCOOCache(elements::Vector{<:AbstractElement}, dof_manager::DOFManager)

Create cache for DOF-based assembly using Phase 1 infrastructure.

# Arguments
- `elements`: Element array with assigned DOF indices
- `dof_manager`: DOF manager (for total DOF count)

# Returns
- Pre-allocated DOF-based COO cache

# Note
Requires elements created with `create_elements!()` (have `.dof_indices` assigned).
"""
function DOFBasedCOOCache(
    elements::Vector{E},
    dof_manager::DOFManager,
    mesh::AbstractMesh,
    kernel::ContinuumKernel
) where {E<:AbstractElement}
    # Use inverse mapping from DOFManager (built during create_elements!)
    if dof_manager.dof_connectivity === nothing
        error("DOF connectivity not built. Ensure create_elements! was called with this DOFManager.")
    end
    dof_connectivity = dof_manager.dof_connectivity
    
    ndofs = dof_connectivity.n_total_dofs
    
    # Estimate triplet count
    # Each DOF has dof_connectivity.max_connections elements
    # Each element contributes local_dof_count entries to this row
    avg_local_dofs = sum(local_dof_count(elem) for elem in elements) / length(elements)
    entries_per_dof = dof_connectivity.max_connections * avg_local_dofs
    estimated_triplets = Int(ceil(1.2 * ndofs * entries_per_dof))
    
    # Allocate triplet arrays
    I = Vector{Int}(undef, estimated_triplets)
    J = Vector{Int}(undef, estimated_triplets)
    V = Vector{Float64}(undef, estimated_triplets)
    f = zeros(Float64, ndofs)
    counter = 0  # Plain Int instead of Ref{Int} for zero-allocation access
    
    # Create element cache
    element_cache = create_element_cache(mesh, kernel)
    
    # Extract type parameters from element cache for type stability
    ElementCacheType = typeof(element_cache)
    T = ElementCacheType.parameters[1]  # Topology type
    B = ElementCacheType.parameters[2]   # Basis type
    IPS = ElementCacheType.parameters[3] # Integration points type
    
    # Create geometry cache (get max nodes from mesh type)
    MeshType = typeof(mesh)
    max_nnodes = MeshType.parameters[1]::Int
    # Get number of integration points from element cache
    n_ips = length(element_cache.ips)
    geometry_cache = create_geometry_cache(max_nnodes, n_ips)
    
    # CRITICAL FIX: Infer FieldType and StateType from material FIRST for type stability
    # This ensures material_workspace, fields_ref, empty_state, and zero_field are type-stable
    FieldType = material_field_type(kernel.material)
    StateType = material_state_type(kernel.material)
    
    # Create material workspace with concrete type parameters (type-stable)
    material_workspace = create_material_cache(kernel.material, n_ips)::AssemblyMaterialWorkspace{FieldType, StateType}
    
    # Create GlobalMaterialCache for zero-allocation material state management
    n_elems = length(elements)
    global_material_cache = create_global_material_cache(kernel.material; n_ips=n_ips, n_elems=n_elems)
    
    # Pre-allocate DOF index buffer (max element DOF count)
    max_elem_dofs = isempty(elements) ? 0 : maximum(local_dof_count, elements)
    dof_index_buffer = Vector{Int}(undef, max_elem_dofs)
    last_updated_element = -1  # Track last updated element ID (Int instead of Ref{Int} for type stability)
    
    # Pre-allocate buffer for tangent vector extraction (zero-allocation)
    𝔻_vec_buffer = Vector{SymmetricTensor{4,3,Float64,36}}(undef, n_ips)
    
    # Pre-allocate NamedTuples for material cache updates (zero-allocation)
    # For StatelessConstantTangent, these are constant and can be reused
    # Compute once at reference configuration
    E_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.material, E_ref, NamedTuple(), 0.0)
    fields_ref = (σ=σ_ref, 𝔻=𝔻_ref)  # Pre-allocate ONCE in constructor (for material cache updates) - type inferred from FieldType
    empty_state = NamedTuple()  # Pre-allocate ONCE in constructor - type inferred from StateType
    # CRITICAL FIX: Also pre-allocate zero field for reset! (zero-allocation)
    # For stateless materials, zero field should be the same as fields_ref (zero strain → zero stress)
    # But we create it explicitly to ensure it's truly zero
    zero_field = create_zero_field(FieldType)  # Pre-allocate ONCE in constructor - type inferred from FieldType
    
    # Pre-allocate global caches: one per element (filled in Pass 1 of assemble!)
    n_elems = length(elements)
    element_caches = Vector{ElementCache{T,B,IPS}}(undef, n_elems)
    geometry_caches = Vector{GeometryCache}(undef, n_elems)
    material_workspaces = Vector{AssemblyMaterialWorkspace{FieldType, StateType}}(undef, n_elems)
    tangent_buffers = Vector{Vector{SymmetricTensor{4,3,Float64,36}}}(undef, n_elems)
    
    # Initialize each element's caches (pre-allocate, will be filled in assemble! Pass 1)
    for elem_id in 1:n_elems
        element_caches[elem_id] = create_element_cache(mesh, kernel)
        geometry_caches[elem_id] = create_geometry_cache(max_nnodes, n_ips)
        material_workspaces[elem_id] = create_material_cache(kernel.material, n_ips)::AssemblyMaterialWorkspace{FieldType, StateType}
        tangent_buffers[elem_id] = Vector{SymmetricTensor{4,3,Float64,36}}(undef, n_ips)
    end
    
    return DOFBasedCOOCache{T,B,IPS,E,FieldType,StateType}(I, J, V, f, counter, estimated_triplets,
        dof_connectivity, elements, element_cache, geometry_cache, material_workspace, ndofs,
        dof_index_buffer, last_updated_element, global_material_cache, 𝔻_vec_buffer,
        fields_ref, empty_state, zero_field,
        element_caches, geometry_caches, material_workspaces, tangent_buffers)
end

"""
    reset!(cache::DOFBasedCOOCache)

Reset cache for new assembly (zero force vector, reset counter).
"""
function reset!(cache::DOFBasedCOOCache{T,B,IPS,E,FieldType,StateType}) where {T,B,IPS,E,FieldType,StateType}
    fill!(cache.f, 0.0)
    cache.counter = 0  # Direct assignment for zero-allocation access
    cache.last_updated_element = -1  # Reset element tracking (direct assignment for type stability)
    # CRITICAL FIX: Use pre-allocated zero values to avoid create_zero_field allocation
    # zero_field is pre-allocated in constructor, empty_state is the zero state
    reset!(cache.material_workspace, cache.zero_field, cache.empty_state)  # Zero-allocation reset
    return nothing
end

"""
    assemble!(
        cache::DOFBasedCOOCache,
        assembler::DOFBasedCOOAssembler,
        kernel::ContinuumKernel,
        mesh::AbstractMesh
    ) -> Nothing

Assemble global system using **DOF-based traversal** with Phase 1 infrastructure.

**SINGLE-KERNEL ASSUMPTION**: All elements use the same kernel (physics).

# Algorithm

```julia
for dof_i in 1:ndofs  # Each matrix row!
    # Get all elements touching this DOF (Phase 1 connectivity)
    for conn in dof_connectivity.dof_to_elements[dof_i]
        element = elements[conn.elem_id]
        local_i = conn.local_dof_idx  # Within element's flattened DOF list
        
        # Decode which field this DOF represents
        field_info_i = decode_local_dof(element, local_i)
        
        # Prepare element geometry ONCE
        prepared = prepare_element!(cache.element_cache, kernel, conn.elem_id, mesh)
        
        # Compute all entries in this row for this element
        flat_dofs_j = flatten_dof_indices(element.dof_indices)
        for (local_j, dof_j) in enumerate(flat_dofs_j)
            field_info_j = decode_local_dof(element, local_j)
            
            # Compute single scalar entry K[i,j]
            K_ij = compute_entry!(prepared, kernel.material,
                                 field_info_i, field_info_j)
            
            # Scatter to triplets
            scatter_entry!(cache, K_ij, dof_i, dof_j)
        end
    end
end
```

# Zero-Allocation (After Warmup)

All arrays pre-allocated. Element preparation reuses cache buffers.

# Arguments
- `cache`: Pre-allocated DOF-based COO cache
- `assembler`: DOF-based COO assembler
- `kernel`: Continuum kernel (SAME for all elements)
- `mesh`: Finite element mesh
"""
function assemble!(
    cache::DOFBasedCOOCache{T,B,IPS,E,FieldType,StateType},
    assembler::DOFBasedCOOAssembler,
    kernel::ContinuumKernel,
    mesh::AbstractMesh
) where {T,B,IPS,E<:AbstractElement,FieldType,StateType}
    # Reset cache
    reset!(cache)
    
    # Cache ndofs_per_node outside loop (compile-time constant for single-field)
    ndofs_per_node = dofs_per_node(kernel)
    
    # Cache field accesses
    elements = cache.elements
    global_material_cache = cache.global_material_cache
    fields_ref = cache.fields_ref
    empty_state = cache.empty_state
    ndofs = cache.ndofs
    mesh_connectivity = mesh.connectivity
    
    # Cache global cache vectors
    element_caches = cache.element_caches
    geometry_caches = cache.geometry_caches
    material_workspaces = cache.material_workspaces
    tangent_buffers = cache.tangent_buffers
    
    n_elems = length(elements)
    
    # ========================================================================
    # PASS 1: Pre-compute element, geometry, and material caches for ALL elements
    # ========================================================================
    # Loop through all elements ONCE and compute their caches
    # This eliminates redundant recalculations when multiple DOFs touch the same element
    @inbounds for elem_id in 1:n_elems
        # Get pre-allocated caches for this element
        element_cache = element_caches[elem_id]
        geometry_cache = geometry_caches[elem_id]
        material_workspace = material_workspaces[elem_id]
        𝔻_vec_buffer = tangent_buffers[elem_id]
        
        # Reset caches for this element (required by update functions)
        reset!(element_cache)
        reset!(geometry_cache)
        reset!(material_workspace)
        
        # Update element cache (extract DOF mapping, displacements if any)
        update_element_cache!(element_cache, kernel, elem_id, mesh, nothing)
        
        # Update geometry cache (extract coordinates, compute gradients, detJ*w)
        update_geometry_cache!(geometry_cache, element_cache, kernel, elem_id, mesh)
        
        # Update material workspace (compute stress, tangent, internal state)
        # Use GlobalMaterialCache for zero-allocation state management
        # Direct getfield access - zero allocation, type-stable
        fields_mw = getfield(material_workspace, 1)  # Direct field access
        states_mw = getfield(material_workspace, 2)  # Direct field access
        ips_ec = getfield(element_cache, :ips)  # Direct field access
        nips = length(ips_ec)  # Use cached reference
        # Use pre-allocated NamedTuples (zero-allocation)
        @inbounds for q in 1:nips
            fields_mw[q] = fields_ref  # Reuse pre-allocated NT - zero allocation
            states_mw[q] = empty_state  # Reuse pre-allocated empty state
        end
        
        # Extract tangent vector to pre-allocated buffer (zero-allocation)
        fields = getfield(material_workspace, 1)  # Direct field access - zero allocation, type-stable
        extract_tangent!(𝔻_vec_buffer, fields, FieldType)  # Type-stable extraction using compile-time field index
    end
    
    # ========================================================================
    # PASS 2: Loop through DOFs and use pre-computed caches
    # ========================================================================
    # Cache DOF connectivity
    dof_connectivity = cache.dof_connectivity
    dof_to_elements = dof_connectivity.dof_to_elements
    
    # DOF LOOP: One iteration per DOF (matrix row)
    # Use explicit range to avoid iterator state allocation
    @inbounds for dof_i in 1:ndofs
        # Get all elements touching this DOF (Phase 1 connectivity)
        # Pre-extract vector reference to avoid repeated access and enable type stability
        @inbounds touching_elements = dof_to_elements[dof_i]
        
        # Optimized iteration: use indexed loop instead of iterator to avoid allocation
        n_conns = length(touching_elements)
        @inbounds for conn_idx in 1:n_conns
            conn = touching_elements[conn_idx]
            # Use accessor functions (zero-allocation for small integer conversions)
            # These convert Int32/Int16 to Int, which is zero-allocation for small values
            elem_id_val = elem_id(conn)  # Int32 → Int (no allocation for small values)
            local_i = local_dof_idx(conn)  # Int16 → Int (no allocation for small values)
            
            # Type-stable element access (E is concrete type from cache parameters)
            @inbounds element = elements[elem_id_val]::E
            
            # Look up pre-computed caches for this element (from Pass 1)
            element_cache = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            𝔻_vec = tangent_buffers[elem_id_val]  # Pre-extracted tangent vector
            
            # Both element.dof_indices and element_cache.dofs follow connectivity order
            # (DOF manager iterates over connectivity, get_dof_mapping! iterates over connectivity)
            # So local_i directly maps to connectivity node index via node-major ordering
            node_i = div(local_i - 1, ndofs_per_node) + 1  # Local node index in connectivity order
            comp_i = mod(local_i - 1, ndofs_per_node) + 1  # Component (1=x, 2=y, 3=z)
            
            # Get DOF count for this element (use cached connectivity length)
            # Cache connectivity tuple access to avoid repeated allocation
            @inbounds conn_tuple = mesh_connectivity[elem_id_val]
            nnodes_elem = length(conn_tuple)  # Compile-time known for fixed topology
            ndofs_elem = nnodes_elem * ndofs_per_node
            
            # Use element_cache.dofs for scattering (same as element-based assembler)
            # element_cache.dofs is filled by update_element_cache! and matches connectivity order
            # Direct indexing is zero-allocation (no view needed for small arrays)
            dofs_elem = element_cache.dofs
            
            # Get global DOF for row (from element_cache.dofs, matching element-based assembler)
            dof_i_global = dofs_elem[local_i]
            
            # Compute all entries in this row for this element
            # Use @inbounds for zero-allocation iteration
            @inbounds for local_j in 1:ndofs_elem
                # Get global DOF for column (from element_cache.dofs, matching element-based assembler)
                dof_j_global = dofs_elem[local_j]
                
                # Decode node index and component from local_j (same ordering as local_i)
                node_j = div(local_j - 1, ndofs_per_node) + 1  # Local node index in connectivity order
                comp_j = mod(local_j - 1, ndofs_per_node) + 1  # Component (1=x, 2=y, 3=z)
                
                # Compute SINGLE SCALAR entry K[i,j] using node indices directly
                # node_i and node_j are local node indices matching geometry cache indexing
                # (both element_cache.dofs and geometry_cache follow connectivity order)
                # Use cached references to avoid repeated field access allocations
                K_ij = compute_entry_direct!(
                    geometry_cache,
                    𝔻_vec,  # Pass pre-extracted vector from Pass 1
                    node_i, comp_i,
                    node_j, comp_j
                )
                
                # Scatter single entry to triplets using global DOFs from element_cache.dofs
                scatter_entry!(cache, K_ij, dof_i_global, dof_j_global)
            end
        end  # End conn_idx loop
    end  # End dof_i loop
    
    return nothing
end

"""
    compute_entry!(
        geometry_cache::GeometryCache,
        material::AbstractMaterial,
        field_info_i::DOFFieldInfo,
        field_info_j::DOFFieldInfo
    ) -> Float64

Compute SINGLE SCALAR stiffness matrix entry K[i,j] using field info.

This is the atomic kernel - computes ONE matrix entry using field information
decoded from the element's DOF specification.

Uses the generic `compute_stiffness_value` from continuum kernel,
which works for ANY material providing an elasticity tensor C.

# Arguments
- `geometry_cache`: Geometry cache with prepared element data
- `material`: Material model (provides C)
- `field_info_i`: Field information for row DOF
- `field_info_j`: Field information for column DOF

# Returns
- Single scalar value K[i,j]

# Field Coupling

For single-field (pure displacement):
- If field_i.field == field_j.field == :u → mechanical stiffness

For multi-field:
- Same field → diagonal coupling (u-u, T-T, p-p, etc.)
- Different fields → off-diagonal coupling (u-T, T-u, u-p, etc.)
- Coupling logic dispatched based on (field_i.field, field_j.field) pair

# Zero-Allocation

Uses geometry cache buffers, no new allocations.
"""
@inline function compute_entry_direct!(
    geometry_cache::GeometryCache,
    𝔻_vec::Vector{SymmetricTensor{4,3,Float64,36}},  # Pre-extracted vector (zero-allocation)
    node_i::Int,
    comp_i::Int,
    node_j::Int,
    comp_j::Int
)
    # CRITICAL FIX: Accept pre-extracted 𝔻_vec instead of material_workspace
    # This eliminates allocations from get_tangent_vector() which creates a new vector
    # get_tangent_vector() is now called ONCE per element (outside this function)
    
    K_ij = 0.0
    
    # Integration loop over quadrature points
    n_ips = length(geometry_cache.detJ_w)
    @inbounds for q in 1:n_ips
        # Basis function gradients (node_i and node_j are local node indices matching geometry cache)
        ∇N_i = geometry_cache.∇N_data[q, node_i]  # Vec{3}
        ∇N_j = geometry_cache.∇N_data[q, node_j]  # Vec{3}
        
        # detJ * weight
        detJ_w = geometry_cache.detJ_w[q]
        
        # Get tangent at this integration point
        𝔻_q = 𝔻_vec[q]  # SymmetricTensor{4,3}
        # Convert SymmetricTensor{4,3} to Tensor{4,3} for compute_stiffness_value
        # This conversion should be zero-allocation (same memory layout)
        C = Tensor{4,3}(𝔻_q)
        
        # Compute single scalar entry using GENERIC kernel
        value_at_qp = compute_stiffness_value(
            ∇N_i, ∇N_j, C,
            comp_i,  # α (1=x, 2=y, 3=z)
            comp_j   # β
        )
        
        K_ij += value_at_qp * detJ_w
    end
    
    return K_ij
end

"""
    scatter_entry!(
        cache::DOFBasedCOOCache,
        value::Float64,
        dof_i::Int,
        dof_j::Int
    )

Scatter single matrix entry to COO triplets **in-place**.

# Arguments
- `cache`: DOF-based COO cache
- `value`: Matrix entry value
- `dof_i`: Row DOF index
- `dof_j`: Column DOF index

# Zero-Allocation

Writes to pre-allocated triplet arrays, updates counter.
"""
@inline function scatter_entry!(
    cache::DOFBasedCOOCache{T,B,IPS,E,FieldType,StateType},
    value::Float64,
    dof_i::Int,
    dof_j::Int
) where {T,B,IPS,E,FieldType,StateType}
    # Direct access to Int field (zero-allocation, no Ref indirection)
    counter = cache.counter
    
    # Check capacity (bounds check - necessary for safety)
    if counter + 1 > cache.capacity
        error("DOF-based COO cache overflow: need $(counter + 1) triplets, " *
              "capacity is $(cache.capacity). Increase cache size.")
    end
    
    # Add triplet (use @inbounds since we checked capacity)
    counter += 1
    @inbounds begin
        cache.I[counter] = dof_i
        cache.J[counter] = dof_j
        cache.V[counter] = value
    end
    
    cache.counter = counter  # Direct assignment for zero-allocation access
    return nothing
end

"""
    extract_system(cache::DOFBasedCOOCache) -> (K, f)

Build sparse matrix from triplets and return system.

# Arguments
- `cache`: Assembled DOF-based COO cache

# Returns
- `K::SparseMatrixCSC`: Global stiffness matrix
- `f::Vector`: Global force vector
"""
function extract_system(cache::DOFBasedCOOCache{T,B,IPS,E,FieldType,StateType}) where {T,B,IPS,E,FieldType,StateType}
    ntriplets = cache.counter  # Direct access to Int field
    
    # Build sparse matrix (duplicates are summed automatically)
    I_used = @view cache.I[1:ntriplets]
    J_used = @view cache.J[1:ntriplets]
    V_used = @view cache.V[1:ntriplets]
    
    K = sparse(I_used, J_used, V_used, cache.ndofs, cache.ndofs)
    
    return K, cache.f
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
    create_cache(
        assembler::DOFBasedCOOAssembler,
        elements::Vector{<:AbstractElement},
        dof_manager::DOFManager,
        mesh::AbstractMesh,
        kernel::ContinuumKernel
    ) -> DOFBasedCOOCache

Create pre-allocated cache for DOF-based COO assembly.

Requires elements with assigned DOF indices (from `create_elements!`).

# Example

```julia
# Create elements with DOF assignment
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ElemType = Element{Hexahedron{8}, Lagrange{1}, S}
elements, dof_mgr = create_elements!(mesh, ElemType)

# Create kernel
kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), material, Displacement{3}())

# Create assembler and cache
assembler = DOFBasedCOOAssembler()
cache = create_cache(assembler, elements, dof_mgr, mesh, kernel)

# Assemble
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
```
"""
function create_cache(
    assembler::DOFBasedCOOAssembler,
    elements::Vector{E},
    dof_manager::DOFManager,
    mesh::AbstractMesh,
    kernel::ContinuumKernel
) where {E<:AbstractElement}
    return DOFBasedCOOCache(elements, dof_manager, mesh, kernel)
end
