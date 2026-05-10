# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
DOF-based COO assembly using row-wise matrix-free integration.

The assembler loops over DOFs (matrix rows) rather than over elements or
over nodes. The cache stores a **kernel column** ([`UniformKernelColumn`](@ref) or
[`PerElementKernelColumn`](@ref)): Pass 1 / Pass 2 use [`kernel_at`](@ref)`(cache, eid)`
so each volume element can carry its own kernel instance without heap
allocation in the assembly loops. Mixed physics on different DOF templates
still requires separate element vectors / handlers.

For each DOF (one matrix row):

1. Find all elements containing the corresponding node.
2. For each touching element, prepare the element geometry once and
   compute only the matrix entries that fall on this row.
3. Scatter the entries into the COO triplets.

# Comparison with the other assembly strategies

Element-based (traditional):

```julia
for element in elements
    K_e = compute_element_stiffness(element)  # full N x N matrix of 3 x 3 blocks
    scatter(K_e)
end
```

Node-based:

```julia
for node_i in nodes
    for element in elements_touching(node_i)
        for node_j in element.nodes
            K_ij = compute_block!(element, i, j)  # single 3 x 3 block
            scatter(K_ij, i, j)
        end
    end
end
```

DOF-based (this file):

```julia
for dof_i in 1:ndofs                     # one iteration per matrix row
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

# Why DOF-based

- True matrix-free `K*v` without any intermediate per-element matrix.
- Natural fit for mixed methods (`p`/`u` coupling with different DOF
  dimensions per field).
- No 3 x 3 nodal block storage.
- DOF-level granularity matches contact mechanics, where conditions
  apply to normal / tangential components, not to whole nodes.

# Performance expectations

- CPU single-thread: roughly 2 - 3x slower than element-based assembly
  (the loop is at the finest granularity).
- CPU multi-thread: comparable to node-based.
- GPU: comparable to node-based (one thread per DOF rather than per node).
- Matrix-free `K*v`: roughly 5 - 10x faster than rebuilding the matrix,
  since nothing is stored.

# When to use this assembler

Good fit:

- Matrix-free Krylov solvers (GMRES, CG).
- Very large problems (> ~1M DOFs where storing K is prohibitive).
- Contact mechanics (normal / tangential DOF operations).
- Mixed methods with different DOF counts per field.

Poor fit:

- Forming an explicit `K` (element-based is faster).
- Direct solvers (they need the explicit matrix anyway).
- Small problems (< ~10k DOFs, where loop overhead dominates).

# Example

```julia
# ... `mesh`, `elements`, `dof_handler = create_elements!(...)`
material = LinearElastic(E=210e9, ν=0.3)
kernel   = ContinuumKernel(
    ContinuumFormulation{ThreeDimensional}(),
    material,
    Displacement{3}(),
)
assembler = DOFBasedCOOAssembler()
cache      = create_cache(assembler, elements, dof_handler, mesh, kernel)
assemble!(cache, assembler, mesh)          # zero-alloc after warmup
K, f = extract_system(cache)
y = similar(f)
apply_K!(y, cache, assembler, mesh, x)    # matrix-free matvec
```
=#

using SparseArrays
using Tensors

using ..JuliaFEM: DOFConnectivity, DOFElementConnection
using ..JuliaFEM: local_dof_count, basis_type
using ..JuliaFEM: create_element_cache, create_geometry_cache, create_material_cache
using ..JuliaFEM: update_geometry_cache!, update_element_cache!
using ..JuliaFEM: GeometryCache, AssemblyMaterialWorkspace
using ..JuliaFEM: create_zero_field, material_field_type, material_state_type
using ..JuliaFEM: reference_fields
using ..JuliaFEM: local_dof_layout, DOFLayoutEntry, entity_local, component
using ..JuliaFEM: AbstractKernel, HeatKernel, ElementWiseScalarDiffusion
using ..JuliaFEM: prepare_dof_based_material_workspace!
using ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry, evaluate_mass_entry,
    compute_internal_force_value, geometry_eltype
using ..JuliaFEM: GlobalMaterialCache
using ..JuliaFEM: PartitionPackedLayout, expand_packed_to_global!

"""
    DOFBasedCOOCache

Pre-allocated cache for DOF-based COO assembly.

Implements the element-as-template assembly strategy:

* Pass 1 (element loop): for each element, fill its per-element
  geometry / element / material caches and tangent buffer.
* Pass 2 (DOF loop): walk the DOF→elements connectivity and use
  the compile-time `local_dof_layout(E)` table — generated once for
  the element template `E` — to decode `(field, entity, component)`
  for every local DOF without runtime arithmetic.

After warmup, both passes are zero-allocation.

# Type parameters
- `T`: topology type (e.g. `Hexahedron{8}`)
- `B`: basis type (e.g. `Lagrange{1}`)
- `IPS`: integration point set type
- `E`: concrete element type (provides `local_dof_layout(E)`)
- `Buf`: per-quadrature-point buffer element type for the kernel
  (`qpoint_buffer_eltype(kernel)`); e.g. `SymmetricTensor{4,3,Float64,36}`
  for `ContinuumKernel`
- `FieldType`: NamedTuple type of material fields (e.g. `(σ, 𝔻)`)
- `StateType`: NamedTuple type of material state (e.g. `()`)
- `KS`: kernel column storage ([`UniformKernelColumn`](@ref) or [`PerElementKernelColumn`](@ref))

# Fields
- `I, J, V`: COO triplets (rows, cols, values)
- `f`: global force vector
- `counter`: current number of triplets written
- `capacity`: maximum number of triplets
- `dof_connectivity`: inverse map DOF → elements touching it
- `elements`: concrete-typed element vector
- `ndofs`: total number of global DOFs
- `fields_ref, empty_state, zero_field`: typed reference snapshot from the
  probe kernel (Pass 1 overwrites per IP when using per-element kernels)
- `kernel_column`: uniform or per-element kernel storage (no heap churn in Pass 1)
- `element_caches, material_workspaces`:
  per-element caches built in Pass 1 and consumed in Pass 2.
- `X_batch, ∇N_batch, detJ_w_batch`: contiguous SoA backing storage for
  geometry across all elements:
  * `X_batch::Matrix{Vec{3,Float64}}` of shape `(max_nnodes, n_elems)` —
    node coordinates; each element's are `X_batch[:, eid]`.
  * `∇N_batch::Array{Vec{3,Float64}, 3}` of shape
    `(n_ips, max_nnodes, n_elems)` — physical gradients;
    `∇N_batch[:, :, eid]` is one element's `∇N`.
  * `detJ_w_batch::Matrix{Float64}` of shape `(n_ips, n_elems)` —
    `detJ * weight`; `detJ_w_batch[:, eid]` is one element's.
  All three are dense, contiguous, and trivially GPU-uploadable (one
  `cudaMemcpy` each); they replace the old per-element `Vector{Vector}` /
  `Vector{Matrix}` patchwork that scattered geometry across the heap.
- `geometry_caches::Vector{GC}`: thin per-element wrappers — each is a
  `GeometryCache{...}` holding three `view(...)` SubArrays into the
  batched storage above. Built once in the constructor; zero per-call
  allocation thereafter. Pass 1 / Pass 2 read and write through these
  views as if they were per-element heap arrays (same `.X[i]`,
  `.∇N_data[q,k]`, `.detJ_w[q]` API).
- `qp_buffers::Matrix{Buf}` of shape `(n_ips, n_elems)`: contiguous
  storage (column-major: each element occupies one column), so a single
  element's per-IP buffer is `qp_buffers[:, eid]`. Trivially GPU-uploadable
  and cache-friendly for the per-element streaming pattern in Pass 1
  and the per-IP integration pattern inside `evaluate_entry`.
"""
mutable struct DOFBasedCOOCache{T<:AbstractTopology,B<:AbstractBasis,IPS,E<:AbstractElement,GC<:GeometryCache,Buf,FieldType<:NamedTuple,StateType<:NamedTuple,KS}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    f::Vector{Float64}
    counter::Int
    capacity::Int
    dof_connectivity::DOFConnectivity
    # DOF handler reference — stored as the *abstract* `DOFHandler`
    # type so the cache's parameter list stays compact (the handler
    # is not on any hot path; surface loads and a few BC helpers are
    # the only callers, and they happily eat one virtual dispatch).
    dof_handler::DOFHandler
    # Pre-resolved field-1 starts vector — concrete `Vector{Int}` so
    # surface-load / preconditioner hot paths can avoid the abstract
    # field access on `dof_handler.field_starts`. Filled at cache
    # construction; mirrors `dof_handler.field_starts[1]`.
    field_starts1::Vector{Int}
    elements::Vector{E}
    ndofs::Int
    fields_ref::FieldType
    empty_state::StateType
    zero_field::FieldType
    kernel_column::KS
    element_caches::Vector{ElementCache{T,B,IPS}}
    # SoA backing storage for geometry across all elements.
    X_batch::Matrix{Vec{3,Float64}}            # (max_nnodes, n_elems)
    N_batch::Array{Float64, 3}                 # (n_ips, max_nnodes, n_elems)
    ∇N_batch::Array{Vec{3,Float64}, 3}         # (n_ips, max_nnodes, n_elems)
    detJ_w_batch::Matrix{Float64}              # (n_ips, n_elems)
    # Per-element view wrappers — each `geometry_caches[eid]` is a
    # `GeometryCache{...}` of four `SubArray`s into the batches above.
    geometry_caches::Vector{GC}
    material_workspaces::Vector{AssemblyMaterialWorkspace{FieldType, StateType}}
    qp_buffers::Matrix{Buf}
end

"""
    DOFBasedCOOCache(elements, dof_handler, mesh, kernel::AbstractKernel)
    DOFBasedCOOCache(elements, dof_handler, mesh, kernels::AbstractVector{<:AbstractKernel})
    DOFBasedCOOCache(elements, dof_handler, mesh, kernel_column)

Create cache for DOF-based assembly.

The single-kernel form wraps `kernel` in a [`UniformKernelColumn`](@ref) (no
extra per-element storage). The vector form builds a
[`PerElementKernelColumn`](@ref): `length(kernels)` must match
`length(elements)`, and all entries must be pairwise compatible (see
[`assert_homogeneous_dof_based_kernel_column!`](@ref)).

# Note
Requires elements created with `create_elements!()` (have `.dof_indices` assigned).
"""
function DOFBasedCOOCache(
    elements::Vector{E},
    dof_handler::DOFHandler,
    mesh::AbstractMesh,
    kernel::AbstractKernel,
) where {E<:AbstractElement}
    return DOFBasedCOOCache(elements, dof_handler, mesh, UniformKernelColumn(kernel))
end

function DOFBasedCOOCache(
    elements::Vector{E},
    dof_handler::DOFHandler,
    mesh::AbstractMesh,
    kernels::AbstractVector{K},
) where {E<:AbstractElement,K<:AbstractKernel}
    n_elems = length(elements)
    length(kernels) == n_elems || throw(DimensionMismatch(
        "per-element kernels: length(kernels)=$(length(kernels)) must match nelements=$n_elems",
    ))
    vec = Vector{K}(undef, n_elems)
    @inbounds for i in 1:n_elems
        vec[i] = kernels[i]
    end
    assert_homogeneous_dof_based_kernel_column!(vec)
    return DOFBasedCOOCache(elements, dof_handler, mesh, PerElementKernelColumn(vec))
end

function DOFBasedCOOCache(
    elements::Vector{E},
    dof_handler::DOFHandler,
    mesh::AbstractMesh,
    kernel_column::KS,
) where {E<:AbstractElement,KS}
    # Inverse map DOF → elements is built once by `create_elements!` and
    # stored on the handler. The handler initializes it with an empty
    # `DOFConnectivity()` placeholder, so we cross-check against
    # `total_dofs` to catch both "forgot create_elements!" and stale
    # connectivity left over from a different element list.
    dof_connectivity = dof_handler.dof_connectivity
    if dof_connectivity.n_total_dofs != dof_handler.total_dofs
        error("DOF connectivity not built (or stale). Call create_elements! on this DOFHandler before assembly.")
    end

    ndofs = dof_connectivity.n_total_dofs
    n_elems = length(elements)

    probe_k = prototype_kernel(kernel_column)
    if probe_k isa HeatKernel && probe_k.material isa ElementWiseScalarDiffusion
        λlen = length(probe_k.material.λ_by_elem)
        λlen == n_elems || throw(
            ArgumentError(
                "ElementWiseScalarDiffusion: length(λ_by_elem)=$λlen must match nelements=$n_elems",
            ),
        )
    end

    # Triplet capacity: each DOF row touches ≤ max_connections elements
    # and each element contributes its local DOF count entries to that row.
    avg_local_dofs = sum(local_dof_count(elem) for elem in elements) / n_elems
    entries_per_dof = dof_connectivity.max_connections * avg_local_dofs
    estimated_triplets = Int(ceil(1.2 * ndofs * entries_per_dof))

    I = Vector{Int}(undef, estimated_triplets)
    J = Vector{Int}(undef, estimated_triplets)
    V = Vector{Float64}(undef, estimated_triplets)
    f = zeros(Float64, ndofs)
    counter = 0

    # Local DOF count can exceed `nnodes * dofs_per_node` for facet-based fields.
    max_local_dofs_elem = maximum(local_dof_count(e) for e in elements)

    # Probe a temporary element cache to extract concrete cache type parameters
    # (T, B, IPS) and the integration-point count. The probe itself is not
    # stored on the cache; we keep one cache per element instead.
    elem_basis = basis_type(elements[1])()
    probe_cache = create_element_cache(
        mesh,
        probe_k;
        max_local_dofs = max_local_dofs_elem,
        basis = elem_basis,
    )
    ProbeT = typeof(probe_cache)
    T = ProbeT.parameters[1]
    B = ProbeT.parameters[2]
    IPS = ProbeT.parameters[3]
    n_ips = length(probe_cache.ips)

    MeshType = typeof(mesh)
    max_nnodes = MeshType.parameters[1]::Int

    # Typed reference snapshot from the probe kernel (Pass 1 may overwrite
    # per element when using a per-element kernel column).
    fields_ref, empty_state = reference_fields(probe_k)
    FieldType = typeof(fields_ref)
    StateType = typeof(empty_state)
    zero_field = fields_ref     # stateless materials ⇒ reference == zero state

    # Per-quadrature-point buffer element type is *kernel-defined* via
    # the `qpoint_buffer_eltype` trait. This is the only place the cache
    # learns what the kernel needs cached per IP — everything else uses
    # the trait/contract surface.
    Buf = qpoint_buffer_eltype(probe_k)

    # Per-element scratch (built in Pass 1, consumed in Pass 2).
    # `qp_buffers` is a contiguous (n_ips, n_elems) matrix — column-major,
    # so each element's per-IP buffer is the column `qp_buffers[:, eid]`.
    element_caches = Vector{ElementCache{T,B,IPS}}(undef, n_elems)
    material_workspaces = Vector{AssemblyMaterialWorkspace{FieldType, StateType}}(undef, n_elems)
    qp_buffers = Matrix{Buf}(undef, n_ips, n_elems)

    # SoA backing for geometry across all elements. One contiguous slab
    # per quantity, sliced by `view(..., eid)` per element. Replaces the
    # old per-element `Vector{Vec}` / `Matrix{Vec}` allocations.
    X_batch       = Matrix{Vec{3,Float64}}(undef, max_nnodes, n_elems)
    N_batch       = Array{Float64, 3}(undef, n_ips, max_nnodes, n_elems)
    ∇N_batch      = Array{Vec{3,Float64}, 3}(undef, n_ips, max_nnodes, n_elems)
    detJ_w_batch  = Matrix{Float64}(undef, n_ips, n_elems)
    fill!(X_batch, zero(Vec{3,Float64}))
    fill!(N_batch, 0.0)
    fill!(∇N_batch, zero(Vec{3,Float64}))
    fill!(detJ_w_batch, 0.0)

    # Build per-element view wrappers once. The `GeometryCache` here is
    # the *view-backed* parameterization — four `SubArray`s pointing
    # into the SoA batches above. From the kernel's perspective these
    # behave identically to the legacy heap-owned `GeometryCache`s.
    probe_geom = GeometryCache(
        view(X_batch, :, 1),
        view(N_batch, :, :, 1),
        view(∇N_batch, :, :, 1),
        view(detJ_w_batch, :, 1),
    )
    GC = typeof(probe_geom)
    geometry_caches = Vector{GC}(undef, n_elems)
    geometry_caches[1] = probe_geom

    for eid in 1:n_elems
        element_caches[eid] = create_element_cache(
            mesh,
            probe_k;
            max_local_dofs = max_local_dofs_elem,
            basis = elem_basis,
        )
        if eid != 1
            geometry_caches[eid] = GeometryCache(
                view(X_batch, :, eid),
                view(N_batch, :, :, eid),
                view(∇N_batch, :, :, eid),
                view(detJ_w_batch, :, eid),
            )::GC
        end
        # Per-element material workspace, allocated directly from the
        # kernel-defined `(fields_ref, empty_state)` pair so multi-material
        # kernels (e.g. `ThermoElasticKernel`) Just Work.
        fields_vec = Vector{FieldType}(undef, n_ips)
        states_vec = Vector{StateType}(undef, n_ips)
        @inbounds for q in 1:n_ips
            fields_vec[q] = fields_ref
            states_vec[q] = empty_state
        end
        material_workspaces[eid] = AssemblyMaterialWorkspace(fields_vec, states_vec)::AssemblyMaterialWorkspace{FieldType, StateType}
    end

    field_starts1 = dof_handler.field_starts[1]::Vector{Int}

    return DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS}(
        I, J, V, f, counter, estimated_triplets,
        dof_connectivity, dof_handler, field_starts1, elements, ndofs,
        fields_ref, empty_state, zero_field,
        kernel_column,
        element_caches,
        X_batch, N_batch, ∇N_batch, detJ_w_batch,
        geometry_caches, material_workspaces, qp_buffers,
    )
end

@inline kernel_at(cache::DOFBasedCOOCache, eid::Int) = kernel_at(cache.kernel_column, eid)

"""
    reset!(cache::DOFBasedCOOCache)

Reset cache for new assembly.

Only `cache.I[1:cache.counter]`, `cache.J[1:cache.counter]`, and
`cache.V[1:cache.counter]` are semantically valid after assembly. Resetting
zeros that written prefix for easier debugging, zeros the force vector, and
sets `counter` back to zero.
"""
function reset!(cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS}) where {T,B,IPS,E,GC,Buf,FieldType,StateType,KS}
    @inbounds for k in 1:cache.counter
        cache.I[k] = 0
        cache.J[k] = 0
        cache.V[k] = 0.0
    end
    fill!(cache.f, 0.0)
    cache.counter = 0
    return nothing
end

"""
    assemble!(cache, assembler, mesh; kwargs...) -> Nothing
    assemble!(cache, assembler, kernel, mesh; kwargs...) -> Nothing

Assemble the global system using DOF-based traversal built on the
"element-as-template" idea. The primary entry point is the three-argument
form: the cache carries a [`UniformKernelColumn`](@ref) or
[`PerElementKernelColumn`](@ref) built at construction time, so Pass 1 / Pass 2
stay allocation-free without threading a kernel object through every call.

The four-argument form ignores `kernel` and exists for backward compatibility
only.

# Nonlinear continuum keywords

Optional keywords are forwarded to [`_prepare_caches!`](@ref): `configuration`
(global displacement `Vector{Float64}`), `global_material_cache` (for
[`StatefulStrainDependent`](@ref) solids), and `Δt`. When omitted, behavior
matches the previous linear / reference Pass~1.

# Algorithm

Two passes, both zero-allocation after warmup:

```text
# Pass 1 — element loop (build per-element scratch)
for elem in 1:n_elems
    update_element_cache!     (DOF mapping, displacements)
    update_geometry_cache!    (coords, ∇N, detJ·w)
    update_material_state!    (σ, 𝔻 at every IP)
    update_qpoint_buffer!     (kernel-specific extract → qp buffer)
end

# Pass 2 — DOF loop (one row per DOF), driven by local_dof_layout(E)
layout = local_dof_layout(E)              # compile-time NTuple{N, DOFLayoutEntry}
for dof_i in 1:ndofs
    for conn in dof_connectivity.dof_to_elements[dof_i]
        e_id, l_i = conn.elem_id, conn.local_dof_idx
        for l_j in 1:N
            K_ij = evaluate_entry(kernel_at(cache, e_id), …)
            scatter_entry!(cache, K_ij, dof_i_global, dof_j_global)
        end
    end
end
```
"""
function assemble!(
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    reset!(cache)
    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    # ========================================================================
    # PASS 2 — DOF loop: one matrix row per DOF, driven by element template
    # ========================================================================
    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers
    ndofs            = cache.ndofs

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    # `local_dof_layout(E)` is a `@generated` compile-time `NTuple{N,
    # DOFLayoutEntry}` describing each local DOF as (field, entity,
    # component).  Looking up `layout[local_i]` here replaces the
    # runtime `div`/`mod` decoding the older implementation used.
    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for dof_i in 1:ndofs
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i = layout[local_i]

            dofs_elem    = element_cache.dofs
            dof_i_global = dofs_elem[local_i]

            k_e = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = dofs_elem[local_j]
                entry_j      = layout[local_j]

                # Kernel-defined microkernel: returns the scalar K[i,j]
                # contribution for this (i,j) pair on this element.
                K_ij = evaluate_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                    Int(elem_id_val),
                )

                scatter_entry!(cache, K_ij, dof_i_global, dof_j_global)
            end
        end
    end

    return nothing
end

@inline function assemble!(
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:assemble!)
    return assemble!(
        cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

@inline function _dof_based_fill_qpoint_buffer!(
    buffer,
    workspace::AssemblyMaterialWorkspace,
    kernel::HeatKernel{Th, ElementWiseScalarDiffusion, F},
    eid::Int,
) where {Th,F}
    return update_qpoint_buffer!(buffer, workspace, kernel, eid)
end

@inline function _dof_based_fill_qpoint_buffer!(
    buffer,
    workspace::AssemblyMaterialWorkspace,
    kernel::AbstractKernel,
    ::Int,
)
    return update_qpoint_buffer!(buffer, workspace, kernel)
end

"""
    _prepare_caches!(cache::DOFBasedCOOCache, mesh::AbstractMesh; kwargs...) -> Nothing

Pass 1 of the DOF-based assembly: fill per-element scratch (element
cache, geometry cache, material workspace, kernel-specific qp buffer) so
that Pass 2 can read each element's data by id without recomputation.

The volume kernel for element `eid` is [`kernel_at`](@ref)`(cache, eid)` from the
cache's kernel column (uniform or per-element).

# Nonlinear continuum mechanics

Optional keywords (defaults preserve the previous linear / reference
Pass~1):

- `configuration`: global displacement vector (same length as
  `cache.ndofs`) forwarded to each kernel's
  [`prepare_dof_based_material_workspace!`](@ref) (e.g. strain-dependent
  solids). When `nothing`, vertex displacements are treated as zero where
  the kernel reads them.
- `global_material_cache`: required when a kernel's Pass~1 material path
  needs persistent IP state (e.g. [`StatefulStrainDependent`](@ref) solids);
  ignored otherwise.
- `Δt`: time increment forwarded to Pass~1 material updates.

[`assemble!`](@ref), [`apply_K!`](@ref), and related entry points forward
these keywords to Pass~1.

Allocation-free after warmup.
"""
@inline function _prepare_caches!(
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    elements             = cache.elements

    element_caches       = cache.element_caches
    geometry_caches      = cache.geometry_caches
    material_workspaces  = cache.material_workspaces
    qp_buffers           = cache.qp_buffers

    n_elems = length(elements)

    @inbounds for eid in 1:n_elems
        k_e                 = kernel_at(cache, eid)
        element_cache       = element_caches[eid]
        geometry_cache      = geometry_caches[eid]
        material_workspace  = material_workspaces[eid]
        qp_buffer           = view(qp_buffers, :, eid)

        reset!(element_cache)
        reset!(geometry_cache)
        reset!(material_workspace)

        elem_inst    = elements[eid]
        dof_indices  = elem_inst.dof_indices
        dofs_storage = element_cache.dofs
        @inbounds for k in eachindex(dof_indices)
            dofs_storage[k] = Int(dof_indices[k])
        end

        update_geometry_cache!(geometry_cache, element_cache, eid, mesh)

        prepare_dof_based_material_workspace!(
            k_e,
            material_workspace,
            geometry_cache,
            element_cache,
            Int(eid),
            configuration,
            global_material_cache,
            Δt,
            E,
        )

        _dof_based_fill_qpoint_buffer!(qp_buffer, material_workspace, k_e, eid)
    end
    return nothing
end

# Backward-compatible entry point (kernel ignored — use cache.kernel_column).
@inline function _prepare_caches!(
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:_prepare_caches!)
    return _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_K!(y, cache, assembler, mesh, x) -> y
    apply_K!(y, cache, assembler, kernel, mesh, x) -> y

Matrix-free product `y = K * x`, computed by the same DOF-row traversal
as `assemble!` but accumulating into `y` instead of scattering into COO
triplets.

The five-argument form is primary; the six-argument form ignores `kernel`.

The matrix `K` is never built. For each row, we walk the touching
elements, ask the kernel for the scalar `K[dof_i, dof_j]` via
`evaluate_entry`, and immediately accumulate `K_ij * x[dof_j]` into a
local scalar — written to `y[dof_i]` exactly once. Because each `dof_i`
is owned by exactly one outer-loop iteration, no atomics are needed even
when DOFs are shared across elements.

# Algorithm

```text
_prepare_caches!(cache, mesh; configuration=…)   # Pass 1, shared

for dof_i in 1:ndofs
    yi = 0.0
    for conn in dof_connectivity.dof_to_elements[dof_i]
        for local_j in 1:N
            K_ij = evaluate_entry(kernel_at(cache, e_id), …)
            yi  += K_ij * x[dof_j_global]
        end
    end
    y[dof_i] = yi
end
```

# Performance

Allocation-free after warmup; the inner loop is the same `evaluate_entry`
call as `assemble!`, so the work per `(i, j)` is identical. The matrix
is never materialised, so memory traffic is `O(ndofs)` instead of
`O(nnz(K))`.

# Use

Wrap as a linear operator for a Krylov solve:

```julia
using LinearAlgebra, Krylov
op = LinearOperator(Float64, n, n, true, true,
                    (y, x) -> apply_K!(y, cache, asm, mesh, x))
u, _ = cg(op, b)
```
"""
function apply_K!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    @assert length(y) == cache.ndofs "y has length $(length(y)); expected $(cache.ndofs)"
    @assert length(x) == cache.ndofs "x has length $(length(x)); expected $(cache.ndofs)"

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers
    ndofs            = cache.ndofs

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for dof_i in 1:ndofs
        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i      = layout[local_i]
            dofs_elem    = element_cache.dofs
            k_e          = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = dofs_elem[local_j]
                entry_j      = layout[local_j]

                K_ij = evaluate_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                    Int(elem_id_val),
                )

                yi += K_ij * x[Int(dof_j_global)]
            end
        end

        y[dof_i] = yi
    end

    return y
end

@inline function apply_K!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:apply_K!)
    return apply_K!(
        y, cache, assembler, mesh, x;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    equilibrium_residual!(
        r, f, Ku_work, cache, assembler, mesh, u,
    ) -> r

Linear equilibrium residual ``r = f - K u`` using the same stiffness tangent as
[`assemble!`](@ref) / [`apply_K!`](@ref).

Writes `Ku_work = K * u` via [`apply_K!`](@ref) into the caller-provided scratch
`Ku_work`, then sets `r[i] = f[i] - Ku_work[i]` for all `i`. All vectors must
have length `cache.ndofs`. Allocation-free after warmup when `Ku_work` is
preallocated (no heap traffic in the subtraction loop).

For ``r = f_{\\mathrm{ext}} - f_{\\mathrm{int}}(u)`` with Cauchy stress assembly,
see [`nonlinear_equilibrium_residual!`](@ref).
"""
function equilibrium_residual!(
    r::AbstractVector{Float64},
    f::AbstractVector{Float64},
    Ku_work::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    u::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    nd = cache.ndofs
    if length(r) != nd || length(f) != nd || length(Ku_work) != nd || length(u) != nd
        throw(DimensionMismatch("equilibrium_residual!: expected length $nd for r, f, Ku_work, u"))
    end
    apply_K!(
        Ku_work, cache, assembler, mesh, u;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    @inbounds for i in 1:nd
        r[i] = f[i] - Ku_work[i]
    end
    return r
end

"""
    assemble_internal_force!(
        f, cache, assembler, mesh;
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> f
    assemble_internal_force!(
        f, cache, assembler, kernel, mesh; kwargs...,
    ) -> f

Zero `f`, run the same Pass~1 as [`assemble!`](@ref) / [`apply_K!`](@ref), then
accumulate the **nonlinear internal force** from Cauchy stress ``σ`` at each
quadrature point (Galerkin discretization of ``∫ σ : ∇(N_i e_α) \\, dV`` for
displacement test functions).

The cache's material field type must include `:σ` (continuum solids with
`(σ, 𝔻)` per IP). Other physics kernels raise [`ArgumentError`](@ref).

Optional keywords are forwarded to [`_prepare_caches!`](@ref). When
`configuration` is passed, [`StatelessConstantTangent`](@ref) solids (e.g.
[`LinearElastic`](@ref)) recompute ``(σ, 𝔻)`` from the current nodal
displacements so ``σ`` tracks ``u`` while the tangent remains constitutive.

The six-argument form ignores `kernel` (backward compatibility).

Allocation-free after warmup aside from clearing `f`.
"""
function assemble_internal_force!(
    f::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    hasfield(FieldType, :σ) ||
        throw(ArgumentError(
            "assemble_internal_force! requires material workspace fields with `:σ` " *
            "(continuum mechanics); got $FieldType.",
        ))
    nd = cache.ndofs
    length(f) == nd ||
        throw(DimensionMismatch(
            "assemble_internal_force!: length(f)=$(length(f)) != ndofs $nd",
        ))
    fill!(f, 0.0)
    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    elements            = cache.elements
    element_caches      = cache.element_caches
    geometry_caches     = cache.geometry_caches
    material_workspaces = cache.material_workspaces
    layout              = local_dof_layout(E)
    ndofs_elem          = length(layout)
    n_elems             = length(elements)
    @inbounds for eid in 1:n_elems
        element_cache  = element_caches[eid]
        geometry_cache = geometry_caches[eid]
        material_ws    = material_workspaces[eid]
        F              = geometry_eltype(geometry_cache)
        n_ips          = length(geometry_cache.detJ_w)
        fields_vec     = material_ws.fields
        @inbounds for local_i in 1:ndofs_elem
            dof_g = Int(element_cache.dofs[local_i])
            entry_i = layout[local_i]
            node_i = entity_local(entry_i)
            comp_i = component(entry_i)
            acc = zero(F)
            @inbounds for q in 1:n_ips
                σ = fields_vec[q].σ
                ∇N_i = geometry_cache.∇N_data[q, node_i]
                detJw = geometry_cache.detJ_w[q]
                acc += compute_internal_force_value(∇N_i, σ, comp_i) * detJw
            end
            f[dof_g] += Float64(acc)
        end
    end
    return f
end

@inline function assemble_internal_force!(
    f::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:assemble_internal_force!)
    return assemble_internal_force!(
        f, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    nonlinear_equilibrium_residual!(
        r, f_ext, f_work, cache, assembler, mesh, u; kwargs...,
    ) -> r

Nonlinear equilibrium residual ``r = f_{\\mathrm{ext}} - f_{\\mathrm{int}}(u)``
with ``f_{\\mathrm{int}}`` from [`assemble_internal_force!`](@ref) at
`configuration = u`.

Scratch `f_work` must have length `cache.ndofs` and is overwritten.
Optional `global_material_cache` and `Δt` are forwarded to Pass~1.

The eight-argument form ignores `kernel`.
"""
function nonlinear_equilibrium_residual!(
    r::AbstractVector{Float64},
    f_ext::AbstractVector{Float64},
    f_work::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    u::AbstractVector{Float64};
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    nd = cache.ndofs
    if length(r) != nd || length(f_ext) != nd || length(f_work) != nd || length(u) != nd
        throw(DimensionMismatch(
            "nonlinear_equilibrium_residual!: expected length $nd for r, f_ext, f_work, u",
        ))
    end
    assemble_internal_force!(
        f_work, cache, assembler, mesh;
        configuration = u,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    @inbounds for i in 1:nd
        r[i] = f_ext[i] - f_work[i]
    end
    return r
end

@inline function nonlinear_equilibrium_residual!(
    r::AbstractVector{Float64},
    f_ext::AbstractVector{Float64},
    f_work::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    u::AbstractVector{Float64};
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:nonlinear_equilibrium_residual!)
    return nonlinear_equilibrium_residual!(
        r, f_ext, f_work, cache, assembler, mesh, u;
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_K_masked_rows!(y, keep_rows, cache, assembler, mesh, x) -> y
    apply_K_masked_rows!(y, keep_rows, cache, assembler, kernel, mesh, x) -> y

Matrix-free product [`apply_K!`](@ref)`(y, …, x)` followed by a row mask:
for each index `i`, if `keep_rows[i]` is false then `y[i] = 0`.

If `x` holds the full global state (or a ghost-filled replica of it), each
rank can keep only its owned rows; summing those vectors over a disjoint row
partition recovers the global `K x`.

`length(keep_rows)` must equal `cache.ndofs`. Allocation-free after warmup
(same hot path as [`apply_K!`](@ref) plus a linear mask pass).

The seven-argument form ignores `kernel` (backward compatibility).
"""
function apply_K_masked_rows!(
    y::AbstractVector{Float64},
    keep_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    ndofs = cache.ndofs
    length(keep_rows) == ndofs ||
        throw(DimensionMismatch("keep_rows length $(length(keep_rows)); expected $ndofs"))

    apply_K!(
        y, cache, assembler, mesh, x;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    @inbounds for i in 1:ndofs
        keep_rows[i] || (y[i] = 0.0)
    end
    return y
end

@inline function apply_K_masked_rows!(
    y::AbstractVector{Float64},
    keep_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_K_masked_rows!)
    return apply_K_masked_rows!(
        y, keep_rows, cache, assembler, mesh, x;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_K_owned_rows!(y, owned_rows, cache, assembler, mesh, x) -> y
    apply_K_owned_rows!(y, owned_rows, cache, assembler, kernel, mesh, x) -> y

Matrix-free matvec like [`apply_K!`](@ref), but only **computes** rows `i` with
`owned_rows[i] == true`. Calls [`fill!`](@ref)`(y, 0)` first, fills those rows,
and leaves all other entries at zero.

Numerically agrees with [`apply_K!`](@ref) on owned rows for the same `x`.
When `count(owned_rows) ≪ ndofs`, this avoids touching non-owned rows (unlike
[`apply_K_masked_rows!`](@ref), which runs a full matvec then masks).

`length(owned_rows)` must equal `cache.ndofs`. Allocation-free after warmup.

The seven-argument form ignores `kernel` (backward compatibility).
"""
function apply_K_owned_rows!(
    y::AbstractVector{Float64},
    owned_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    @assert length(y) == cache.ndofs "y has length $(length(y)); expected $(cache.ndofs)"
    @assert length(x) == cache.ndofs "x has length $(length(x)); expected $(cache.ndofs)"

    ndofs = cache.ndofs
    length(owned_rows) == ndofs ||
        throw(DimensionMismatch("owned_rows length $(length(owned_rows)); expected $ndofs"))

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    fill!(y, 0.0)

    @inbounds for dof_i in 1:ndofs
        owned_rows[dof_i] || continue

        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i      = layout[local_i]
            dofs_elem    = element_cache.dofs
            k_e          = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = dofs_elem[local_j]
                entry_j      = layout[local_j]

                K_ij = evaluate_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                    Int(elem_id_val),
                )

                yi += K_ij * x[Int(dof_j_global)]
            end
        end

        y[dof_i] = yi
    end

    return y
end

@inline function apply_K_owned_rows!(
    y::AbstractVector{Float64},
    owned_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_K_owned_rows!)
    return apply_K_owned_rows!(
        y, owned_rows, cache, assembler, mesh, x;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_f_int_owned_rows!(
        y, owned_rows, cache, assembler, mesh;
        configuration = nothing, global_material_cache = nothing, Δt = 0.0,
    ) -> y
    apply_f_int_owned_rows!(
        y, owned_rows, cache, assembler, kernel, mesh; kwargs...,
    ) -> y

Like [`assemble_internal_force!`](@ref), but only fills rows `i` with
`owned_rows[i] == true` (others stay zero). Uses the same Pass~1 and Cauchy
stress contraction as the full vector assembly.

`length(owned_rows) == cache.ndofs`. Optional keywords match [`apply_K_owned_rows!`](@ref).

The seven-argument form ignores `kernel`.
"""
function apply_f_int_owned_rows!(
    y::AbstractVector{Float64},
    owned_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    hasfield(FieldType, :σ) ||
        throw(ArgumentError(
            "apply_f_int_owned_rows! requires material workspace fields with `:σ`; got $FieldType.",
        ))
    ndofs = cache.ndofs
    length(y) == ndofs ||
        throw(DimensionMismatch("y length $(length(y)); expected $ndofs"))
    length(owned_rows) == ndofs ||
        throw(DimensionMismatch("owned_rows length $(length(owned_rows)); expected $ndofs"))

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements            = cache.elements
    element_caches      = cache.element_caches
    geometry_caches     = cache.geometry_caches
    material_workspaces = cache.material_workspaces
    dof_connectivity    = cache.dof_connectivity
    dof_to_elements     = dof_connectivity.dof_to_elements
    layout              = local_dof_layout(E)
    ndofs_elem          = length(layout)

    fill!(y, 0.0)

    @inbounds for dof_i in 1:ndofs
        owned_rows[dof_i] || continue

        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            material_ws    = material_workspaces[elem_id_val]

            entry_i   = layout[local_i]
            node_i    = entity_local(entry_i)
            comp_i    = component(entry_i)
            F         = geometry_eltype(geometry_cache)
            n_ips     = length(geometry_cache.detJ_w)
            fields_vec = material_ws.fields
            acc = zero(F)
            @inbounds for q in 1:n_ips
                σ = fields_vec[q].σ
                ∇N_i = geometry_cache.∇N_data[q, node_i]
                detJw = geometry_cache.detJ_w[q]
                acc += compute_internal_force_value(∇N_i, σ, comp_i) * detJw
            end
            yi += Float64(acc)
        end

        y[dof_i] = yi
    end

    return y
end

@inline function apply_f_int_owned_rows!(
    y::AbstractVector{Float64},
    owned_rows::AbstractVector{Bool},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_f_int_owned_rows!)
    return apply_f_int_owned_rows!(
        y, owned_rows, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_f_int_owned_rows_from_packed!(
        Ap_owned, packed, work, layout, cache, assembler, mesh; kwargs...,
    ) -> Ap_owned
    apply_f_int_owned_rows_from_packed!(
        Ap_owned, packed, work, layout, cache, assembler, kernel, mesh; kwargs...,
    ) -> Ap_owned

Owned-row internal force in compact layout: `Ap_owned[k]` is the global row
`layout.packed_to_global[k]` for ``k \\in 1:\\texttt{layout.n\\_owned}``.

`work` is a scratch vector of length `cache.ndofs` used to
[`expand_packed_to_global!`](@ref) the patch displacement, then Pass~1 uses
`configuration \\equiv work` unless `configuration` is passed explicitly (full
length `cache.ndofs`).

The nine-argument form ignores `kernel`.
"""
function apply_f_int_owned_rows_from_packed!(
    Ap_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    hasfield(FieldType, :σ) ||
        throw(ArgumentError(
            "apply_f_int_owned_rows_from_packed! requires material workspace fields with `:σ`; got $FieldType.",
        ))
    ndofs = cache.ndofs
    layout.ndofs_global == ndofs ||
        throw(DimensionMismatch(
            "layout.ndofs_global $(layout.ndofs_global) != cache.ndofs $ndofs",
        ))
    no = layout.n_owned
    n_packed = layout.n_packed
    length(Ap_owned) == no ||
        throw(DimensionMismatch("Ap_owned length $(length(Ap_owned)), n_owned $no"))
    length(packed) ≥ n_packed ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n_packed"))
    length(work) == ndofs ||
        throw(DimensionMismatch("work length $(length(work)); expected $ndofs"))

    owned_rows = layout.owned_rows
    length(owned_rows) == ndofs ||
        throw(DimensionMismatch("owned_rows length $(length(owned_rows)); expected $ndofs"))

    p2g = layout.packed_to_global

    fill!(work, 0.0)
    expand_packed_to_global!(work, packed, layout)
    prep_cfg = configuration === nothing ? work : configuration

    _prepare_caches!(
        cache, mesh;
        configuration = prep_cfg,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    geometry_caches     = cache.geometry_caches
    material_workspaces = cache.material_workspaces
    dof_connectivity    = cache.dof_connectivity
    dof_to_elements     = dof_connectivity.dof_to_elements
    loc_layout          = local_dof_layout(E)

    @inbounds for k in 1:no
        dof_i = p2g[k]
        owned_rows[dof_i] ||
            throw(ArgumentError(
                "packed owned slot $k → global $dof_i not marked owned in layout",
            ))

        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            geometry_cache = geometry_caches[elem_id_val]
            material_ws    = material_workspaces[elem_id_val]

            entry_i    = loc_layout[local_i]
            node_i     = entity_local(entry_i)
            comp_i     = component(entry_i)
            F          = geometry_eltype(geometry_cache)
            n_ips      = length(geometry_cache.detJ_w)
            fields_vec = material_ws.fields
            acc = zero(F)
            @inbounds for q in 1:n_ips
                σ = fields_vec[q].σ
                ∇N_i = geometry_cache.∇N_data[q, node_i]
                detJw = geometry_cache.detJ_w[q]
                acc += compute_internal_force_value(∇N_i, σ, comp_i) * detJw
            end
            yi += Float64(acc)
        end

        Ap_owned[k] = yi
    end

    return Ap_owned
end

@inline function apply_f_int_owned_rows_from_packed!(
    Ap_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    work::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_f_int_owned_rows_from_packed!)
    return apply_f_int_owned_rows_from_packed!(
        Ap_owned, packed, work, layout, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_K_contributions!(y, cache, assembler, mesh, x, element_ids) -> y
    apply_K_contributions!(y, cache, assembler, kernel, mesh, x, element_ids) -> y

Add the stiffness matrix-vector contribution from the listed volume elements
only (1-based indices into `cache.elements`):

```text
y[dof_i] += Σ_{e ∈ element_ids} Σ_j K_ij(e) · x[dof_j]
```

Uses the same [`_prepare_caches!`](@ref) pass and [`evaluate_entry`](@ref)
calls as [`apply_K!`](@ref). Does **not** zero `y`; for disjoint element
sets that cover the mesh exactly once, callers typically [`fill!`](@ref)`(y,
0)` per partial vector then sum the partials to recover `K * x`.

Concurrent writes to the same `dof_i` from different threads or ranks
require a reduction strategy; this routine performs plain `+=` without
atomics.

The eight-argument form ignores `kernel` (backward compatibility).

# See also

- [`apply_K!`](@ref) — full product with one write per row (serial).
"""
function apply_K_contributions!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    x::AbstractVector{Float64},
    element_ids;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    @assert length(y) == cache.ndofs "y has length $(length(y)); expected $(cache.ndofs)"
    @assert length(x) == cache.ndofs "x has length $(length(x)); expected $(cache.ndofs)"

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements        = cache.elements
    element_caches  = cache.element_caches
    geometry_caches = cache.geometry_caches
    qp_buffers      = cache.qp_buffers
    n_elems         = length(elements)

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for elem_id_val in element_ids
        (1 <= elem_id_val <= n_elems) ||
            throw(ArgumentError("element id $elem_id_val out of range 1:$n_elems"))

        k_e            = kernel_at(cache, Int(elem_id_val))
        element        = elements[elem_id_val]::E
        element_cache  = element_caches[elem_id_val]
        geometry_cache = geometry_caches[elem_id_val]
        qp_buffer      = view(qp_buffers, :, elem_id_val)
        dofs_elem      = element_cache.dofs

        @inbounds for local_i in 1:ndofs_elem
            dof_i_global = Int(dofs_elem[local_i])
            entry_i      = layout[local_i]

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = Int(dofs_elem[local_j])
                entry_j      = layout[local_j]

                K_ij = evaluate_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                    Int(elem_id_val),
                )

                y[dof_i_global] += K_ij * x[dof_j_global]
            end
        end
    end

    return y
end

@inline function apply_K_contributions!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    x::AbstractVector{Float64},
    element_ids;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_K_contributions!)
    return apply_K_contributions!(
        y, cache, assembler, mesh, x, element_ids;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    apply_M!(y, cache, assembler, mesh, x) -> y
    apply_M!(y, cache, assembler, kernel, mesh, x) -> y

Matrix-free product `y = M * x` where `M` is the consistent mass
matrix (or, for `HeatKernel`, the heat-capacity matrix). Same DOF-row
traversal as `apply_K!`; only difference is that the inner kernel call
is `evaluate_mass_entry` instead of `evaluate_entry`, so the bilinear
form `(N_i, ρ N_j)` replaces `(B_i : C : B_j)`.

The five-argument form is primary; the six-argument form ignores `kernel`.

Kernels that don't override `evaluate_mass_entry` see the default
`= 0.0`, so `apply_M!` produces a structural-zero `y` and serves as a
documented opt-in: `ContinuumKernel(...; density = ρ)` /
`HeatKernel(...; heat_capacity = ρ·c_p)` switches it on.

Allocation-free; reuses the same `_prepare_caches!` Pass 1 as `apply_K!`
so the geometry / `N_data` / `qp_buffers` are populated once and used by
both operators when the caller does `apply_K!` then `apply_M!` (or vice
versa) on the same cache.

# Use — eigen / time stepping

Wrap as a `LinearOperator` to feed mass-matrix into eigen and transient
solvers without ever assembling `M`:

```julia
op_K = matrix_free_op(cache, asm, mesh)
op_M = (y, x) -> apply_M!(y, cache, asm, mesh, x)
```
"""
function apply_M!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    @assert length(y) == cache.ndofs "y has length $(length(y)); expected $(cache.ndofs)"
    @assert length(x) == cache.ndofs "x has length $(length(x)); expected $(cache.ndofs)"

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers
    ndofs            = cache.ndofs

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for dof_i in 1:ndofs
        yi                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i      = layout[local_i]
            dofs_elem    = element_cache.dofs
            k_e          = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = dofs_elem[local_j]
                entry_j      = layout[local_j]

                M_ij = evaluate_mass_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                )

                yi += M_ij * x[Int(dof_j_global)]
            end
        end

        y[dof_i] = yi
    end

    return y
end

@inline function apply_M!(
    y::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    x::AbstractVector{Float64};
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:apply_M!)
    return apply_M!(
        y, cache, assembler, mesh, x;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    assemble_M!(cache, assembler, mesh) -> Nothing
    assemble_M!(cache, assembler, kernel, mesh) -> Nothing

Assemble the consistent mass / heat-capacity matrix into the cache's
COO triplets via the same DOF-row traversal as `assemble!`, calling
`evaluate_mass_entry` instead of `evaluate_entry`. After this, the
caller pulls `M, _ = extract_system(cache)`.

The three-argument form is primary; the four-argument form ignores `kernel`.

Cache reuse: this overwrites the triplets from any prior
`assemble!`. The intended use pattern when both `K` and `M` are needed
is:

```julia
assemble!(cache, asm, mesh);   K, _ = extract_system(cache)
assemble_M!(cache, asm, mesh); M, _ = extract_system(cache)
```

Each `extract_system` builds an independent `SparseMatrixCSC`, so the
two matrices coexist after `M` is extracted; the cache then holds the
last assembled triplets only.

Allocation-free in the inner loop after warmup. Uses zero
`evaluate_mass_entry` calls' worth of work for kernels that don't
implement mass (the constant-`0` default folds at compile time), so
running `assemble_M!` on a static-only kernel is harmless and produces
a structural-zero `M`.
"""
function assemble_M!(
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    reset!(cache)
    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers
    ndofs            = cache.ndofs

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for dof_i in 1:ndofs
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            element_cache  = element_caches[elem_id_val]
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i = layout[local_i]

            dofs_elem    = element_cache.dofs
            dof_i_global = dofs_elem[local_i]
            k_e          = kernel_at(cache, Int(elem_id_val))

            @inbounds for local_j in 1:ndofs_elem
                dof_j_global = dofs_elem[local_j]
                entry_j      = layout[local_j]

                M_ij = evaluate_mass_entry(
                    k_e,
                    geometry_cache,
                    qp_buffer,
                    entry_i,
                    entry_j,
                )

                scatter_entry!(cache, M_ij, dof_i_global, dof_j_global)
            end
        end
    end

    return nothing
end

@inline function assemble_M!(
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:assemble_M!)
    return assemble_M!(
        cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    compute_diagonal!(d, cache, assembler, mesh) -> d
    compute_diagonal!(d, cache, assembler, kernel, mesh) -> d

Matrix-free extraction of `diag(K)` using the same DOF-row traversal
as `apply_K!` — same complexity (one element-row pass, one
`evaluate_entry` per touching element), no `K` materialisation.

The four-argument form ignores `kernel`.

The returned diagonal is the diagonal of the *unconstrained* operator.
For a Jacobi preconditioner consistent with `matrix_free_op(...; dirichlet=c)`
follow up with `apply_constraint_diag!(d, c)` (or just use
`JacobiPreconditioner(cache, asm, mesh; dirichlet=c)` which
chains the two).

Allocation-free after warmup (`d` is mutated in place).
"""
function compute_diagonal!(
    d::AbstractVector{Float64},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
    assembler::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType,KS}
    @assert length(d) == cache.ndofs "d has length $(length(d)); expected $(cache.ndofs)"

    _prepare_caches!(
        cache, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
    fill!(d, 0.0)

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers
    ndofs            = cache.ndofs

    dof_connectivity = cache.dof_connectivity
    dof_to_elements  = dof_connectivity.dof_to_elements

    layout     = local_dof_layout(E)

    @inbounds for dof_i in 1:ndofs
        di                = 0.0
        touching_elements = dof_to_elements[dof_i]
        n_conns           = length(touching_elements)

        @inbounds for conn_idx in 1:n_conns
            conn        = touching_elements[conn_idx]
            elem_id_val = elem_id(conn)
            local_i     = local_dof_idx(conn)

            element        = elements[elem_id_val]::E
            geometry_cache = geometry_caches[elem_id_val]
            qp_buffer      = view(qp_buffers, :, elem_id_val)

            entry_i = layout[local_i]
            k_e     = kernel_at(cache, Int(elem_id_val))

            # Diagonal entry only — same `evaluate_entry` call, j == i.
            di += evaluate_entry(
                k_e,
                geometry_cache,
                qp_buffer,
                entry_i,
                entry_i,
                Int(elem_id_val),
            )
        end

        d[dof_i] = di
    end

    return d
end

@inline function compute_diagonal!(
    d::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    assembler::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Float64 = 0.0,
)
    _depwarn_redundant_kernel_arg!(:compute_diagonal!)
    return compute_diagonal!(
        d, cache, assembler, mesh;
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    scatter_entry!(
        cache::DOFBasedCOOCache,
        value::Float64,
        dof_i::Int,
        dof_j::Int
    )

Scatter single matrix entry to COO triplets in place.

# Arguments
- `cache`: DOF-based COO cache
- `value`: Matrix entry value
- `dof_i`: Row DOF index
- `dof_j`: Column DOF index

# Zero-Allocation

Writes to pre-allocated triplet arrays, updates counter.
"""
@inline function scatter_entry!(
    cache::DOFBasedCOOCache,
    value::Float64,
    dof_i::Int,
    dof_j::Int,
)
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
function extract_system(cache::DOFBasedCOOCache)
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
    create_cache(assembler, elements, dof_handler, mesh, kernel::AbstractKernel)
    create_cache(assembler, elements, dof_handler, mesh, kernels::AbstractVector)

Create pre-allocated cache for DOF-based COO assembly.

The single-kernel form wraps `kernel` in a [`UniformKernelColumn`](@ref). The
vector form must have `length(kernels) == length(elements)` and is validated by
[`assert_homogeneous_dof_based_kernel_column!`](@ref).

Requires elements with assigned DOF indices (from `create_elements!`).

# Example

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ElemType = Element{Hexahedron{8}, Lagrange{1}, S}
elements, handler = create_elements!(mesh, ElemType)

kernel = ContinuumKernel(ContinuumFormulation{ThreeDimensional}(), material, Displacement{3}())

assembler = DOFBasedCOOAssembler()
cache = create_cache(assembler, elements, handler, mesh, kernel)

assemble!(cache, assembler, mesh)
K, f = extract_system(cache)
```
"""
function create_cache(
    assembler::DOFBasedCOOAssembler,
    elements::Vector{E},
    dof_handler::DOFHandler,
    mesh::AbstractMesh,
    kernel::AbstractKernel,
) where {E<:AbstractElement}
    return DOFBasedCOOCache(elements, dof_handler, mesh, kernel)
end

function create_cache(
    assembler::DOFBasedCOOAssembler,
    elements::Vector{E},
    dof_handler::DOFHandler,
    mesh::AbstractMesh,
    kernels::AbstractVector{K},
) where {E<:AbstractElement,K<:AbstractKernel}
    return DOFBasedCOOCache(elements, dof_handler, mesh, kernels)
end
