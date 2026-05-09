# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractMultiplyGhostLayout

Supertype for strategies that fill the matrix-free work buffer before
[`apply_K!`](@ref) runs. The default [`LocalMultiplyLayout`](@ref) copies
the global Krylov vector `x` unchanged; future MPI/GPU layouts may inject
ghost DOF values without changing the assembler kernel.
"""
abstract type AbstractMultiplyGhostLayout end

"""
    LocalMultiplyLayout <: AbstractMultiplyGhostLayout

Single-process layout: `prepare_multiply_workspace!` is [`copyto!`](@ref)
from `x` into `work`.
"""
struct LocalMultiplyLayout <: AbstractMultiplyGhostLayout end

"""
    prepare_multiply_workspace!(work, x, layout) -> work

Fill `work` from the Krylov vector `x` according to `layout`. For
[`LocalMultiplyLayout`](@ref), `work` and `x` must have the same length
and `work === x` is allowed (aliases).
"""
function prepare_multiply_workspace!(
    work::AbstractVector{Float64},
    x::AbstractVector{Float64},
    ::LocalMultiplyLayout,
)
    n = length(work)
    length(x) == n ||
        throw(DimensionMismatch("work length $(length(work)), x length $(length(x))"))
    if work !== x
        copyto!(work, x)
    end
    return work
end

"""
    ReferenceMaskMultiplyLayout(mask::BitVector) <: AbstractMultiplyGhostLayout

Copy only entries where `mask[i] == true` from `x` into `work` (`work[i] = x[i]`).
Entries with `mask[i] == false` are left untouched (must not be read by the
kernel). Use an all-`true` mask for [`LocalMultiplyLayout`](@ref)-equivalent
behavior.

Allocation-free [`prepare_multiply_workspace!`](@ref) given precomputed `mask`.
Typical use: `mask[i]` true for every global DOF referenced by a partition’s
owned elements (owned ∪ ghost / stencil closure).
"""
struct ReferenceMaskMultiplyLayout <: AbstractMultiplyGhostLayout
    mask::BitVector
end

function prepare_multiply_workspace!(
    work::AbstractVector{Float64},
    x::AbstractVector{Float64},
    layout::ReferenceMaskMultiplyLayout,
)
    n = length(work)
    length(x) == n ||
        throw(DimensionMismatch("work length $(length(work)), x length $(length(x))"))
    length(layout.mask) == n ||
        throw(DimensionMismatch("mask length $(length(layout.mask)), vector length $n"))
    @inbounds for i in eachindex(layout.mask)
        if layout.mask[i]
            work[i] = x[i]
        end
    end
    return work
end

"""
    MeshPartitionLayout

Lightweight metadata describing which parallel partition owns each volume
element (1-based element index matching `cache.elements`). Single-process
runs can ignore it; tests and future MPI drivers use it to choose element
subsets for [`apply_K_contributions!`](@ref).

# Fields

- `element_part_id::Vector{Int}` — `element_part_id[e]` is the partition
  id that owns element `e` (any positive integers; gaps are allowed).

# Constructors

    MeshPartitionLayout(element_part_id::Vector{Int})

Throws if any entry is `< 1`.
"""
struct MeshPartitionLayout
    element_part_id::Vector{Int}
    function MeshPartitionLayout(element_part_id::Vector{Int})
        @inbounds for i in eachindex(element_part_id)
            element_part_id[i] < 1 &&
                throw(ArgumentError("element_part_id entries must be ≥ 1 (got $(element_part_id[i]) at index $i)"))
        end
        new(element_part_id)
    end
end

"""
    uniform_single_partition(nelements::Int) -> MeshPartitionLayout

All elements belong to partition `1`.
"""
function uniform_single_partition(nelements::Int)
    nelements ≥ 0 || throw(ArgumentError("nelements must be non-negative"))
    return MeshPartitionLayout(ones(Int, nelements))
end

"""
    element_indices_for_part(layout::MeshPartitionLayout, part::Int) -> Vector{Int}

All element indices `e` with `layout.element_part_id[e] == part`.
"""
function element_indices_for_part(layout::MeshPartitionLayout, part::Int)
    return findall(==(part), layout.element_part_id)
end

"""
    validate_partition(layout::MeshPartitionLayout, nelements::Int)

Ensure `length(layout.element_part_id) == nelements`.
"""
function validate_partition(layout::MeshPartitionLayout, nelements::Integer)
    length(layout.element_part_id) == Int(nelements) ||
        throw(DimensionMismatch("partition length $(length(layout.element_part_id)), nelements $nelements"))
    return nothing
end

# --- Structured brick slabs (matches `create_structured_box_mesh(Hex8)` element order) ------------

@inline function _brick_hex_elem_ijk(e::Int, nx::Int, ny::Int)
    off = e - 1
    i = off % nx + 1
    j = (off ÷ nx) % ny + 1
    k = off ÷ (nx * ny) + 1
    return i, j, k
end

"""
    brick_hex_slab_upper(ncells::Int, nparts::Int) -> Vector{Int}

For a 1-based cell index line of length `ncells`, return `hi[p]` = last cell
index owned by slab `p` (contiguous, nearly equal widths). Used by
[`brick_hex_partition_slabs`](@ref).
"""
function brick_hex_slab_upper(ncells::Int, nparts::Int)
    ncells ≥ 1 || throw(ArgumentError("ncells must be ≥ 1"))
    nparts ≥ 1 || throw(ArgumentError("nparts must be ≥ 1"))
    nparts == 1 && return Int[ncells]
    base = ncells ÷ nparts
    rem = ncells % nparts
    hi = Vector{Int}(undef, nparts)
    cur = 0
    @inbounds for p in 1:nparts
        w = base + (p ≤ rem ? 1 : 0)
        cur += w
        hi[p] = cur
    end
    @assert hi[nparts] == ncells
    return hi
end

@inline function _part_for_slab_line(icell::Int, hi::Vector{Int})
    @inbounds for p in eachindex(hi)
        if icell ≤ hi[p]
            return Int(p)
        end
    end
    return Int(length(hi))
end

"""
    brick_hex_partition_slabs(nx, ny, nz, nparts; axis=:x) -> MeshPartitionLayout

Build an [`MeshPartitionLayout`](@ref) for a structured `nx × ny × nz` Hex8
brick whose elements are enumerated like [`create_structured_box_mesh`](@ref)
(loops `k`, then `j`, then `i`). Cells are grouped into `nparts` contiguous
slabs along `axis` (`:x`, `:y`, or `:z`).

This is a **reference partitioner** for tests and drivers; production runs
typically use graph partitioners (ParMETIS, PT‑Scotch, …).
"""
function brick_hex_partition_slabs(
    nx::Int, ny::Int, nz::Int, nparts::Int;
    axis::Symbol = :x,
)
    nx ≥ 1 && ny ≥ 1 && nz ≥ 1 ||
        throw(ArgumentError("nx, ny, nz must be ≥ 1"))
    nparts ≥ 1 || throw(ArgumentError("nparts must be ≥ 1"))
    ne = nx * ny * nz
    nc_axis = axis === :x ? nx : axis === :y ? ny : nz
    nparts > nc_axis &&
        throw(ArgumentError(
            "nparts ($nparts) cannot exceed cell count ($nc_axis) along axis $(repr(axis))",
        ))

    hi = if axis === :x
        brick_hex_slab_upper(nx, nparts)
    elseif axis === :y
        brick_hex_slab_upper(ny, nparts)
    elseif axis === :z
        brick_hex_slab_upper(nz, nparts)
    else
        throw(ArgumentError("axis must be :x, :y, or :z (got $(repr(axis)))"))
    end

    ids = Vector{Int}(undef, ne)
    @inbounds for e in 1:ne
        i, j, k = _brick_hex_elem_ijk(e, nx, ny)
        icell = axis === :x ? i : axis === :y ? j : k
        ids[e] = _part_for_slab_line(icell, hi)
    end
    return MeshPartitionLayout(ids)
end

"""
    element_counts_by_part(layout::MeshPartitionLayout) -> Dict{Int,Int}

Map partition id → number of elements (utility for sanity checks and loads).
"""
function element_counts_by_part(layout::MeshPartitionLayout)
    d = Dict{Int,Int}()
    @inbounds for p in layout.element_part_id
        d[p] = get(d, p, 0) + 1
    end
    return d
end

"""
    referenced_global_dofs(elements, element_ids) -> Vector{Int}

Sorted unique global DOF indices appearing on the listed elements (1-based
element indices into `elements`). Uses [`element_dofs`](@ref).

This helper **allocates** (set + vector). For repeated parallel staging use a
preallocated [`BitVector`](@ref) with [`mark_referenced_dofs!`](@ref) and
[`collect_true_indices!`](@ref), or [`fill_referenced_dof_indices!`](@ref).
"""
function referenced_global_dofs(
    elements::AbstractVector{El},
    element_ids::AbstractVector{Int},
) where {El <: Element}
    isempty(elements) && throw(ArgumentError("elements must be non-empty"))
    seen = Set{Int}()
    ne = length(elements)
    @inbounds for idx in eachindex(element_ids)
        e = element_ids[idx]
        (1 ≤ e ≤ ne) || throw(ArgumentError("element id $e out of range 1:$ne"))
        for d in element_dofs(elements[e])
            push!(seen, Int(d))
        end
    end
    return sort!(collect(seen))
end

# --- Zero-allocation staging (buffers supplied by caller) ---------------------

"""
    sum_element_dof_slots(elements, element_ids) -> Int

Sum of `n_element_dofs(elements[e])` over `element_ids` — upper bound on how
many flat DOF slots appear (with multiplicity) before deduplication.
"""
function sum_element_dof_slots(
    elements::AbstractVector{El},
    element_ids::AbstractVector{Int},
) where {El <: Element}
    ne = length(elements)
    total = 0
    @inbounds for idx in eachindex(element_ids)
        e = element_ids[idx]
        (1 ≤ e ≤ ne) || throw(ArgumentError("element id $e out of range 1:$ne"))
        total += n_element_dofs(elements[e])
    end
    return total
end

"""
    mark_referenced_dofs!(mask, elements, element_ids, n_total_dofs) -> mask

`fill!(mask, false)` then set `mask[d] = true` for every global DOF `d`
referenced by the listed elements. Requires `length(mask) == n_total_dofs` and
`1 ≤ d ≤ n_total_dofs` for all element DOFs.

Allocation-free once `mask` exists.
"""
function mark_referenced_dofs!(
    mask::BitVector,
    elements::AbstractVector{El},
    element_ids::AbstractVector{Int},
    n_total_dofs::Int,
) where {El <: Element}
    length(mask) == n_total_dofs ||
        throw(DimensionMismatch("mask length $(length(mask)), n_total_dofs $n_total_dofs"))
    fill!(mask, false)
    ne = length(elements)
    @inbounds for idx in eachindex(element_ids)
        e = element_ids[idx]
        (1 ≤ e ≤ ne) || throw(ArgumentError("element id $e out of range 1:$ne"))
        for d in element_dofs(elements[e])
            di = Int(d)
            (1 ≤ di ≤ n_total_dofs) ||
                throw(ArgumentError("DOF index $di out of range 1:$n_total_dofs"))
            mask[di] = true
        end
    end
    return mask
end

"""
    collect_true_indices!(buffer, mask) -> n

Write increasing indices `i` with `mask[i] == true` into `buffer[1:n]` and
return `n`. Caller must ensure `length(buffer)` is at least the number of
true entries (≤ `length(mask)`).

Allocation-free.
"""
function collect_true_indices!(buffer::Vector{Int}, mask::BitVector)::Int
    c = 0
    @inbounds for i in eachindex(mask)
        if mask[i]
            c += 1
            c ≤ length(buffer) ||
                throw(DimensionMismatch("buffer length $(length(buffer)) < $c true mask entries"))
            buffer[c] = i
        end
    end
    return c
end

"""
    fill_referenced_dof_indices!(
        dof_buffer, mask_tmp, elements, element_ids, n_total_dofs) -> n

[`mark_referenced_dofs!`](@ref) into `mask_tmp`, then
[`collect_true_indices!`](@ref) into `dof_buffer`; returns `n` unique indices.

Requires `length(mask_tmp) == n_total_dofs` and `length(dof_buffer) ≥ n`.
Allocation-free given existing buffers.
"""
function fill_referenced_dof_indices!(
    dof_buffer::Vector{Int},
    mask_tmp::BitVector,
    elements::AbstractVector{El},
    element_ids::AbstractVector{Int},
    n_total_dofs::Int,
)::Int where {El <: Element}
    mark_referenced_dofs!(mask_tmp, elements, element_ids, n_total_dofs)
    return collect_true_indices!(dof_buffer, mask_tmp)
end

"""
    ghost_dof_mask!(ghost, referenced, owned) -> ghost

For each DOF `i`, `ghost[i] = referenced[i] & !owned[i]` (broadcast semantics).
All three vectors must have equal length.

Allocation-free.
"""
function ghost_dof_mask!(
    ghost::BitVector,
    referenced::BitVector,
    owned::BitVector,
)
    length(ghost) == length(referenced) == length(owned) ||
        throw(DimensionMismatch("ghost/referenced/owned lengths $(length(ghost)), $(length(referenced)), $(length(owned))"))
    @inbounds for i in eachindex(ghost)
        ghost[i] = referenced[i] & !owned[i]
    end
    return ghost
end

"""
    node_partition_owner_min!(owner, layout, mesh::Mesh) -> owner

For each mesh node, set `owner[n]` to the **minimum** partition id among
volume elements touching that node (`typemax(Int)` entries mean isolated
nodes — should not happen on a covered volume mesh).

`length(owner)` must equal `length(mesh.nodes)`. Uses only `mesh.connectivity`
and [`MeshPartitionLayout`](@ref).

Allocation-free.
"""
function node_partition_owner_min!(
    owner::Vector{Int},
    layout::MeshPartitionLayout,
    mesh::Mesh{N,T},
) where {N,T}
    nn = length(mesh.nodes)
    length(owner) == nn ||
        throw(DimensionMismatch("owner length $(length(owner)), nnodes $nn"))
    validate_partition(layout, length(mesh.connectivity))
    fill!(owner, typemax(Int))
    conn = mesh.connectivity
    @inbounds for e in eachindex(conn)
        p = layout.element_part_id[e]
        tup = conn[e]
        for k in 1:N
            n = Int(tup[k])
            prev = owner[n]
            if p < prev
                owner[n] = p
            end
        end
    end
    return owner
end

"""
    mark_owned_vertex_field_dofs!(
        owned, handler::DOFHandler, node_owner, part; field_idx=1) -> owned

`fill!(owned, false)` then, for every mesh node `n` with `node_owner[n] == part`,
mark the contiguous Vertex DOF block starting at `field_starts[field_idx][n]`
(`dpe` components per node).

Throws unless field `field_idx` exists and uses entity type `Vertex`.

Allocation-free given `owned` (`length(owned) == handler.total_dofs`).
"""
function mark_owned_vertex_field_dofs!(
    owned::BitVector,
    handler::DOFHandler{M, S, NF},
    node_owner::Vector{Int},
    part::Int;
    field_idx::Int = 1,
) where {M, S, NF}
    (1 ≤ field_idx ≤ NF) || throw(ArgumentError("field_idx must be in 1:$NF (got $field_idx)"))
    fname = fieldnames(S)[field_idx]
    _Q, EntityType, dpe = _field_quantity_and_entity(S, fname)
    EntityType === Vertex ||
        throw(ArgumentError("mark_owned_vertex_field_dofs! requires a Vertex field (got entity $EntityType)"))

    nnodes = length(node_owner)
    nnodes == length(handler.field_starts[field_idx]) ||
        throw(DimensionMismatch("node_owner length $nnodes, field entity count $(length(handler.field_starts[field_idx]))"))

    length(owned) == handler.total_dofs ||
        throw(DimensionMismatch("owned length $(length(owned)), total_dofs $(handler.total_dofs)"))

    starts = handler.field_starts[field_idx]
    fill!(owned, false)
    @inbounds for n in 1:nnodes
        if node_owner[n] == part
            s = starts[n]
            for c in 0:(dpe - 1)
                owned[s + c] = true
            end
        end
    end
    return owned
end
