# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using SparseArrays: SparseMatrixCSC

"""
DOFHandler — type-stable global DOF distribution.

# Design philosophy

The previous `DOFManager` used `Dict{Int, Vector{Int}}` for storage. That
matches the "v1" approach the long-term type-stable vision explicitly rejected:
runtime hash lookups, type-unstable `Vector{Int}` allocations, and runtime
introspection of field type parameters inside hot loops.

`DOFHandler` flips this on its head:

1. The DOFSet `S` (NamedTuple of `DOF{Q,E}` types) is encoded as a type
   parameter, so the field structure is known at compile time.
2. Per-field DOF storage is a flat `Vector{Int}` indexed by entity id;
   one DOF range per entity, contiguous, no Dicts.
3. Element DOF assignment is implemented via a `@generated` function that
   unrolls the field loop at compile time. The inner loop becomes a plain
   `Vector{Int}` lookup followed by `UInt64` conversion — no introspection,
   no allocations.

# The element-as-template idea

Every `Element{K,P,S,N}` *is* a compile-time template: topology, basis,
field specification, and total DOF count are all type parameters. The
DOFHandler exploits this by generating a type-specific assignment routine
for each `Element{K,P,S,N}` it sees. The generated code knows exactly:

* how many DOFs the element has (`N`)
* which field each local DOF belongs to
* which entity (vertex / cell) that field lives on
* how to look that entity up in the handler's flat storage

So `_make_element_dofs(handler, ::Type{Element{K,P,S,N}}, eid, conn)` is
fully unrolled, type-stable, and zero-allocation per call.

# Storage layout

For each field `f` in `S` (in field-declaration order):

    field_starts[f][entity_id] = first global DOF index for that entity

DOFs of entity `i` in field `f` form the range
`field_starts[f][i] : field_starts[f][i] + dpe_f - 1`,
where `dpe_f = dof_size(quantity_type(field_type))` is a compile-time
constant.

DOFs are assigned in block order: all DOFs of field 1 first, then all
DOFs of field 2, and so on. This is the most common assembly-friendly
ordering and matches the semantics of the legacy `DOFManager`.

# Supported entities

Supported:
- `Vertex`: one entry per mesh node
- `Cell`: one entry per mesh element
- `Edge`, `Face`: global facet numbering via [`AbstractFacetConnectivityMaps`](@ref)
  ([`Hex8FacetMaps`](@ref), [`Tet4FacetMaps`](@ref), [`Wedge6FacetMaps`](@ref),
  [`Pyr5FacetMaps`](@ref)) on [`Mesh{8, Hex8}`](@ref), [`Mesh{20, Hex20}`](@ref),
  [`Mesh{4, Tet4}`](@ref), [`Mesh{10, Tet10}`](@ref), [`Mesh{6, Wedge6}`](@ref),
  or [`Mesh{5, Pyr5}`](@ref).
  Several unknowns per topological facet use `dof_size(quantity) > 1`
  (e.g. `DOF{Vec{2,Float64}, Edge}`): one contiguous global block per facet id,
  indexed via `field_starts` and `component` in generated `_make_element_dofs`
  (no per-quadrature heap lookups).

The handler stores `facet_maps::Union{Nothing, AbstractFacetConnectivityMaps}`;
it is built automatically when the [`DOFSet`](@ref) uses `Edge` or `Face` fields.
"""

# ============================================================================
# DOFHandler type
# ============================================================================

"""
    DOFHandler{M<:AbstractMesh, S<:DOFSet, NF}

Type-stable DOF distribution.

# Type parameters
- `M`:  mesh type (concrete)
- `S`:  DOFSet (NamedTuple) describing fields
- `NF`: number of fields, `length(fieldnames(S))`

# Fields
- `mesh::M`
- `facet_maps::Union{Nothing, AbstractFacetConnectivityMaps}` — edge/face ids
  for facet maps (`Hex8`/`Hex20`, `Tet4`/`Tet10`, `Wedge6`, `Pyr5`); `nothing` when all
  fields use only `Vertex` / `Cell`
- `field_starts::NTuple{NF, Vector{Int}}` — per-field per-entity first DOF
- `total_dofs::Int`
- `dof_connectivity::DOFConnectivity` — inverse mapping (DOF → elements).
  Filled in by `create_elements!`. Until then it carries an empty
  placeholder whose `n_total_dofs == 0`; downstream code that needs the
  real mapping checks `connectivity.n_total_dofs == handler.total_dofs`.
"""
mutable struct DOFHandler{M<:AbstractMesh, S<:DOFSet, NF}
    mesh::M
    facet_maps::Union{Nothing, AbstractFacetConnectivityMaps}
    field_starts::NTuple{NF, Vector{Int}}
    total_dofs::Int
    dof_connectivity::DOFConnectivity
end

"""
    DOFManager

Backwards-compatible alias for `DOFHandler`. New code should use
`DOFHandler` directly. `DOFManager` will be removed in a future release.
"""
const DOFManager = DOFHandler

# ============================================================================
# Compile-time field analysis (unexported helpers)
# ============================================================================

"""
    _field_quantity_and_entity(::Type{S}, fname)

Return `(QuantityType, EntityType, dof_per_entity)` for field `fname` of
DOFSet `S`. Works at the type level only — pure function, no runtime work
when called inside a `@generated` function.
"""
function _field_quantity_and_entity(::Type{S}, fname::Symbol) where {S<:DOFSet}
    FT = fieldtype(S, fname)
    if !(FT <: DOF)
        error("DOFHandler: field :$fname has type $FT, expected DOF{Q,E}")
    end
    Q = FT.parameters[1]
    E = FT.parameters[2]
    Qresolved = quantity_type(FT)  # Displacement{3} → Vec{3}
    return Qresolved, E, dof_size(Qresolved)
end

"""
    _field_entity_count(mesh, ::Type{E})

Number of entities of type `E` in `mesh`. Used at handler-setup time only.
"""
_field_entity_count(mesh::AbstractMesh, ::Type{Vertex}, _) = length(mesh.nodes)
_field_entity_count(mesh::AbstractMesh, ::Type{Cell}, _)   = length(mesh.connectivity)

function _field_entity_count(::AbstractMesh, ::Type{Edge}, maps::Union{Nothing, AbstractFacetConnectivityMaps})
    maps === nothing && error("DOFHandler: Edge field requires facet_maps on the handler.")
    return maps.n_edges
end

function _field_entity_count(::AbstractMesh, ::Type{Face}, maps::Union{Nothing, AbstractFacetConnectivityMaps})
    maps === nothing && error("DOFHandler: Face field requires facet_maps on the handler.")
    return maps.n_faces
end

function _dofset_uses_edge_or_face(::Type{S}) where {S<:DOFSet}
    for fname in fieldnames(S)
        FT = fieldtype(S, fname)
        FT <: DOF || continue
        E = FT.parameters[2]
        if E === Edge || E === Face
            return true
        end
    end
    return false
end

function _facet_maps_for_dofset(mesh::AbstractMesh, ::Type{S}) where {S<:DOFSet}
    !_dofset_uses_edge_or_face(S) && return nothing
    if mesh isa Mesh{8, Hex8}
        return build_hex8_facet_maps(mesh::Mesh{8, Hex8})
    elseif mesh isa Mesh{20, Hex20}
        return build_hex20_facet_maps(mesh::Mesh{20, Hex20})
    elseif mesh isa Mesh{4, Tet4}
        return build_tet4_facet_maps(mesh::Mesh{4, Tet4})
    elseif mesh isa Mesh{10, Tet10}
        return build_tet10_facet_maps(mesh::Mesh{10, Tet10})
    elseif mesh isa Mesh{6, Wedge6}
        return build_wedge6_facet_maps(mesh::Mesh{6, Wedge6})
    elseif mesh isa Mesh{5, Pyr5}
        return build_pyr5_facet_maps(mesh::Mesh{5, Pyr5})
    end
    error(
        "DOFHandler: Edge/Face fields require Mesh{8,Hex8}, Mesh{20,Hex20}, Mesh{4,Tet4}, Mesh{10,Tet10}, Mesh{6,Wedge6}, or Mesh{5,Pyr5}; got $(typeof(mesh)).",
    )
end

# ============================================================================
# Constructor
# ============================================================================

"""
    DOFHandler(mesh, ::Type{S}) → DOFHandler

Build a fresh DOF handler for the given mesh and DOFSet specification.

DOFs are assigned in block order: all DOFs of field 1, then all DOFs of
field 2, etc. Within a field, entities are visited in entity-id order
(node id for `Vertex`, element id for `Cell`).

# Example
```julia
mesh = build_mesh(...)
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
handler = DOFHandler(mesh, S)
```
"""
function DOFHandler(mesh::M, ::Type{S}) where {M<:AbstractMesh, S<:DOFSet}
    field_names = fieldnames(S)
    NF = length(field_names)
    facet_maps = _facet_maps_for_dofset(mesh, S)

    starts_vec = Vector{Vector{Int}}(undef, NF)
    next_dof = 1
    for (fi, fname) in enumerate(field_names)
        _, EntityType, dpe = _field_quantity_and_entity(S, fname)
        nent = _field_entity_count(mesh, EntityType, facet_maps)
        v = Vector{Int}(undef, nent)
        @inbounds for e in 1:nent
            v[e] = next_dof
            next_dof += dpe
        end
        starts_vec[fi] = v
    end
    total = next_dof - 1
    field_starts = ntuple(i -> starts_vec[i], NF)
    return DOFHandler{M, S, NF}(mesh, facet_maps, field_starts, total, DOFConnectivity())
end

# ============================================================================
# Global layout helpers (mixed / multi-field solvers)
# ============================================================================

"""
    global_field_ranges(handler::DOFHandler{M,S,NF}) -> NTuple{NF,UnitRange{Int}}

Contiguous global DOF index ranges per field, in **field-declaration order**
(the same block layout as [`DOFHandler`](@ref): all DOFs of field 1, then field 2, …).

Use with [`saddle_point_blocks`](@ref) to extract `u` / `p` blocks from an assembled
mixed stiffness matrix without hard-coding `3 * nnodes`.

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
_, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
ru, rp = global_field_ranges(handler)
length(ru) + length(rp) == handler.total_dofs
```
"""
function global_field_ranges(handler::DOFHandler{M, S, NF}) where {M, S, NF}
    mesh = handler.mesh
    field_names = fieldnames(S)
    maps = handler.facet_maps
    chunk_sizes = Vector{Int}(undef, NF)
    @inbounds for fi in 1:NF
        fname = field_names[fi]
        _, EntityType, dpe = _field_quantity_and_entity(S, fname)
        nent = _field_entity_count(mesh, EntityType, maps)
        chunk_sizes[fi] = nent * dpe
    end
    starts = Vector{Int}(undef, NF + 1)
    starts[1] = 1
    @inbounds for fi in 1:NF
        starts[fi + 1] = starts[fi] + chunk_sizes[fi]
    end
    tdof = starts[end] - 1
    tdof == handler.total_dofs || error(
        "global_field_ranges: layout mismatch (computed total $tdof, handler.total_dofs $(handler.total_dofs))",
    )
    return ntuple(fi -> starts[fi]:(starts[fi + 1] - 1), Val(NF))
end

"""
    global_facet_dof(handler::DOFHandler, field_index::Int, facet_gid::Int) -> Int

Global DOF index for the **first component** of field `field_index` on geometric facet
`facet_gid`. Indices follow [`DOFHandler`](@ref) storage: `field_index` is the field’s position
in the [`DOFSet`](@ref) tuple (`1` for the first declared field), and `facet_gid` matches
[`Tet4FacetMaps`](@ref).`elem_face_gid` entries (`1 … facet_maps.n_faces`).

The selected field must live on [`Face`](@ref); for vertex or cell unknowns, index
[`DOFHandler`](@ref).`field_starts[field_index]` directly.

Typical use: locate a boundary [`RT0FaceFlux`](@ref) unknown after
[`tet_facet_gid_from_corners`](@ref), then apply [`PenaltyDirichlet`](@ref) /
[`EliminatedDirichlet`](@ref) for a prescribed normal flux (integrated flux DOF).
"""
function global_facet_dof(handler::DOFHandler{M, S, NF}, field_index::Int, facet_gid::Int) where {M, S, NF}
    (1 ≤ field_index ≤ NF) ||
        throw(ArgumentError("global_facet_dof: field_index $field_index out of range 1:$NF"))
    fname = fieldnames(S)[field_index]
    FT = fieldtype(S, fname)
    FT <: DOF ||
        throw(ArgumentError("global_facet_dof: field :$fname has type $FT, expected DOF{…}"))
    E = FT.parameters[2]
    E === Face || throw(
        ArgumentError(
            "global_facet_dof: field :$fname uses entity $E, not Face — index field_starts[$field_index] instead.",
        ),
    )
    maps = handler.facet_maps
    maps === nothing && throw(ArgumentError("global_facet_dof: handler has no facet_maps (Face field required)."))
    fs = handler.field_starts[field_index]
    (1 ≤ facet_gid ≤ length(fs)) ||
        throw(ArgumentError("global_facet_dof: facet_gid $facet_gid out of range 1:$(length(fs))"))
    return @inbounds fs[facet_gid]
end

"""
    saddle_point_blocks(K, r_u, r_p)

Return a named tuple `(A, B, Bt, C)` of **views** into `K`:

  * `A = K[r_u, r_u]` — primal–primal block
  * `B = K[r_u, r_p]` — rectangular coupling block
  * `Bt = K[r_p, r_u]` — other coupling rectangle (`B` and `Bt` are transposes of each other **only** when `K` is symmetric)
  * `C = K[r_p, r_p]` — constraint / pressure block (often singular when compressibility is zero)

Typical usage: `ru, rp = global_field_ranges(handler)` for a two-field `u`–`p`
[`DOFSet`](@ref). Many mixed elasticity–pressure Jacobians are symmetric; assembled operators from
non-associated plasticity, convection, or strongly asymmetric contact/friction can leave `K`
non-symmetric, so do not assume `Bt == transpose(B)` unless the formulation guarantees it.

Views follow ordinary Julia slicing rules (including sparse submatrices).
"""
function saddle_point_blocks(
    K::AbstractMatrix,
    r_u::AbstractRange{Int},
    r_p::AbstractRange{Int},
)
    @views (
        A = K[r_u, r_u],
        B = K[r_u, r_p],
        Bt = K[r_p, r_u],
        C = K[r_p, r_p],
    )
end

"""
    saddle_point_matrix_blocks(K::SparseMatrixCSC, r_u, r_p)

Like [`saddle_point_blocks`](@ref), but each block is a dedicated
[`SparseMatrixCSC`](@ref) submatrix (typically a copy), convenient for block linear algebra,
sparse factorisations, or building approximate Schur preconditioners.

Indexing uses the same global ranges as [`global_field_ranges`](@ref).
"""
function saddle_point_matrix_blocks(
    K::SparseMatrixCSC{Float64, Ti},
    r_u::AbstractRange{Int},
    r_p::AbstractRange{Int},
) where {Ti<:Integer}
    return (
        A = K[r_u, r_u],
        B = K[r_u, r_p],
        Bt = K[r_p, r_u],
        C = K[r_p, r_p],
    )
end

# ============================================================================
# @generated per-element DOF assignment
# ============================================================================

"""
    _make_element_dofs(handler, ::Type{Element{K,P,S,N}}, elem_id, connectivity)
        → NTuple{N, UInt64}

Compute the global DOF indices for one element. Fully unrolled at compile
time using `@generated` dispatch on the Element template.

# Performance
Zero allocations. ~3.7 ns per Hex8 element on the prototype benchmark.

# Implementation
At code-generation time, for each field `(fname → DOF{Q,E})` in `S`:
- if `E === Vertex`, emit one tuple slot per `(local_node × component)`,
  reading `handler.field_starts[fidx][connectivity[local_node]] + (c-1)`;
- if `E === Cell`, emit one tuple slot per `component`, reading
  `handler.field_starts[fidx][elem_id] + (c-1)`;
- if `E === Edge`, emit slots via `handler.facet_maps.elem_edge_gid[k, elem_id]`;
- if `E === Face`, emit slots via `handler.facet_maps.elem_face_gid[k, elem_id]`.

The emitted code is a single `Expr(:tuple, ...)` whose entries are pure
arithmetic on integer loads, so the compiler inlines and SIMD-optimizes
it freely.
"""
@generated function _make_element_dofs(
    handler::DOFHandler{M, S, NF},
    ::Type{Element{K, P, S, NDOF}},
    elem_id::Integer,
    connectivity::NTuple{Nnodes, T}
) where {M, S, NF, K, P, NDOF, Nnodes, T<:Integer}
    field_names = fieldnames(S)
    expressions = Expr[]
    total_emitted = 0

    for (fidx, fname) in enumerate(field_names)
        FT = fieldtype(S, fname)
        if !(FT <: DOF)
            return :(error("DOFHandler: field :$($fname) has type $($FT), expected DOF{Q,E}"))
        end
        Q = quantity_type(FT)
        E = FT.parameters[2]
        dpe = dof_size(Q)

        if E === Vertex
            for k in 1:Nnodes, c in 1:dpe
                push!(expressions, :(UInt64(@inbounds(handler.field_starts[$fidx][Int(connectivity[$k])]) + $(c - 1))))
                total_emitted += 1
            end
        elseif E === Cell
            for c in 1:dpe
                push!(expressions, :(UInt64(@inbounds(handler.field_starts[$fidx][Int(elem_id)]) + $(c - 1))))
                total_emitted += 1
            end
        elseif E === Edge
            NK = nedges(K)
            for k in 1:NK, c in 1:dpe
                push!(
                    expressions,
                    :(UInt64(@inbounds(handler.field_starts[$fidx][Int(handler.facet_maps.elem_edge_gid[$k, Int(elem_id)])]) + $(c - 1))),
                )
                total_emitted += 1
            end
        elseif E === Face
            NF = nfaces(K)
            for k in 1:NF, c in 1:dpe
                push!(
                    expressions,
                    :(UInt64(@inbounds(handler.field_starts[$fidx][Int(handler.facet_maps.elem_face_gid[$k, Int(elem_id)])]) + $(c - 1))),
                )
                total_emitted += 1
            end
        else
            return :(error("DOFHandler: entity type $($E) is not supported in `_make_element_dofs`."))
        end
    end

    if total_emitted != NDOF
        return :(error("DOFHandler: element template Element{$($K),$($P),$($S),$($NDOF)} expected $($NDOF) DOFs, " *
                       "but the field specification yields $($total_emitted)."))
    end

    return Expr(:tuple, expressions...)
end

# ============================================================================
# create_elements!
# ============================================================================

"""
    create_elements!(mesh, ElementType) → (elements, handler)

Build the element list and the matching `DOFHandler`. The element type
fully encodes the template: topology `K`, basis `P`, DOFSet `S`, and total
DOF count `N`. The handler is built fresh for the given DOFSet.

The inverse mapping `handler.dof_connectivity` (DOF → elements) is built
automatically and stored on the handler for the assembler to use.

# Example
```julia
mesh = build_my_hex8_mesh(...)
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
elements, handler = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
```
"""
function create_elements!(
    mesh::Mesh{Nm, MeshTopo},
    ::Type{Element{K, P, S, NDOF}}
) where {Nm, MeshTopo, K, P, S<:DOFSet, NDOF}
    if K !== MeshTopo
        @warn "Element topology $K does not match mesh topology $MeshTopo. " *
              "Heterogeneous meshes are not yet supported by DOFHandler."
    end

    handler = DOFHandler(mesh, S)
    ET = Element{K, P, S, NDOF}
    elements = Vector{ET}(undef, length(mesh.connectivity))

    @inbounds for (eid, conn) in enumerate(mesh.connectivity)
        dofs = _make_element_dofs(handler, ET, eid, conn)
        elements[eid] = ET(UInt(eid), dofs)
    end

    # Build inverse mapping (DOF → elements) for the assembler
    handler.dof_connectivity = build_dof_connectivity(elements, handler)

    return elements, handler
end

# Convenience overload: outer Element type without explicit N (N inferred
# from S and K via the generated `ndofs(K, S)`)
function create_elements!(
    mesh::Mesh,
    ::Type{Element{K, P, S}}
) where {K, P, S<:DOFSet}
    NDOF = ndofs(K, S)
    return create_elements!(mesh, Element{K, P, S, NDOF})
end

# Single-field convenience: wrap bare DOF{Q,E} into a one-field DOFSet
function create_elements!(
    mesh::Mesh,
    ::Type{Element{K, P, S}}
) where {K, P, S<:DOF}
    Swrapped = NamedTuple{(:dof,), Tuple{S}}
    return create_elements!(mesh, Element{K, P, Swrapped})
end

# ============================================================================
# Per-node DOF query (legacy DOFManager compatibility)
# ============================================================================

"""
    get_node_dofs(handler::DOFHandler, node_id::Int) → Vector{Int}

Return all global DOF indices attached to a given mesh node, across all
fields whose entity type is `Vertex`.

This is a legacy convenience helper for boundary-condition application.
For zero-allocation hot loops, prefer reading `handler.field_starts[fi]`
directly, since you know the field index and dofs-per-entity at compile
time.
"""
function get_node_dofs(handler::DOFHandler{M, S, NF}, node_id::Integer) where {M, S, NF}
    dofs = Int[]
    field_names = fieldnames(S)
    for (fi, fname) in enumerate(field_names)
        Q, E, dpe = _field_quantity_and_entity(S, fname)
        if E === Vertex
            start = handler.field_starts[fi][Int(node_id)]
            for c in 0:(dpe - 1)
                push!(dofs, start + c)
            end
        end
    end
    return dofs
end

# ============================================================================
# Element-set helpers (kept from legacy DOFManager API)
# ============================================================================

"""
    get_element_ids(mesh, set_name) → Vector{Int}

Return element IDs that belong to the named element set.
"""
function get_element_ids(mesh, element_set_name::String)
    if haskey(mesh.element_sets, element_set_name)
        return mesh.element_sets[element_set_name]
    else
        error("Element set '$element_set_name' not found in mesh. " *
              "Available sets: $(keys(mesh.element_sets))")
    end
end

get_element_ids(mesh, set_name::Symbol) = get_element_ids(mesh, String(set_name))
