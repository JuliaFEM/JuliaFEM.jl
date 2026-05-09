# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    InterfaceDOFHandler{M, S, NF}

DOF numbering on an [`AbstractInterfaceMesh`](@ref), independent of volume
[`DOFHandler`](@ref). Same block ordering: field 1 entities, then field 2, …

Fields:

- `mesh::M` — [`InterfaceMesh`](@ref)
- `field_starts::NTuple{NF, Vector{Int}}` — first global DOF per entity
- `total_dofs::Int`
- `dof_connectivity::DOFConnectivity` — filled by
  [`create_interface_elements!`](@ref), same role as on [`DOFHandler`](@ref).
  Carries an empty placeholder until then.

Only `Vertex` and `Cell` entity kinds are supported on interface meshes until
interface facet maps exist.
"""
mutable struct InterfaceDOFHandler{M<:AbstractInterfaceMesh, S<:DOFSet, NF}
    mesh::M
    field_starts::NTuple{NF, Vector{Int}}
    total_dofs::Int
    dof_connectivity::DOFConnectivity
end

"""
    InterfaceDOFHandler(mesh::InterfaceMesh, ::Type{S}) → InterfaceDOFHandler

Build global DOF indices for interface unknowns (Lagrange multipliers, interface
temperatures, …).
"""
function InterfaceDOFHandler(mesh::InterfaceMesh, ::Type{S}) where {S<:DOFSet}
    _dofset_uses_edge_or_face(S) &&
        error("InterfaceDOFHandler: Edge/Face fields are not supported on InterfaceMesh yet.")
    field_names = fieldnames(S)
    NF = length(field_names)
    starts_vec = Vector{Vector{Int}}(undef, NF)
    next_dof = 1
    for (fi, fname) in enumerate(field_names)
        _, EntityType, dpe = _field_quantity_and_entity(S, fname)
        nent = _field_entity_count(mesh, EntityType, nothing)
        v = Vector{Int}(undef, nent)
        @inbounds for e in 1:nent
            v[e] = next_dof
            next_dof += dpe
        end
        starts_vec[fi] = v
    end
    total = next_dof - 1
    field_starts = ntuple(i -> starts_vec[i], NF)
    return InterfaceDOFHandler{typeof(mesh), S, NF}(mesh, field_starts, total, DOFConnectivity())
end

@generated function _make_interface_element_dofs(
    handler::InterfaceDOFHandler{M, S, NF},
    ::Type{Element{K, P, S, NDOF}},
    elem_id::Integer,
    connectivity::NTuple{Nnodes, T},
) where {M, S, NF, K, P, NDOF, Nnodes, T<:Integer}
    field_names = fieldnames(S)
    expressions = Expr[]
    total_emitted = 0

    for (fidx, fname) in enumerate(field_names)
        FT = fieldtype(S, fname)
        if !(FT <: DOF)
            return :(error("InterfaceDOFHandler: field :$($fname) has type $($FT), expected DOF{Q,E}"))
        end
        Q = quantity_type(FT)
        E = FT.parameters[2]
        dpe = dof_size(Q)

        if E === Vertex
            for k in 1:Nnodes, c in 1:dpe
                push!(
                    expressions,
                    :(UInt64(@inbounds(handler.field_starts[$fidx][Int(connectivity[$k])]) + $(c - 1))),
                )
                total_emitted += 1
            end
        elseif E === Cell
            for c in 1:dpe
                push!(
                    expressions,
                    :(UInt64(@inbounds(handler.field_starts[$fidx][Int(elem_id)]) + $(c - 1))),
                )
                total_emitted += 1
            end
        else
            return :(error("InterfaceDOFHandler: entity type $($E) is not supported."))
        end
    end

    if total_emitted != NDOF
        return :(error(
            "InterfaceDOFHandler: Element{$($K),$($P),$($S),$($NDOF)} expected $($NDOF) DOFs, got $($total_emitted).",
        ))
    end

    return Expr(:tuple, expressions...)
end

"""
    create_interface_elements!(mesh::InterfaceMesh, ElementType) → (elements, handler)

Like [`create_elements!`](@ref) for volumes: build [`Element`](@ref) templates on the
interface mesh and attach [`InterfaceDOFHandler`](@ref) + [`DOFConnectivity`](@ref).
"""
function create_interface_elements!(
    mesh::InterfaceMesh{Nm, MeshTopo},
    ::Type{Element{K, P, S, NDOF}},
) where {Nm, MeshTopo, K, P, S<:DOFSet, NDOF}
    K !== MeshTopo && @warn "Interface element topology $K does not match InterfaceMesh topology $MeshTopo."
    handler = InterfaceDOFHandler(mesh, S)
    ET = Element{K, P, S, NDOF}
    elements = Vector{ET}(undef, length(mesh.connectivity))
    @inbounds for (eid, conn) in enumerate(mesh.connectivity)
        dofs = _make_interface_element_dofs(handler, ET, eid, conn)
        elements[eid] = ET(UInt(eid), dofs)
    end
    handler.dof_connectivity = build_dof_connectivity(elements, handler.total_dofs)
    return elements, handler
end

function create_interface_elements!(
    mesh::InterfaceMesh,
    ::Type{Element{K, P, S}},
) where {K, P, S<:DOFSet}
    NDOF = ndofs(K, S)
    return create_interface_elements!(mesh, Element{K, P, S, NDOF})
end

function create_interface_elements!(
    mesh::InterfaceMesh,
    ::Type{Element{K, P, S}},
) where {K, P, S<:DOF}
    Swrapped = NamedTuple{(:dof,), Tuple{S}}
    return create_interface_elements!(mesh, Element{K, P, Swrapped})
end

build_dof_connectivity(elements::Vector, handler::InterfaceDOFHandler) =
    build_dof_connectivity(elements, handler.total_dofs)
