# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

module JuliaFEMGmshExt

using JuliaFEM
using Gmsh
using Tensors: Vec

const _GMSH_LINEAR_TYPE = Dict{Int32,DataType}(
    Int32(2) => Tri3,
    Int32(3) => Quad4,
    Int32(4) => Tet4,
    Int32(5) => Hex8,
)

function _physical_symbol(name::AbstractString, dim::Integer, tag::Integer)::Symbol
    s = strip(String(name))
    if isempty(s)
        return Symbol("physical_", dim, "_", tag)
    end
    symstr = replace(s, r"[^0-9a-zA-Z_]+" => "_")
    if isempty(symstr)
        return Symbol("physical_", dim, "_", tag)
    end
    return Symbol(symstr)
end

function _mesh_dim!(gmsh, dim::Union{Nothing,Int})::Int32
    if dim === nothing
        et3, _, _ = gmsh.model.mesh.getElements(3, -1)
        if !isempty(et3)
            return Int32(3)
        end
        et2, _, _ = gmsh.model.mesh.getElements(2, -1)
        if !isempty(et2)
            return Int32(2)
        end
        throw(ArgumentError("Gmsh model has no 2D or 3D mesh elements"))
    end
    dim == 2 || dim == 3 || throw(ArgumentError("dim must be 2 or 3, got $dim"))
    return Int32(dim)
end

function _topology_from_elem_types(elem_types::Vector{Int32})::DataType
    isempty(elem_types) && throw(ArgumentError("Gmsh returned no element types for the requested dimension"))
    Ts = DataType[]
    for gtyp in elem_types
        T = get(_GMSH_LINEAR_TYPE, gtyp, nothing)
        T === nothing && throw(
            ArgumentError(
                "Unsupported Gmsh element type $gtyp (only linear types " *
                "$(join(sort!(collect(keys(_GMSH_LINEAR_TYPE))), ", ")) are supported)",
            ),
        )
        push!(Ts, T)
    end
    all(==(Ts[1]), Ts) ||
        throw(ArgumentError("Mixed Gmsh element types in one dimension are not supported: $Ts"))
    return Ts[1]
end

function _import_mesh(gmsh; dim::Union{Nothing,Int}, quiet::Bool)
    mesh_dim = _mesh_dim!(gmsh, dim)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(mesh_dim, -1)
    Ttop = _topology_from_elem_types(elem_types)

    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    length(node_tags) * 3 == length(coord) ||
        throw(ArgumentError("Unexpected Gmsh node coordinate layout"))

    n_nodes = length(node_tags)
    tag_to_idx = Dict{UInt64,UInt32}()
    nodes = Vector{Vec{3,Float64}}(undef, n_nodes)
    @inbounds for i in 1:n_nodes
        tag_to_idx[node_tags[i]] = UInt32(i)
        base = 3 * (i - 1)
        nodes[i] = Vec(coord[base + 1], coord[base + 2], coord[base + 3])
    end

    N = nnodes(Ttop())
    connectivity = NTuple{N,UInt32}[]
    elem_tag_to_julia = Dict{UInt64,UInt32}()
    @inbounds for it in eachindex(elem_types)
        tags_i = elem_tags[it]
        nt_i = elem_node_tags[it]
        n_elem = length(tags_i)
        expected_len = n_elem * N
        length(nt_i) == expected_len ||
            throw(ArgumentError("Gmsh node tag list length mismatch for element type $(elem_types[it])"))
        for e in 1:n_elem
            julia_e = UInt32(length(connectivity) + 1)
            elem_tag_to_julia[tags_i[e]] = julia_e
            offs = N * (e - 1)
            conn = ntuple(N) do k
                tag = nt_i[offs + k]
                idx = get(tag_to_idx, tag, nothing)
                idx === nothing && throw(ArgumentError("Unknown node tag $tag in element connectivity"))
                idx
            end
            push!(connectivity, conn)
        end
    end

    element_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1):UInt32(length(connectivity))))
    node_sets = Dict{Symbol,Set{UInt32}}(:all => Set(UInt32(1):UInt32(n_nodes)))

    for (pdim, ptag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(Int(pdim), Int(ptag))
        sym = _physical_symbol(name, pdim, ptag)
        if pdim == mesh_dim
            acc = get!(Set{UInt32}, element_sets, sym)
            for ent in gmsh.model.getEntitiesForPhysicalGroup(Int(pdim), Int(ptag))
                _, e_tags_b, _ = gmsh.model.mesh.getElements(Int(pdim), Int(ent))
                for it in eachindex(e_tags_b)
                    for etag in e_tags_b[it]
                        ji = get(elem_tag_to_julia, etag, nothing)
                        ji === nothing || push!(acc, ji)
                    end
                end
            end
        elseif pdim == mesh_dim - 1
            acc = get!(Set{UInt32}, node_sets, sym)
            for ent in gmsh.model.getEntitiesForPhysicalGroup(Int(pdim), Int(ptag))
                ntags_b, _, _ = gmsh.model.mesh.getNodes(Int(pdim), Int(ent))
                for t in ntags_b
                    ni = get(tag_to_idx, t, nothing)
                    ni === nothing || push!(acc, ni)
                end
            end
        end
    end

    for d in (element_sets, node_sets)
        for k in collect(keys(d))
            k === :all && continue
            isempty(d[k]) && delete!(d, k)
        end
    end

    return Mesh{N,Ttop}(nodes, connectivity, element_sets, node_sets)
end

function JuliaFEM.read_gmsh_msh(path::AbstractString; dim=nothing, quiet::Bool=true)
    argv = quiet ? String["-v", "0"] : String[]
    started = Gmsh.initialize(argv; finalize_atexit=false)
    try
        Gmsh.gmsh.open(String(path))
        return _import_mesh(Gmsh.gmsh; dim, quiet)
    finally
        started && Gmsh.finalize()
    end
end

function JuliaFEM.mesh_from_current_gmsh_model(; dim=nothing, quiet::Bool=true)
    Bool(Gmsh.gmsh.isInitialized()) ||
        throw(ArgumentError("Gmsh is not initialized; call Gmsh.initialize first"))
    _ = quiet
    return _import_mesh(Gmsh.gmsh; dim, quiet)
end

end # module
