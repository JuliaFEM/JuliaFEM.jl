# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
- read meshes from different formats
- reorder connectivity, create element sets, node sets, ...
- create partitions for parallel runs
- renumber elements / nodes
- maybe precheck for bad elements
- check surface normal direction in boundary elements
- orientation of 2d elements
- etc only topology related stuff
=#

mutable struct Mesh
    nodes :: Dict{Int, Vector{Float64}}
    node_sets :: Dict{Symbol, Set{Int}}
    elements :: Dict{Int, Vector{Int}}
    element_types :: Dict{Int, Symbol}
    element_codes :: Dict{Int, Symbol}
    element_sets :: Dict{Symbol, Set{Int}}
    surface_sets :: Dict{Symbol, Vector{Tuple{Int, Symbol}}}
    surface_types :: Dict{Symbol, Symbol}
end

function Mesh()
    return Mesh(Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict())
end

"""
    Mesh(m::Dict)

Create a new `Mesh` using data `m`. It is assumed that `m` is in format what
`abaqus_read_mesh` in `AbaqusReader.jl` is returning.
"""
function Mesh(m::Dict)
    mesh = Mesh()
    mesh.nodes = m["nodes"]
    mesh.elements = m["elements"]
    mesh.element_types = m["element_types"]
    for (k, v) in m["surface_types"]
        mesh.surface_types[Symbol(k)] = v
    end
    for (nset_name, node_ids) in m["node_sets"]
        mesh.node_sets[Symbol(nset_name)] = Set(node_ids)
    end
    for (elset_name, element_ids) in m["element_sets"]
        mesh.element_sets[Symbol(elset_name)] = Set(element_ids)
    end
    for (surfset_name, surfaces) in m["surface_sets"]
        mesh.surface_sets[Symbol(surfset_name)] = surfaces
    end
    return mesh
end

"""
    add_node!(mesh, nid, ncoords)

Add node into the mesh. `nid` is node id and `ncoords` are the node
coordinates.
"""
function add_node!(mesh::Mesh, nid::Int, ncoords::Vector{Float64})
    mesh.nodes[nid] = ncoords
end

"""
    add_nodes!(mesh, nodes)

Add nodes into the mesh.
"""
function add_nodes!(mesh::Mesh, nodes::Dict{Int, Vector{Float64}})
    for (nid, ncoords) in nodes
        add_node!(mesh, nid, ncoords)
    end
end

"""
    add_node_to_node_set!(mesh, nid, ncoords)

Add nodes into a node set. `set_name` is the name of the set and `nids...`
are all the node id:s that wants to be added.
"""
function add_node_to_node_set!(mesh::Mesh, set_name, nids...)
    if !haskey(mesh.node_sets, set_name)
        mesh.node_sets[set_name] = Set{Int}()
    end
    push!(mesh.node_sets[set_name], nids...)
    return
end

"""
    create_node_set_from_element_set!(mesh, set_names...)

Create a new node set from the nodes in an element set. ´set_names...´ are all
the set names to be inserted in the function.
"""
function create_node_set_from_element_set!(mesh::Mesh, set_names::String...)
    for set_name in set_names
        set_name = Symbol(set_name)
        @info("Creating node set $set_name from element set")
        node_ids = Set{Int}()
        for elid in mesh.element_sets[set_name]
            push!(node_ids, mesh.elements[elid]...)
        end
        mesh.node_sets[set_name] = node_ids
    end
    return
end

"""
    create_node_set_from_element_set!(mesh, set_name)

Create a new node set from an element set.
"""
function create_node_set_from_element_set!(mesh::Mesh, set_name::Symbol)
    create_node_set_from_element_set!(mesh, string(set_name))
end

"""
    add_element!(mesh, elid, eltype, connectivity)

Add an element into the mesh. ´elid´ is the element id, ´eltype´ is the type of
the element and ´connectivity´ is the connectivity of the element.
"""
function FEMBase.add_element!(mesh::Mesh, elid, eltype, connectivity)
    mesh.elements[elid] = connectivity
    mesh.element_types[elid] = eltype
    return nothing
end

"""
    add_elements!(mesh, elements)

Add elements into the mesh.
"""
function FEMBase.add_elements!(mesh::Mesh, elements::Dict{Int, Tuple{Symbol, Vector{Int}}})
    for (elid, (eltype, elcon)) in elements
        add_element!(mesh, elid, eltype, elcon)
    end
    return nothing
end

"""
    add_element_to_element_set!(mesh, set_name, elids...)

Add elements into the mesh. ´set_name´ is the name of the element set and
´elids..´ are id:s of all the elements that wants to be added.
"""
function add_element_to_element_set!(mesh::Mesh, set_name, elids...)
    if !haskey(mesh.element_sets, set_name)
        mesh.element_sets[set_name] = Set{Int}()
    end
    push!(mesh.element_sets[set_name], elids...)
end

"""
    copy(mesh)

Return a copy of the mesh.
"""
function Base.copy(mesh::Mesh)
    mesh2 = Mesh()
    mesh2.nodes = copy(mesh.nodes)
    mesh2.node_sets = copy(mesh.node_sets)
    mesh2.elements = copy(mesh.elements)
    mesh2.element_types = copy(mesh.element_types)
    mesh2.element_sets = copy(mesh.element_sets)
    return mesh2
end

"""
    filter_by_element_id(mesh, element_ids)

Filter elements by their id's.
"""
function filter_by_element_id(mesh::Mesh, element_ids::Vector{Int})
    mesh2 = copy(mesh)
    mesh2.elements = Dict()
    for elid in element_ids
        if haskey(mesh.elements, elid)
            mesh2.elements[elid] = mesh.elements[elid]
        end
    end
    return mesh2
end

"""
    filter_by_element_set(mesh, set_name)

Filter elements by an element set.
"""
function filter_by_element_set(mesh::Mesh, set_name)
    filter_by_element_id(mesh::Mesh, collect(mesh.element_sets[set_name]))
end

"""
    create_element(mesh, id)

Create an element from the mesh by it's id.
"""
function create_element(mesh::Mesh, id::Int)
    connectivity = mesh.elements[id]
    element_type = getfield(JuliaFEM, mesh.element_types[id])
    element = Element(element_type, connectivity)
    element.id = id
    update!(element, "geometry", mesh.nodes)
    return element
end

function create_elements(mesh::Mesh; element_type=nothing)
    element_ids = collect(keys(mesh.elements))
    if element_type != nothing
        filter!(id -> mesh.element_types[id] == element_type, element_ids)
    end
    elements = [create_element(mesh, id) for id in element_ids]
    return elements
end

function create_elements(mesh::Mesh, element_sets::Symbol...; element_type=nothing)
    if isempty(element_sets)
        element_ids = collect(keys(mesh.elements))
    else
        element_ids = Set{Int}()
        for set_name in element_sets
            element_ids = union(element_ids, mesh.element_sets[set_name])
        end
    end

    if element_type != nothing
        filter!(id -> mesh.element_types[id] == element_type, element_ids)
    end

    elements = [create_element(mesh, id) for id in element_ids]

    nelements = length(elements)
    content = Dict{Symbol, Int}()
    for elid in element_ids
        eltype = mesh.element_types[elid]
        content[eltype] = get(content, eltype, 0) + 1
    end
    s = join(("$v x $k" for (k, v) in content), ", ")
    v = join(element_sets, ", ")
    @info("Created $nelements elements ($s) from element set: $v.")

    return elements
end

"""
    create_elements(mesh::Mesh, element_set::String)

# Examples

Suppose that there is a `mesh` with element set `Body_1`. Creating elements
based on that element set is done then

```julia
create_elements(mesh, "Body_1")
```
"""
function create_elements(mesh::Mesh, element_sets::String...)
    return create_elements(mesh, map(Symbol, element_sets)...)
end


"""
    find_nearest_nodes(mesh, coords, npts=1; node_set=nothing)

find npts nearest nodes from the mesh and return their id numbers as a list.
"""
function find_nearest_nodes(mesh::Mesh, coords::Vector{Float64}, npts::Int=1; node_set=nothing)
    dist = Dict{Int, Float64}()
    for (nid, c) in mesh.nodes
        if node_set != nothing && !(nid in mesh.node_sets[Symbol(node_set)])
            continue
        end
        dist[nid] = norm(coords-c)
    end
    s = sort(collect(dist), by=x->x[2])
    nd = s[1:npts] # [(id1, dist1), (id2, dist2), ..., (id_npts, dist_npts)]
    node_ids = [n[1] for n in nd]
    return node_ids
end

function find_nearest_node(mesh::Mesh, coords::Vector{Float64}; node_set=nothing)
    return first(find_nearest_nodes(mesh, coords, 1; node_set=node_set))
end

"""
    reorder_element_connectivity!(mesh, mapping)

Apply a new node ordering to elements. JuliaFEM uses the same node ordering as
ABAQUS. If the mesh is parsed from FEM format with some other node ordering,
this function can be used to reorder the nodes.

Parameters
----------
mapping :: Dict{Symbol, Vector{Int}}
    e.g. :Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

"""
function reorder_element_connectivity!(mesh::Mesh, mapping::Dict{Symbol, Vector{Int}})
    for (elid, eltype) in mesh.element_types
        haskey(mapping, eltype) || continue
        new_order = mapping[eltype]
        element_connectivity = mesh.elements[elid]
        new_element_connectivity = element_connectivity[new_order]
        mesh.elements[elid] = new_element_connectivity
    end
end

function JuliaFEM.Problem(mesh::Mesh, ::Type{P}, name::AbstractString, dimension::Int) where P<:FieldProblem
    problem = Problem(P, name, dimension)
    problem.elements = create_elements(mesh, name)
    return problem
end

function JuliaFEM.Problem(mesh::Mesh, ::Type{P}, name, dimension, parent_field_name) where P<:BoundaryProblem
    problem = Problem(P, name, dimension, parent_field_name)
    problem.elements = create_elements(mesh, name)
    return problem
end
