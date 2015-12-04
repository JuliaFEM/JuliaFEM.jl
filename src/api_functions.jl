# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function add_boundary_condition!(case::Simulation, bc::NeumannBC)
    push!(case.neumann_boundary_conditions, bc)
end

function add_node!(model::Model, index::Union{Int64, ASCIIString},
    coords::Vector{Float64})
    node = Node(index, coords)
    model.nodes[index] = node
end


function add_boundary_condition!(case::Simulation, bc::DirichletBC)
    push!(case.dirichlet_boundary_conditions, bc)
end

function add_solver!(case::Simulation, solver)
    case.solver = solver
end

function add_material!(model::Model, set_name::ASCIIString, material::Material)
    element_set = model.elsets[set_name]
    set_ids = element_set.elements
    for each in set_ids
        element = model.elements[each]
        element.material = material
    end
    element_set.material = material
end

function add_element!(model::Model, idx::Union{Int64, ASCIIString},
    eltype::Symbol, node_ids::Vector{Int64})

    element = Element(idx, node_ids, eltype)
    model.elements[idx] = element
end

function add_element_set!(model::Model, elset::ElementSet)
    name = elset.name
    model.elsets[name] = elset
end

function add_element_set!(model::Model, name::ASCIIString, ids::Vector{Int64})
    elset = ElementSet(name, ids)
    model.elsets[name] = elset
end

function add_node_set!(model::Model, name::ASCIIString, ids::Vector{Int64})
    nset = NodeSet(name, ids)
    model.nsets[name] = nset
end

function add_element_set!(model::Model, name::ASCIIString,
    elements::Vector{Element})
    elset = ElementSet(name, elements)
model.elsets[name] = elset
end

function add_element_set!(case::Simulation, name::ASCIIString)
    push!(case.sets, name)
end

function add_simulation!(model::Model, name::ASCIIString, case::Simulation)
    model.load_cases[name] = case
end
#function Base.convert{T<:AbstractFloat}(::Type{Node}, data::Vector{T})
#    Node(data)
#end:q
#
#
#function set_material!{S <: AbstractString}(model::Model, material::Material, set_name::S)
#    model.sets[set_name].material = material
#end
#
#
#function get_element_set(model::Model, set_name::ASCIIString)
#    get_set = model.sets[set_name]
#    isa(get_set, ElementSet) ? get_set : err("Found set $(set_name)
#    but it is not a ElementSet. Check if you have used
#    dublicate set names.")
#end
#
#function add_boundary_condition!{B <: BoundaryCondition}(case::Simulation, bc::B)
#    push!(case.boundary_conditions, bc)
#end
#
#function add_boundary_condition!{B <: BoundaryCondition}(case::Simulation, bc::Vector{B})
#    map(x-> push!(case.boundary_conditions, x), bc)
#end
#
#function add_loadcase!(model::Model, case::Simulation)
#    push!(model.load_cases, case)
#end
#
#function add_loadcase!{T<:Simulation}(model::Model, case::Vector{T})
#    map(x-> push!(model.load_cases, x), case)
#end
#
#"""
#"""
#function build_core_elements()
#
#end
#
#"""
#"""
#function set_core_element_material()
#
#end
#
#"""
#element_has_type( ::Type{Val{:C3D4}}) = Tet4
#"""
#function create_problems()
#
#end
#
#"""
#"""
#function add_problems!()
#
#end
#
#"""
#Get set from model
#"""
#function get_set{S <: AbstractString}(model::Model, name::S)
#    try
#        return model.set[name]
#    catch
#        err("Given set: $(name) does not exist in Model")
#    end
#end
#
#"""
#"""
#function push!(model::Model, element::Element)
#    push!(model.elements, element...)
#end
#
#"""
#"""
#function push!(model::Model, elements::Vector{Element})
#    push!(model.element, elements...)
#end
#
#"""
#"""
#function push!(model::Model, node::Node)
#    push!(model.nodes, node)
#end
#
#function push!(model::Model, nodes::Vector{Node})
#    push!(mode, nodes...)
#end
#
#"""
#"""
#function add_set!(model::Model, set::Union{NodeSet, ElementSet})
#    model.sets[set.name] = set
#end
#
#function add_set!{S <: AbstractString}(model::Model, ::Type{Val{:NSET}},
#    name::S, ids::Vector{Integer})
#    new_set = NodeSet(name, ids)
#    add_set!(model, new_set)
#end
#
#function add_set!{S <: AbstractString}(model::Model, ::Type{Val{:ELSET}},
#    name::S, ids::Vector{Integer})
#    new_set = ElementSet(name, ids)
#    add_set!(model, new_set)
#end
