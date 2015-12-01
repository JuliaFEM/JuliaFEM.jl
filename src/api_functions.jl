# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function add_boundary_condition!{P<:APIProblem}(case::LoadCase{P}, bc)
    push!(case.boundary_conditions, bc)
end

function add_solver!{P<:APIProblem, S<:APISolver}(case::LoadCase{P},
    bc::Type{S})
    case.solver = bc
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
#function add_boundary_condition!{B <: BoundaryCondition}(case::LoadCase, bc::B)
#    push!(case.boundary_conditions, bc)
#end
#
#function add_boundary_condition!{B <: BoundaryCondition}(case::LoadCase, bc::Vector{B})
#    map(x-> push!(case.boundary_conditions, x), bc)
#end
#
#function add_loadcase!(model::Model, case::LoadCase)
#    push!(model.load_cases, case)
#end
#
#function add_loadcase!{T<:LoadCase}(model::Model, case::Vector{T})
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
