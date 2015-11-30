# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract BoundaryCondition

type NeumannBC <: BoundaryCondition
    value
end

type DirichletBC <: BoundaryCondition

end

typealias DisplacementBC DirichletBC
typealias TemperatureBC DirichletBC

typealias ForceBC NeumannBC
typealias HeatFluxBC NeumannBC

"""
"""
type Material{S <: AbstractString, T<:Real}
    name :: AbstractString
    scalar_data :: Dict{S, T}
end

Material(name, data) = Material(name, Dict(data))
Material(name) = Material(name, Dict{AbstractString, Real}())
Material() = Material("", Dict{AbstractString, Real}())

function Base.setindex!{T <: AbstractString }(material::Material, val::Real, name::T)
    material.scalar_data[name] = val 
end

"""
"""
type Node{T<:Real}
    id::Integer
    coords::Array{T, 1}
end

#"""
#"""
#type MElement{I <: Integer} #, T <: AbstractElement}
#    id :: Integer
#    connectivity :: Array{I, 1}
#    element_type :: Tet4
#end

"""
Set of nodes. Holds name and the ids
"""
type NodeSet
    name :: AbstractString
    node_ids :: Array{Integer, 1} 
end

#julia> type myn
#       arr :: Array{Integer, 1}
#       finder :: Dict{Integer, Integer}
#       end
#
#julia> myn(arr::Array{Int64, 1}) = myn(arr, Dict(zip(arr, collect(1:length(arr)))))


"""
"""
type ElementSet
    name :: AbstractString
    element_ids :: Array{Integer, 1}
    material :: Material
end

ElementSet(name, el_ids) = ElementSet(name, el_ids, Material())

type LoadCase{ T <: AbstractProblem }
    problem :: T
    boundary_conditions :: Vector{BoundaryProblem}
end


"""
"""
type Model
    nodes :: Vector{Node}
    elements :: Vector{Element}
    sets :: Dict{AbstractString, Union{NodeSet, ElementSet}}
    load_cases :: Vector{LoadCase}
 #   settings :: Dict{AbstractString, Real}
end

Model() = Model(Node[],
                Element[],
                Dict{AbstractString, Union{NodeSet, ElementSet}}(),
                LoadCase[],
)

function set_material!{S <: AbstractString}(model::Model, material::Material, set_name::S)
    model.sets[set_name].material = material
end


"""
"""
function build_core_elements()

end

"""
"""
function set_core_element_material()

end

"""
element_has_type( ::Type{Val{:C3D4}}) = Tet4
"""
function create_problems()

end

"""
"""
function add_problems!()

end

"""
Get set from model
"""
function get_set{S <: AbstractString}(model::Model, name::S)
    try
        return model.set[name]
    catch
        err("Given set: $(name) does not exist in Model")
    end
end

"""
"""
function add_element!(model::Model, element::Element)
    push!(model.elements, element)
end

"""
"""
function add_node!(model::Model, node::Node)
    push!(model.nodes, node)
end

"""
"""
function add_set!(model::Model, set::Union{NodeSet, ElementSet})
    model.sets[set.name] = set
end

function add_set!{S <: AbstractString}(model::Model, ::Type{Val{:NSET}},
    name::S, ids::Vector{Integer})
    new_set = NodeSet(name, ids)
    add_set!(model, new_set)
end

function add_set!{S <: AbstractString}(model::Model, ::Type{Val{:ELSET}},
    name::S, ids::Vector{Integer})
    new_set = ElementSet(name, ids)
    add_set!(model, new_set)
end
