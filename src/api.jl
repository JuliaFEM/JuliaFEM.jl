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
type Material
    name :: AbstractString
    scalar_data :: Dict{AbstractString, Float64}
    owners :: Vector
end

# Base.set_index(...) = ...
# m = Material(jotian)
# m["youngs modulus"] = 200e3
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
"""
type NodeSet
    name :: AbstractString
    node_ids :: Array{Integer, 1} 
end

"""
"""
type ElementSet
    name :: AbstractString
    element_ids :: Array{Integer, 1}
end

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
    material :: Dict{AbstractString, Material}
    load_cases :: Vector{LoadCase}
 #   settings :: Dict{AbstractString, Real}
end

Model() = Model(Node[],
                Element[],
                Dict{AbstractString, Union{NodeSet, ElementSet}}(),
                Dict{AbstractString, Material}(),
                LoadCase[],
                )

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
"""
function add_element!(model, element)
    push!(model.elements, element)
end

"""
"""
function add_node!(model, node)
    push!(model.nodes, node)
end

"""
"""
function add_set(model, set)
    model.sets[set.name] = set
end
