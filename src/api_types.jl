# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

type NeumannBC
    set_name :: ASCIIString
    value :: Any
end

type DirichletBC 
    set_name :: ASCIIString
    value :: Any
end

typealias DisplacementBC DirichletBC
typealias TemperatureBC DirichletBC

typealias ForceBC NeumannBC
typealias HeatFluxBC NeumannBC

"""
"""
type Material
    name :: ASCIIString
    scalar_data :: Dict{ASCIIString, Float64}
end

#Material(name, data) = Material(name, Dict(data))
Material(name) = Material(name, Dict{ASCIIString, Float64}())
Material() = Material("", Dict{ASCIIString, Float64}())

function Base.setindex!{T <: AbstractString }(material::Material, val::Real, name::T)
    material.scalar_data[name] = val 
end


type Node 
    id :: Union{Integer, ASCIIString}
    coords :: Array{Float64, 1}
end

type Element
    id :: Union{Integer, ASCIIString, Void}
    connectivity :: Vector{Int64}
    element_type :: Symbol
    results :: Any # Dict{}
    material :: Any
end

Element(a, b, c) = Element(a, b, c, nothing, Material())

type NodeSet
    name :: ASCIIString
    nodes :: Vector{Int64}
end

"""
"""
type ElementSet
    name :: ASCIIString
    elements :: Vector{Union{Int64, ASCIIString}} 
    material :: Material
end

ElementSet(name::ASCIIString, elements::Vector{Element}) =
    ElementSet(name, map(x-> x.id, elements), Material())

ElementSet(name::ASCIIString, ids::Vector{Int64}) =
    ElementSet(name, ids, Material())

"""
Simulation
"""
type Simulation
    problem :: Symbol
    neumann_boundary_conditions :: Vector{NeumannBC}
    dirichlet_boundary_conditions :: Vector{DirichletBC}
    solver
    sets :: Vector{ASCIIString}
end

Simulation(a) = Simulation(a, NeumannBC[], DirichletBC[], nothing, ASCIIString[])


"""
Model type

Used for constructing the calculation model
"""
type Model
    name :: ASCIIString 
    nodes :: Dict{Union{Int64, ASCIIString}}
    elements :: Dict{Union{ASCIIString, Int64}, Element} 
    elsets :: Dict{ASCIIString, ElementSet}
    nsets :: Dict{ASCIIString, NodeSet}
    load_cases :: Dict{ASCIIString, Simulation}
    #settings :: Dict{AbstractString, Real}
end

function Model(name::ASCIIString, abq_input::Dict)
    model  = Model(name)
    nodes        = abq_input["nodes"]
    elements     = abq_input["elements"]
    element_sets = abq_input["elsets"]
    node_sets    = abq_input["nsets"]
    for id in keys(nodes)
        coords = nodes[id]
        add_node!(model, id, coords)
    end

    for id in keys(elements)
        data = elements[id]
        eltype = data["type"]
        conn = data["connectivity"]
#        println(eltype, " ", conn)
        add_element!(model, id, eltype, conn)
    end
    
    for name in keys(node_sets)
        ids = node_sets[name]
        add_node_set!(model, name, ids)
    end

    for name in keys(element_sets)
        ids = element_sets[name]
        add_element_set!(model, name, ids)
    end
    model
end

Model(name::ASCIIString) = Model(
    name,
    Dict{Union{Int64, ASCIIString}, Node}(),
    Dict{Union{Int64, ASCIIString}, Element}(),
    Dict{ASCIIString, NodeSet}(),
    Dict{ASCIIString, ElementSet}(),
    Dict{ASCIIString, Simulation}(),
)

function Base.setindex!(dict::Dict{Union{ASCIIString, Int64}, Node},
    vals::Vector{Float64}, idx::Union{ASCIIString, Int64})
    dict[idx] = Node(idx, vals)
end

