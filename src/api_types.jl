# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


type NeumannBC
    set_name :: String
    value :: Any
end

type DirichletBC 
    set_name :: String
    value :: Any
end

typealias DisplacementBC DirichletBC
typealias TemperatureBC DirichletBC

typealias ForceBC NeumannBC
typealias HeatFluxBC NeumannBC

"""
"""
type Material
    name :: String
    scalar_data :: Dict{String, Any}
end

#Material(name, data) = Material(name, Dict(data))
Material(name) = Material(name, Dict{String, Any}())
Material() = Material("", Dict{String, Any}())

function Base.setindex!{T <: AbstractString }(material::Material, val, name::T)
    material.scalar_data[name] = val 
end


type Node 
    id :: Union{Integer, String}
    coords :: Array{Float64, 1}
end

type Element
    id :: Union{Integer, String, Void}
    connectivity :: Vector{Int64}
    element_type :: Symbol
    results :: Any # Dict{}
    material :: Any
end

Element(a, b, c) = Element(a, b, c, nothing, Material())

type NodeSet
    name :: String
    nodes :: Vector{Int64}
end

"""
"""
type ElementSet
    name :: String
    elements :: Vector{Union{Int64, String}} 
    material :: Material
end

ElementSet(name::String, elements::Vector{Element}) =
    ElementSet(name, map(x-> x.id, elements), Material())

ElementSet(name::String, ids::Vector{Int64}) =
    ElementSet(name, ids, Material())

"""
Simulation
"""
type Simulation
    problem :: Symbol
    neumann_boundary_conditions :: Vector{NeumannBC}
    dirichlet_boundary_conditions :: Vector{DirichletBC}
    solver
    sets :: Vector{String}
end

Simulation(a) = Simulation(a, NeumannBC[], DirichletBC[], nothing, String[])


"""
Model type

Used for constructing the calculation model
"""
type Model
    name :: String 
    nodes :: Dict{Union{Int64, String}}
    elements :: Dict{Union{String, Int64}, Element} 
    elsets :: Dict{String, ElementSet}
    nsets :: Dict{String, NodeSet}
    load_cases :: Dict{String, Simulation}
    renum_nodes :: Dict{Union{Int64, String}, Int64}
    #settings :: Dict{AbstractString, Real}
end

function Model(name::String, abq_input::Dict)
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

Model(name::String) = Model(
    name,
    Dict{Union{Int64, String}, Node}(),
    Dict{Union{Int64, String}, Element}(),
    Dict{String, NodeSet}(),
    Dict{String, ElementSet}(),
    Dict{String, Simulation}(),
    Dict{Union{Int64, String}, Int64}()
)

function Base.setindex!(dict::Dict{Union{String, Int64}, Node},
    vals::Vector{Float64}, idx::Union{String, Int64})
    dict[idx] = Node(idx, vals)
end

