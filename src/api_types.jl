# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#abstract APIType
#
#abstract Mesh <: APIType
#abstract APIElement <: Mesh
#abstract APINode <: Mesh
#
#abstract APIProblem <: APIType
#abstract HeatProblem <: APIProblem
#abstract ElasticityProblem <: APIProblem
#
#abstract APISolver <: APIType
#abstract SolverLinear <: APISolver
# define these
#abstract Problem
#abstract Element

#abstract BoundaryCondition

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


type Node{T<:Real} <: APINode
    id :: Union{Integer, ASCIIString}
    coords :: Array{T, 1}
end

type Element
    id :: Union{Integer, ASCIIString, Void}
    connectivity :: Vector{Int64}
    element_type :: Symbol
    fields :: Any # Dict{}
end

#Element{I<:Integer}(eltype::Symbol, conn::Vector{I}) = Element(
#    nothing,
#    conn,
#    eltype )
#    
#Element{I<:Integer}(conn::Vector{I}, eltype::ASCIIString) = Element(
#    nothing,
#    conn,
#    eltype)

#"""
#Set of nodes. Holds name and the ids
#"""
#type NodeSet}
#    name :: AbstractString
#    node_ids :: Array{Integer, 1}
#end

# NodeSet(arr::Array{Int64, 1}) = myn(arr, Dict(zip(arr, collect(1:length(arr)))))


"""
"""
type ElementSet
    name :: ASCIIString
    elements :: Vector{Int64} 
    material :: Material
end#

ElementSet(name::ASCIIString, elements::Vector{Element}) =
    ElementSet(name, map(x-> x.id, elements), Material())

ElementSet(name::ASCIIString, ids::Vector{Int64}) =
    ElementSet(name, ids, Material())

type LoadCase{ P <: APIProblem }
    problem :: Type{P}
    boundary_conditions #:: Vector{Union{NeumannBC, DirichletBC}}
    solver
end

#LoadCase{P <: APIProblem}(a :: P) = LoadCase(a, Vector{Union{NeumannBC,DirichletBC}}())
LoadCase(a) = LoadCase(a, [], nothing)
#LoadCase{P <: Problem, B <: BoundaryCondition}(a :: P, b :: B) = LoadCase(a, Vector{Union{NeumannBC,DirichletBC}}([b]))
#LoadCase{P <: Problem, B <: BoundaryCondition}(a :: P, b :: Vector{B}) =
#LoadCase(a, Vector{Union{NeumannBC,DirichletBC}}(b))

"""
"""
type Model
    name 
    nodes 
    elements :: Dict{Union{ASCIIString, Int64}, Element} 
    elsets
    nsets
    load_cases
    #settings :: Dict{AbstractString, Real}
    #results
end

Model(name::ASCIIString, abq_input::Dict) = Model(
    name,
    abq_input["nodes"],
    abq_input["elements"],
    abq_input["elsets"],
    abq_input["nsets"],
    Dict())

Model(name::ASCIIString) = Model(
    name,
    Dict(),
    Dict{Union{ASCIIString, Int64}, Element}(),
    Dict(),
    Dict(),
    Dict(),
)

#function Base.setindex!(dicti::Dict{Union{ASCIIString, Int64}, Element}, el::Element, idx::Union{ASCIIString, Int64})
#    el.id = idx
#    dicti[idx] = el
#    return dicti
#end

#Model(name::ASCIIString) = Model(name,
#                                 Dict{Union{Int64, ASCIIString}, Node}(),
#                                 Dict{Union{Int64, ASCIIString}, Element}(),
#                                 Dict{AbstractString, 
#                                    Union{NodeSet, ElementSet}}(),
#                                 LoadCase[],
#)

