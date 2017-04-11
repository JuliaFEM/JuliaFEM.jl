# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias Node Vector{Float64}

abstract AbstractPoint

type Point{P<:AbstractPoint}
    id :: Int
    weight :: Float64
    coords :: Vector{Float64}
    fields :: Dict{AbstractString, Field}
    properties :: P
end

function setindex!{T}(point::Point, val::Pair{Float64, T}, field_name)
    point.fields[field_name] = Field(val)
end

function getindex(point::Point, field_name)
    return point.fields[field_name]
end

function getindex(point::Point, idx::Int)
    return point.coords[idx]
end

function haskey(point::Point, field_name)
    return haskey(point.fields, field_name)
end

function (point::Point)(field_name, time=0.0)
    point.fields[field_name](time).data
end

function update!{T}(point::Point, field_name, val::Pair{Float64, T})
    if haskey(point, field_name)
        update!(point[field_name], val)
    else
        point[field_name] = val
    end
end

#= TODO: in future
type Node <: AbstractPoint
end

type MaterialPoint <: AbstractPoint
end
=#

type IntegrationPoint <: AbstractPoint
end

typealias IP Point{IntegrationPoint}

function IP(id, weight, coords)
    return IP(id, weight, coords, Dict(), IntegrationPoint())
end

typealias SparseMatrixFEM{Tv,Ti<:Integer} SparseMatrixCOO{Tv,Ti}
typealias SparseVectorFEM{Tv,Ti<:Integer} SparseVector{Tv,Ti}

"""
General linearized problem to solve
    (K₁+K₂)Δu  +   C1'*Δλ = f₁+f₂
         C2Δu  +     D*Δλ = g
"""
type Assembly

    M :: SparseMatrixCOO  # mass matrix

    # for field assembly
    K :: SparseMatrixCOO   # stiffness matrix
    Kg :: SparseMatrixCOO  # geometric stiffness matrix
    f :: SparseMatrixCOO   # force vector
    fg :: SparseMatrixCOO  #

    # for boundary assembly
    C1 :: SparseMatrixCOO
    C2 :: SparseMatrixCOO
    D :: SparseMatrixCOO
    g :: SparseMatrixCOO
    c :: SparseMatrixCOO

    u :: Vector{Float64}  # solution vector u
    u_prev :: Vector{Float64}  # previous solution vector u
    u_norm_change :: Real  # change of norm in u

    la :: Vector{Float64}  # solution vector la
    la_prev :: Vector{Float64}  # previous solution vector u
    la_norm_change :: Real # change of norm in la

    removed_dofs :: Vector{Int64} # manually remove dofs from assembly
end

function Assembly()
    return Assembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        [], [], Inf,
        [], [], Inf,
        [])
end

function empty!(assembly::Assembly)
    empty!(assembly.K)
    empty!(assembly.Kg)
    empty!(assembly.f)
    empty!(assembly.fg)
    empty!(assembly.C1)
    empty!(assembly.C2)
    empty!(assembly.D)
    empty!(assembly.g)
    empty!(assembly.c)
end

function isempty(assembly::Assembly)
    T = isempty(assembly.K)
    T &= isempty(assembly.Kg)
    T &= isempty(assembly.f)
    T &= isempty(assembly.fg)
    T &= isempty(assembly.C1)
    T &= isempty(assembly.C2)
    T &= isempty(assembly.D)
    T &= isempty(assembly.g)
    T &= isempty(assembly.c)
    return T
end
