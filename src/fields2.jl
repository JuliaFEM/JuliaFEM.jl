# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

abstract AbstractField

abstract Discrete <: AbstractField
abstract Continuous <: AbstractField
abstract Constant <: AbstractField
abstract Variable <: AbstractField
abstract TimeVariant <: AbstractField
abstract TimeInvariant <: AbstractField

type Field{A<:Union{Discrete,Continuous}, B<:Union{Constant,Variable}, C<:Union{TimeVariant,TimeInvariant}}
    data
end

# Different field combinations
typealias DCTI Field{Discrete,   Constant, TimeInvariant}
typealias DVTI Field{Discrete,   Variable, TimeInvariant}
typealias DCTV Field{Discrete,   Constant, TimeVariant}
typealias DVTV Field{Discrete,   Variable, TimeVariant}
typealias CCTI Field{Continuous, Constant, TimeInvariant}
typealias CVTI Field{Continuous, Variable, TimeInvariant} # can be used to interpolate in spatial dimension
typealias CCTV Field{Continuous, Constant, TimeVariant} # can be used to interpolate in time
typealias CVTV Field{Continuous, Variable, TimeVariant}

# Basic data structure for discrete field
type Increment{T}
    time :: Float64
    data :: T
end

typealias VectorIncrement Increment{Vector}

function Base.getindex{T}(increment::Increment{Vector{T}}, i::Int64)
    return increment.data[i]
end

# Basic data structure for continuous field
type Basis
    basis :: Function
    dbasis :: Function
end

# Functions simplifying definition of fields.

"""
All other data than vectors are considered as constant time invariant fields.
"""
function Field(data)
    DCTI(data)
end

"""
Vector data is considered as variable field time invariant field.
"""
function Field(data::Vector)
    DVTI(data)
end

"""
Data given in (time, value) pairs, where value is not vector, is considered as
constant time variant field.
"""
function Field{T}(data::Pair{Float64, T}...)
    increments = [Increment{T}(d[1], d[2]) for d in data]
    DCTV(increments)
end

"""
Data given in (time, value) pairs, where value is a vector, is considered as
variable time variant field.
"""
function Field{T}(data::Pair{Float64, Vector{T}}...)
    increments = [Increment{Vector{T}}(d[1], d[2]) for d in data]
    DVTV(increments)
end

""" Special case, constant time-variant vector, converted automatically. """
function Base.convert{T}(::Type{DCTV}, data::Pair{Float64, Vector{T}}...)
    increments = [Increment(d[1], d[2]) for d in data]
    DCTV(increments)
end

## Other field related functions

function Base.getindex(field::DVTV, i::Int64)
    return field.data[i]
end


### FIELDSET ###

typealias FieldSet Dict{ASCIIString, Field}
