# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

abstract Field
abstract DiscreteField <: Field
abstract ContinuousField <: Field

### DEFAULT DISCRETE FIELD ###

# 1. Increment

type Increment{T} <: AbstractVector{T}
    data :: Vector{T}
end

function Base.size(increment::Increment)
    return size(increment.data)
end

function Base.linearindexing(::Type{Increment})
    return LinearFast()
end

function Base.getindex(increment::Increment, i::Int)
    return increment.data[i]
end

function Base.setindex!(increment::Increment, v, i::Int)
    increment.data[i] = v
end

function Base.dot(k::Number, increment::Increment)
    return k*increment
end

function Base.convert(::Type{Increment}, data::Number)
    return Increment([data])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 2})
    return Increment([data[:,i] for i=1:size(data, 2)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 3})
    return Increment([data[:,:,i] for i=1:size(data, 3)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 4})
    return Increment([data[:,:,:,i] for i=1:size(data, 4)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 5})
    return Increment([data[:,:,:,:,i] for i=1:size(data, 5)])
end

function Base.zeros(::Type{Increment}, T, dims...)
    return Increment(zeros(T, dims...))
end

""" Flatten increment to Vector.

Examples
--------

>>> inc = ones(Increment, 2, 4)
>>> vec(inc)
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

"""
function Base.vec(increment::Increment)
    return [increment...;]
end

function Base.similar{T}(increment::Increment, data::Vector{T})
    return Increment(reshape(data, round(Int, length(data)/length(increment)), length(increment)))
end

function Base.convert{T}(::Type{Vector{T}}, increment::Increment)
    return Increment[increment]
end

# 2. TimeStep

type TimeStep
    time :: Float64
    increments :: Vector{Increment}
end

function Base.size(timestep::TimeStep)
    return size(timestep.increments)
end

function Base.endof(timestep::TimeStep)
    return endof(timestep.increments)
end

function Base.length(timestep::TimeStep)
    return length(timestep.increments)
end

function Base.linearindexing(::Type{TimeStep})
    return Base.LinearFast()
end

function Base.getindex(timestep::TimeStep, i::Int)
    return timestep.increments[i]
end

#function TimeStep(data::Union{Number, Array}...)
#    return TimeStep(0.0, Increment[Increment(d) for d in data])
#end

function TimeStep()
    return TimeStep(0.0, [])
end

function TimeStep{T}(data::T...)
    return TimeStep(0.0, Increment[Increment(d) for d in data])
end

function Base.convert(::Type{TimeStep}, value::Number)
    return TimeStep(0.0, Increment[Increment(value)])
end

function Base.push!(timestep::TimeStep, increment::Increment)
    push!(timestep.increments, increment)
end

# FIXME: having some serious problems here to get tuple form working.

# 3. DefaultDiscreteField
immutable DefaultDiscreteField <: DiscreteField
    timesteps :: Vector{TimeStep}
#=
    function DefaultDiscreteField(data::Array)
        if (typeof(data) == Vector{Int64}) || (typeof(data) == Vector{Float64})
            new(TimeStep[TimeStep(data)])
        else
            new(data)
        end
    end
=#
end

#=
type DefaultDiscreteField <: DiscreteField
    timesteps :: Vector{TimeStep}
    function DefaultDiscreteField(data...)
        timesteps = TimeStep[]
        for (i, d) in enumerate(data)
            @debug("i = $i, d = $d")
            if isa(d, Tuple)
                # contains time vector
                increments = Increment[Increment(d[2])]
                push!(timesteps, TimeStep(d[1], increments))
            else
                increments = Increment[Increment(d)]
                push!(timesteps, TimeStep(i-1.0, increments))
            end
        end
        new(timesteps)
    end
end
=#


function Base.size(field::DefaultDiscreteField)
    return size(field.timesteps)
end

function Base.length(field::DefaultDiscreteField)
    return length(field.timesteps)
end

function Base.start(::DefaultDiscreteField)
    return 1
end

function Base.next(field::DefaultDiscreteField, state)
    return (field[state+1], state+1)
end

function Base.done(field::DefaultDiscreteField, state)
    return state > length(field)
end

function eltype(::Type{DefaultDiscreteField})
    return TimeStep
end

function Base.linearindexing(::Type{DefaultDiscreteField})
    return LinearFast()
end

function Base.getindex(field::DefaultDiscreteField, i::Int)
    return field.timesteps[i]
end

function Base.endof(field::DefaultDiscreteField)
    return endof(field.timesteps)
end

function Base.first(field::DefaultDiscreteField)
    return field[1][end]
end

function Base.last(field::DefaultDiscreteField)
    return field[end][end]
end

function Base.push!(field::DefaultDiscreteField, timestep::TimeStep)
    push!(field.timesteps, timestep)
end

"""Quickly create fields.

Examples
--------
>>> Field([1, 2])  # creates field with one timestep and vector value [1, 2]
>>> Field(1, 2) # creates field with two timesteps, each having scalar value
>>> Field([1, 2], [3, 4]) # creates field with two timesteps, each having vector value
>>> Field( (0.0, [1, 2]), (0.5, [3, 4]) ) # like above, but give time also
"""
function Base.convert(::Type{DefaultDiscreteField}, data...)
    timesteps = TimeStep[]
    for (i, d) in enumerate(data)
        if isa(d, Tuple)
            @debug("is tuple, has time, d = $d")
            # contains time vector
            increments = Increment[Increment(d[2])]
            push!(timesteps, TimeStep(d[1], increments))
        else
#           @debug("array without time, d = $d")
#           @debug(typeof(d))
            increments = Increment[Increment(d)]
            push!(timesteps, TimeStep(i-1.0, increments))
        end
    end
    field = DefaultDiscreteField(timesteps)
    return field
end

function Base.convert(::Type{DefaultDiscreteField}, data::Vector{TimeStep})
    field = DefaultDiscreteField(data)
#    @debug(field)
    return field
end

### CONTINUOUS FIELDS ###

type DefaultContinuousField <: ContinuousField
    field :: Function
end

function Base.call(field::DefaultContinuousField, xi::Vector, time::Number)
    return field.field(xi, time)
end

function Base.convert(::Type{DefaultContinuousField}, f::Function)
    return DefaultContinuousField(f)
end


### FIELDSET ###

typealias FieldSet Dict{ASCIIString, Field}

# 1. given numbers, arrays or tuples -> discrete field

function Base.convert(::Type{Field}, data::Union{Number, Array, Tuple}...)
    return DiscreteField(data...)
end

function Base.convert(::Type{DiscreteField}, data::Union{Number, Array, Tuple}...)
    return convert(DefaultDiscreteField, data...)
end

# 2. given function -> continuous field

function Base.convert(::Type{Field}, data::Function)
    return ContinuousField(data)
end

function Base.convert(::Type{ContinuousField}, data::Function)
    return convert(DefaultContinuousField, data)
end

