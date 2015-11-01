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
    size(increment.data)
end

function Base.linearindexing(::Type{Increment})
    LinearFast()
end

function Base.getindex(increment::Increment, i::Int)
    increment.data[i]
end

function Base.setindex!(increment::Increment, v, i::Int)
    increment.data[i] = v
end

function Base.dot(k::Number, increment::Increment)
    k*increment
end

function Base.convert(::Type{Increment}, data::Number)
    Increment([data])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 2})
    Increment([data[:,i] for i=1:size(data, 2)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 3})
    Increment([data[:,:,i] for i=1:size(data, 3)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 4})
    Increment([data[:,:,:,i] for i=1:size(data, 4)])
end

function Base.convert{T}(::Type{Increment}, data::Array{T, 5})
    Increment([data[:,:,:,:,i] for i=1:size(data, 5)])
end

function Base.zeros(::Type{Increment}, dims...)
    Increment(zeros(dims...))
end

function Base.vec(increment::Increment)
    [increment...;]
end

function Base.similar{T}(increment::Increment, data::Vector{T})
    Increment(reshape(data, round(Int, length(data)/length(increment)), length(increment)))
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
    Base.LinearFast()
end

function Base.getindex(timestep::TimeStep, i::Int)
    return timestep.increments[i]
end

#function TimeStep(data::Union{Number, Array}...)
#    return TimeStep(0.0, Increment[Increment(d) for d in data])
#end

function TimeStep()
    TimeStep(0.0, [])
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

# 3. DefaultDiscreteField
type DefaultDiscreteField <: DiscreteField
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

function Base.linearindexing(::Type{DefaultDiscreteField})
    return LinearFast()
end

function Base.getindex(field::DefaultDiscreteField, i::Int)
    field.timesteps[i]
end

function Base.length(field::DefaultDiscreteField)
    length(field.timesteps)
end

function Base.endof(field::DefaultDiscreteField)
    endof(field.timesteps)
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
#           @debug("is tuple, has time, d = $d")
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
    convert(DefaultDiscreteField, data...)
end

# 2. given function -> continuous field

function Base.convert(::Type{Field}, data::Function)
    return ContinuousField(data)
end

function Base.convert(::Type{ContinuousField}, data::Function)
    convert(DefaultContinuousField, data)
end

