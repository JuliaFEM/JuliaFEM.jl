# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

#abstract AbstractField{T,N} <: AbstractArray{T,N}

abstract AbstractField

abstract DiscreteField <: AbstractField

abstract ContinuousField <: AbstractField
abstract TimeContinuousField <: ContinuousField
abstract SpatialContinuousField <: ContinuousField
abstract TimeAndSpatialContinuousField <: ContinuousField
# should we introduce time and spatial discontinuous fields
# for discontinuous galerkin?

### DEFAULT DISCRETE FIELD ###

# 1. Increment

# FIXME: This should be Vector.
#typealias Increment Vector
type Increment{T} <: AbstractVector{T}
    data :: Vector{T}
end
Base.size(increment::Increment) = Base.size(increment.data)
Base.linearindexing(::Type{Increment}) = Base.LinearFast()
Base.getindex(increment::Increment, i::Int) = increment.data[i]
Base.setindex!(increment::Increment, v, i::Int) = (increment.data[i] = v)
Base.similar{T}(increment::Increment, ::Type{T}) = Increment(similar(increment.data))
Base.dot(v::Number, i::Increment) = v*i

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
    [increment.data...;]
end
function Base.similar{T}(increment::Increment{Vector{T}}, data::Vector{T})
    Increment(reshape(data, round(Int, length(data)/length(increment)), length(increment)))
end

# 2. TimeStep

type TimeStep{T} <: AbstractVector{T}
    time :: Float64
    increments :: Vector{T}
end
Base.size(timestep::TimeStep) = Base.size(timestep.increments)
Base.linearindexing(::Type{TimeStep}) = Base.LinearFast()
Base.getindex(timestep::TimeStep, i::Int) = timestep.increments[i]

function Base.convert(::Type{TimeStep}, time::Number, increment::Increment)
    TimeStep(time, Increment[increment])
end

function Base.push!(timestep::TimeStep, increment::Increment)
    push!(timestep.increments, increment)
end

# 3. DefaultDiscreteField

type DefaultDiscreteField <: DiscreteField
    timesteps :: Vector{TimeStep}
end
Base.size(field::DefaultDiscreteField) = Base.size(field.timesteps)
Base.linearindexing(::Type{DefaultDiscreteField}) = Base.LinearFast()
Base.getindex(field::DefaultDiscreteField, i::Int) = field.timesteps[i]
Base.length(field::DefaultDiscreteField) = length(field.timesteps)
Base.endof(field::DefaultDiscreteField) = endof(field.timesteps)
Base.first(field::DefaultDiscreteField) = field[1][end]
Base.last(field::DefaultDiscreteField) = field[end][end]
function Base.push!(field::DefaultDiscreteField, timestep::TimeStep)
    push!(field.timesteps, timestep)
end


typealias Field DefaultDiscreteField

### CONTINUOUS FIELDS ###


# fix print_matrix
#function Base.print_matrix(::Base.AbstractIOBuffer, field::ContinuousField, args...)
    # TODO: anything nice to print?
#end

### FIELDSET ###

typealias FieldSet Dict{ASCIIString, AbstractField}

"""Quicky add discrete field to fieldset.

Examples
--------
>>> fs = FieldSet()
>>> fs["myfield"] = [1, 2, 3, 4]
"""
function Base.convert(::Type{AbstractField}, data::Union{Array, Number})
    increment = Increment(data)
    timestep = TimeStep(0.0, Increment[increment])
    field = DefaultDiscreteField(TimeStep[timestep])
    return field
end

""" Quicky add several time steps at once in tuple.

Examples
--------
>>> fs = FieldSet()
>>> fs["myfield"] = (0.0, [1, 2, 3, 4]), (0.5, [2, 3, 4, 5])

or

>>> fs["myfield"] = [1, 2, 3, 4], [2, 3, 4, 5]

"""
function Base.convert(::Type{AbstractField}, data::Tuple)
    timesteps = TimeStep[]
    for (i, timestep) in enumerate(data)
        if isa(timestep, Tuple)
            push!(timesteps, TimeStep(Float64(timestep[1]), Increment(timestep[2])))
        else
            push!(timesteps, TimeStep(Float64(i-1), Increment(timestep)))
        end
    end
    return DefaultDiscreteField(timesteps)
end

### BASIS ###

abstract AbstractBasis

""" Defined to dimensionless coordinate ξ∈[-1,1]^n. """
type SpatialBasis <: AbstractBasis
    basis :: Function
    dbasisdxi :: Function
end

typealias Basis SpatialBasis

""" Defined to to interval t∈[0, 1]. """
type TemporalBasis <: AbstractBasis
    basis :: Function
    dbasisdt :: Function
end
function TemporalBasis()
    basis(t) = [1-t, t]
    dbasis(t) = [-1, 1]
    return TemporalBasis(basis, dbasis)
end

function call(b::TemporalBasis, value::Number)
    b.basis(value)
end

function call(b::SpatialBasis, value::Vector)
    b.basis(value)
end

### INTERPOLATION IN TIME DOMAIN ###

function Base.call(field::Field, basis::TemporalBasis, time)
    # FieldSet -> Field -> TimeStep -> Increment -> data
    # special cases, -Inf, +Inf and ~0.0
    if time > field[end].time
        return field[end][end]
    end
    if (time < field[1].time) || abs(time-field[1].time) < 1.0e-12
        return field[1][end]
    end
    i = length(field)
    while field[i].time >= time
        i -= 1
    end
    field[i].time == time && return field[i][end]
    t1 = field[i].time
    t2 = field[i+1].time
    inc1 = field[i][end]
    inc2 = field[i+1][end]
    # TODO: may there be some reasons for "unphysical" jumps in
    # fields w.r.t time which should be taken account in some way?
    # i.e. dt between two fields → 0
    dt = t2 - t1
    b = basis.basis((time-t1)/dt)
    r = Increment[inc1, inc2]
    return dot(b, r)
end
function Base.call(field::DiscreteField, time)
    return Base.call(field, TemporalBasis(), time)
end

function Base.call(field::Field, basis::TemporalBasis, time,
                   derivative::Type{Val{:derivative}})
    # FieldSet -> Field -> TimeStep -> Increment -> data

    if length(field) == 1
        # just one timestep, time derivative cannot be evaluated.
        error("Field length = $(length(field)), cannot evaluate time derivative")
    end

    function eval_field(i, j)
        timesteps = TimeStep[field[i], field[j]]
        increments = Increment[timesteps[1][end], timesteps[2][end]]
        J = norm(timesteps[2].time - timesteps[1].time)
        dbasisdt = basis.dbasisdt( (time-timesteps[1].time)/J )
        return dot(dbasisdt, increments)/J
    end

    # special cases, +Inf, -Inf, ~0.0
    if (time > field[end].time) || isapprox(time, field[end].time)
        return eval_field(endof(field)-1, endof(field))
    end
    if (time < field[1].time) || isapprox(time, field[1].time)
        return eval_field(1, 2)
    end

    # search for a correct "bin" between time steps
    i = length(field)
    #while field[i].time >= time + 1.0e-12
    while (field[i].time > time) && !isapprox(field[i].time, time)
        i -= 1
    end

    if isapprox(field[i].time, time)
        # This is the hard case, maybe discontinuous time
        # derivative if linear approximation.
        # we are on the "mid node" in time axis
        field1 = eval_field(i-1,i)
        field2 = eval_field(i,i+1)
        return 1/2*(field1 + field2)
    end

    return eval_field(i, i+1)

end

### INTERPOLATION IN SPATIAL DOMAIN ###

function Base.call(increment::Increment, basis::SpatialBasis, xi::Vector)
    basis = basis.basis(xi)
    sum([basis[i]*increment[i] for i=1:length(increment)])
end

function Base.call(increment::Increment, basis::SpatialBasis, xi::Vector,
                   geometry::Increment, gradient::Type{Val{:gradient}})
    dbasis = basis.dbasisdxi(xi)
    J = sum([dbasis[:,i]*geometry[i]' for i=1:length(geometry)])
    grad = inv(J)*dbasis
    gradf = sum([grad[:,i]*increment[i]' for i=1:length(increment)])'
    return gradf
end

### INTEGRATIONPOINT ###

"""
Integration point

xi :: Array{Float64, 1}
    (dimensionless) coordinates of integration point
weight :: Float64
    Integration weight
attributes :: Dict{Any, Any}
    This is used to save internal variables of IP needed e.g. for incremental
    material models.
"""
type IntegrationPoint
    xi :: Vector
    weight :: Float64
    fields :: Dict{ASCIIString, FieldSet}
end
function IntegrationPoint(xi, weight)
    IntegrationPoint(xi, weight, Dict())
end

call(b::SpatialBasis, ip::IntegrationPoint) = b.basis(ip.xi)


