# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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

typealias FieldSet Dict{AbstractString, Field}

### Basic data structure for discrete field

type Increment{T}
    time :: Float64
    data :: T
end

function convert{T}(::Type{Increment{T}}, data::Pair{Float64,T})
    return Increment{T}(data[1], data[2])
end

function convert{T}(::Type{Increment{Vector{Vector{T}}}}, data::Pair{Float64, Matrix{T}})
    time = data[1]
    content = data[2]
    return Increment(time, Vector{T}[content[:,i] for i=1:size(content,2)])
end

function getindex{T}(increment::Increment{Vector{T}}, i::Int64)
    return increment.data[i]
end

function Base.:*(d, increment::Increment)
    return d*increment.data
end

### Basic data structure for continuous field

type Basis
    basis :: Function
    dbasis :: Function
end

function (basis::Basis)(xi::Vector)
    basis.basis(xi)
end

function (basis::Basis)(xi::Vector, ::Type{Val{:grad}})
    basis.dbasis(xi)
end

### Different field combinations and other typealiases

typealias DCTI Field{Discrete,   Constant, TimeInvariant}
typealias DVTI Field{Discrete,   Variable, TimeInvariant}
typealias DCTV Field{Discrete,   Constant, TimeVariant}
typealias DVTV Field{Discrete,   Variable, TimeVariant}
typealias CCTI Field{Continuous, Constant, TimeInvariant}
typealias CVTI Field{Continuous, Variable, TimeInvariant} # can be used to interpolate in spatial dimension
typealias CCTV Field{Continuous, Constant, TimeVariant} # can be used to interpolate in time
typealias CVTV Field{Continuous, Variable, TimeVariant}

typealias ScalarIncrement{T} Increment{T}
typealias VectorIncrement{T} Increment{Vector{T}}
typealias TensorIncrement{T} Increment{Matrix{T}}

typealias DiscreteField      Union{DCTI, DVTI, DCTV, DVTV}
typealias ContinuousField    Union{CCTI, CVTI, CCTV, CVTV}
typealias ConstantField      Union{DCTI, DCTV, CCTI, CCTV}
typealias VariableField      Union{DVTI, DVTV, CVTI, CVTV}
typealias TimeInvariantField Union{DCTI, DVTI, CCTI, CVTI}
typealias TimeVariantField   Union{DCTV, DVTV, CCTV, CVTV}


### Convenient functions to create fields

#function Base.convert(::Type{Field}, data)
#    return Field(data)
#end

function Field(data)
    return DCTI(data)
end

function Field(data::Vector)
    return DVTI(data)
end

function Field{T}(data::Pair{Float64, T}...)
    return DCTV([Increment{T}(d[1], d[2]) for d in data])
end
#=
function Field{T}(data::Pair{Float64, Vector{T}}...)
    return DVTV([Increment{Vector{T}}(d[1], d[2]) for d in data])
end

function Field{T}(data::Pair{Float64, Dict{Int64, T}}...)
    return DVTV([Increment{Dict{Int64, T}}(d[1], d[2]) for d in data])
end
=#

function Field{T<:Union{Vector, Dict}}(data::Pair{Float64, T}...)
    return DVTV([Increment{T}(d[1], d[2]) for d in data])
end

function Field(data::Dict)
    return DVTI(data)
end

function convert{T}(::Type{DCTV}, data::Pair{Real, Vector{T}}...)
    return DCTV([Increment{Vector{T}}(d[1], d[2]) for d in data])
end

""" Create new discrete, constant, time variant field.

Examples
--------
julia> t0 = 0.0; t1=1.0; y0 = 0.0; y1 = 1.0
julia> f = DCTV(t0 => y0, t1 => y1)

"""
#function convert{T,v<:Real}(::Type{DCTV}, data::Pair{v, T}...)
#    return DCTV([Increment(d[1],d[2]) for d in data])
#end
function DCTV(data::Pair...)
    return DCTV([Increment(d[1],d[2]) for d in data])
end

function Field(func::Function)
    if method_exists(func, Tuple{})
        return CCTI(func)
    elseif method_exists(func, Tuple{Float64})
        return CCTV(func)
    elseif method_exists(func, Tuple{Vector})
        return CVTI(func)
    elseif method_exists(func, Tuple{Vector, Number})
        return CVTV(func)
    else
        error("no proper definition found for function: check methods.")
    end
end

function CVTI(basis::Function, dbasis::Function)
    return CVTI(Basis(basis, dbasis))
end

function Field(basis::Function, dbasis::Function)
    return CVTI(basis, dbasis)
end

### Accessing and manipulating discrete fields

function getindex(field::DVTV, i::Int64)
    return field.data[i]
end

function push!(field::DCTV, data::Pair)
    push!(field.data, data)
end

function push!(field::DVTV, data::Pair)
    push!(field.data, data)
end

function getindex(field::DVTI, i::Int64)
    return field.data[i]
end

function getindex(field::DVTI, I::Array{Int64, 1})
    return [field.data[i] for i in I]
end

function getindex(field::DCTV, i::Int64)
    return field.data[i]
end

function getindex(field::Field, i::Int64)
    return field.data[i]
end

function length(field::DVTI)
    return length(field.data)
end

function length(field::DCTI)
    return 1
end

function length(field::DVTV)
    return length(field.data)
end

function length(field::DCTV)
    return length(field.data)
end

function first(field::Union{DCTV, DVTV})
    return field[1]
end

function isapprox(f1::DCTI, f2::DCTI)
    isapprox(f1.data, f2.data)
end

for op = (:+, :*, :/, :-)
    @eval ($op)(increment::Increment, field::DCTI) = ($op)(increment.data, field.data)
    @eval ($op)(field::DCTI, increment::Increment) = ($op)(increment.data, field.data)
    @eval ($op)(field1::DCTI, field2::DCTI) = ($op)(field1.data, field2.data)
    @eval ($op)(field::DCTI, k::Number) = ($op)(field.data, k)
    @eval ($op)(k::Number, field::DCTI) = ($op)(field.data, k)
end

function Base.:+(f1::DVTI, f2::DVTI)
    return DVTI(f1.data + f2.data)
end

function Base.:-(f1::DVTI, f2::DVTI)
    return DVTI(f1.data - f2.data)
end

function Base.:*{T<:Real}(c::T, field::DVTI)
    return DVTI(c*field.data)
end

function Base.:*(N::Matrix, f::DCTI)
    return f.data*N'
end



#   Multiply DVTI field with another vector T. Vector length
#   must match to the field length and this can be used mainly
#   for interpolation purposes, i.e., u = ∑ Nᵢuᵢ
function Base.:*(T::Vector, f::DVTI)
    @assert length(T) <= length(f)
    return sum([T[i]*f[i] for i=1:length(T)])
end

function vec(field::DVTI)
    return [field.data...;]
end

function vec(field::DCTV)
    error("trying to vectorize $field does not make sense")
end

function endof(field::Field)
    return endof(field.data)
end

#function Base.similar{T}(field::DVTI, data::Vector{T})
#    return Increment(reshape(data, round(Int, length(data)/length(increment)), length(increment)))
#end

function similar{T}(field::DVTI, data::Vector{T})
    n = length(field.data)
    data = reshape(data, round(Int, length(data)/n), n)
    newdata = Vector[data[:,i] for i=1:n]
    return typeof(field)(newdata)
end

function start(::DVTI)
    return 1
end

function next(f::DVTI, state)
    return f.data[state], state+1
end

function done(f::DVTI, s)
    return s > length(f.data)
end

""" Update time-dependent fields with new values.

Examples
--------

julia> f = Field(0.0 => 1.0)
julia> update!(f, 1.0 => 2.0)

Now field has two (time, value) pairs: (0.0, 1.0) and (1.0, 2.0)

Notes
-----
Time vector is assumed to be ordered t_i-1 < t_i < t_i+1. If updating
field with already existing time the old value is replaced with new one.

"""
function update!{T}(field::Union{DCTV, DVTV}, val::Pair{Float64, T})
    time, data = val
    if isapprox(last(field).time, time)
        last(field).data = data
    else
        push!(field.data, Increment(val...))
    end
end

function update!{T}(field::Union{DCTI, DVTI}, val::T)
    field.data = val
end

### Accessing continuous fields

function (field::CVTI)(xi::Vector)
    return field.data(xi)
end

function (field::CVTV)(xi, time::Float64)
    return field.data(xi, time)
end

function (field::CVTI)(xi::Vector, ::Type{Val{:Grad}})
    return field.data(xi, Val{:Grad})
end

function (field::CCTV)(time::Float64)
    return field.data(time)
end

function convert(::Type{Basis}, field::CVTI)
    return field.data
end

### Interpolation

""" Interpolate time-invariant field in time direction. """
function (field::DVTI)(time::Float64)
    return field
end
function (field::DCTI)(time::Float64)
    return field.data
end
function (field::CVTI)(time::Float64)
    return field.data()
end
function (field::CCTI)(time::Float64)
    return field.data()
end

""" Interpolate constant time-variant field in time direction. """
function (field::DCTV)(time::Real)
    time < first(field).time && return DCTI(first(field).data)
    time > last(field).time && return DCTI(last(field).data)
    for i=reverse(1:length(field))
        isapprox(field[i].time, time) && return DCTI(field[i].data)
    end
    for i=reverse(2:length(field))
        t0 = field[i-1].time
        t1 = field[i].time
        if t0 < time < t1
            y0 = field[i-1].data
            y1 = field[i].data
            dt = t1-t0
            new_data = y0*(1-(time-t0)/dt) + y1*(1-(t1-time)/dt)
            return DCTI(new_data)
        end
    end
    error("interpolate DCTV: unknown failure when interpolating $(field.data) for time $time")
end

function (field::DVTV)(time::Float64)
    time < first(field).time && return DVTI(first(field).data)
    time > last(field).time && return DVTI(last(field).data)
    for i=reverse(1:length(field))
        isapprox(field[i].time, time) && return DVTI(field[i].data)
    end
    for i=reverse(2:length(field))
        t0 = field[i-1].time
        t1 = field[i].time
        if t0 < time < t1
            y0 = field[i-1].data
            y1 = field[i].data
            dt = t1-t0
            new_data = y0*(1-(time-t0)/dt) + y1*(1-(t1-time)/dt)
            return DVTI(new_data)
        end
    end
    error("interpolate DVTV: unknown failure when interpolating $(field.data) for time $time")
end

""" Interpolate constant field in spatial dimension. """
function (basis::CVTI)(field::DCTI, xi::Vector)
    return field.data
end

""" Interpolate variable field in spatial dimension. """
function (basis::CVTI)(values::DVTI, xi::Vector)
    N = basis(xi)
    return sum([N[i]*values[i] for i=1:length(N)])
end

function (basis::CVTI)(geometry::DVTI, xi::Vector, ::Type{Val{:grad}})
    dbasis = basis(xi, Val{:grad})
#    J = sum([dbasis[:,i]*geometry[i]' for i=1:length(geometry)])
    J = sum([kron(dbasis[:,i], geometry[i]') for i=1:length(geometry)])
    invJ = isa(J, Vector) ? inv(J[1]) : inv(J)
    grad = invJ * dbasis
    return grad
end

function (basis::CVTI)(geometry::DVTI, values::DVTI, xi::Vector, ::Type{Val{:grad}})
    grad = basis(geometry, xi, Val{:grad})
#    gradf = sum([grad[:,i]*values[i]' for i=1:length(geometry)])'
    gradf = sum([kron(grad[:,i], values[i]') for i=1:length(values)])'
    return length(gradf) == 1 ? gradf[1] : gradf
end

function (basis::CVTI)(xi::Vector, time::Number)
    basis(xi)
end

function Base.:*(grad::Matrix, field::DVTI)
    n, m = size(grad)
    return sum([kron(grad[:,i], field[i]') for i=1:m])'
end

function DVTV(data::Pair{Float64, Vector}...)
    return DVTV([Increment(d[1], d[2]) for d in data])
end

function start(f::DVTV)
    return start(f.data)
end

function next(f::DVTV, state)
    return next(f.data, state)
end

function done(f::DVTV, state)
    return done(f.data, state)
end

""" Return time vector from time variable field. """
function keys(field::DVTV)
    return Float64[increment.time for increment in field]
end

function setindex!(field::Field, val, idx::Int64)
    field.data[idx] = val
end
