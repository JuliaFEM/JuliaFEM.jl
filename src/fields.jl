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

typealias FieldSet Dict{String, Field}

### Different field combinations and other typealiases

typealias DCTI Field{Discrete,   Constant, TimeInvariant}
typealias DVTI Field{Discrete,   Variable, TimeInvariant}
typealias DCTV Field{Discrete,   Constant, TimeVariant}
typealias DVTV Field{Discrete,   Variable, TimeVariant}
typealias CCTI Field{Continuous, Constant, TimeInvariant}
typealias CVTI Field{Continuous, Variable, TimeInvariant} # can be used to interpolate in spatial dimension
typealias CCTV Field{Continuous, Constant, TimeVariant} # can be used to interpolate in time
typealias CVTV Field{Continuous, Variable, TimeVariant}

# Discrete fields

type Increment{T}
    time :: Float64
    data :: T
end

typealias ScalarIncrement{T} Increment{T}
typealias VectorIncrement{T} Increment{Vector{T}}
typealias TensorIncrement{T} Increment{Matrix{T}}

""" Discrete, constant, time-invariant field. This is constant in both spatial
direction and time direction, i.e. df/dX = 0 and df/dt = 0.

This is the most basic type of field having no anything special functionality.

Examples
--------

julia> f = DCTI()
julia> update!(f, 1.0)

Multiplying by constant works:

julia> 2*f
2.0

Interpolation in time direction gives the same constant:

julia> f(1.0)
1.0

By default, when calling Field with scalar, DCTI is assumed, i.e.

julia> Field(0.0) == DCTI(0.0)
true

"""
function DCTI()
    return DCTI(nothing)
end

function Field()
    return DCTI()
end

function Field(data)
    return DCTI(data)
end

function ==(x::DCTI, y::DCTI)
    return ==(x.data, y.data)
end

function ==(x::DCTI, y)
    return ==(x.data, y)
end

function isapprox(x::DCTI, y::DCTI)
    isapprox(x.data, y.data)
end

function isapprox(x::DCTI, y)
    isapprox(x.data, y)
end

""" "Spatial" length of constant field is always 1. """
function length(field::DCTI)
    return 1
end

""" Kind of spatial interpolation of DCTI. """
function Base.:*(N::Matrix, f::DCTI)
    @assert length(N) == 1
    return N[1]*f.data
end

function update!(field::DCTI, data)
    field.data = data
end

""" Interpolate time-invariant field in time direction. """
function (field::DCTI)(time::Float64)
    return field.data
end

""" Discrete, variable, time-invariant field. This is constant in time direction,
but not in spatial direction, i.e. df/dt = 0 but df/dX != 0. The basic structure
of data is Vector, and it is implicitly assumed that length of field matches to
the number of shape functions, so that interpolation in spatial direction works.

Examples
--------
"""
function DVTI()
    return DVTI([])
end

""" For vector data, DVTI is automatically created.

julia> DVTI([1.0, 2.0]) == Field([1.0, 2.0])
true

"""
function Field(data::Vector)
    return DVTI(data)
end

""" For dictionary data, DVTI is automatically created.

Define e.g. nodal coordinates in dictionary
julia> X = Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0])
julia> Field(X) == DVTI(X)

"""
function Field(data::Dict)
    return DVTI(data)
end

function ==(x::DVTI, y::DVTI)
    return ==(x.data, y.data)
end

function isapprox(x::DVTI, y)
    return isapprox(x.data, y)
end

""" Default slicing of field.

julia> f = DVTI([1.0, 2.0])
julia> f[1]
1.0

"""
function getindex(field::DVTI, i::Int64)
    return field.data[i]
end

""" Multi-slicing of field.

julia> f = DVTI([1.0, 2.0, 3.0])
julia> f[[1, 3]]
[1.0, 3.0]

"""
function getindex(field::DVTI, I::Array{Int64, 1})
    return [field.data[i] for i in I]
end

function length(field::DVTI)
    return length(field.data)
end

function Base.:+(f1::DVTI, f2::DVTI)
    return DVTI(f1.data + f2.data)
end

function Base.:-(f1::DVTI, f2::DVTI)
    return DVTI(f1.data - f2.data)
end

function update!(field::DVTI, data::Union{Vector, Dict})
    field.data = data
end

""" Take scalar product of DVTI and constant T. """
function Base.:*(T::Number, field::DVTI)
    return DVTI(T*field.data)
end

""" Take dot product of DVTI field and vector T. Vector length must match to the
field length and this can be used mainly for interpolation purposes, i.e., u = ∑ Nᵢuᵢ.
"""
function Base.:*(T::Vector, f::DVTI)
    @assert length(T) <= length(f)
    return sum([T[i]*f[i] for i=1:length(T)])
end

""" Take outer product of DVTI field and matrix T. """
function Base.:*(T::Matrix, f::DVTI)
    n, m = size(T)
    return sum([kron(T[:,i], f[i]') for i=1:m])'
end

function vec(field::DVTI)
    return [field.data...;]
end

""" Interpolate time-invariant field in time direction. """
function (field::DVTI)(time::Float64)
    return field
end

""" Create a similar DVTI field from vector data. 

julia> f1 = DVTI(Vector[[1.0, 2.0], [3.0, 4.0]])
julia> f2 = similar(f1, [2.0, 3.0, 4.0, 5.0])
julia> f2 == DVTI(Vector[[2.0, 3.0], [4.0, 5.0]])
true

"""
function similar(field::DVTI, data::Vector)
    n = length(field)
    m = length(data)
    dim = round(Int, m/n)
    @assert dim*n == m
    new_data = reshape(data, dim, n)
    new_field = DVTI()
    new_field.data = [new_data[:,i] for i=1:n]
    return new_field
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


### Convenient functions to create fields

#function Base.convert(::Type{Field}, data)
#    return Field(data)
#end



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

function getindex(field::DCTV, i::Int64)
    return field.data[i]
end

function getindex(field::Field, i::Int64)
    return field.data[i]
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

for op = (:+, :*, :/, :-)
    @eval ($op)(increment::Increment, field::DCTI) = ($op)(increment.data, field.data)
    @eval ($op)(field::DCTI, increment::Increment) = ($op)(increment.data, field.data)
    @eval ($op)(field1::DCTI, field2::DCTI) = ($op)(field1.data, field2.data)
    @eval ($op)(field::DCTI, k::Number) = ($op)(field.data, k)
    @eval ($op)(k::Number, field::DCTI) = ($op)(field.data, k)
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
