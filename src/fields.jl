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

### Basic data structure for discrete field

type Increment{T}
    time :: Float64
    data :: T
end

function Base.convert{T}(::Type{Increment{T}}, data::Pair{Float64,T})
    return Increment{T}(data[1], data[2])
end

function Base.convert{T}(::Type{Increment{Vector{Vector{T}}}}, data::Pair{Float64, Matrix{T}})
    time = data[1]
    content = data[2]
    return Increment(time, Vector{T}[content[:,i] for i=1:size(content,2)])
end

function Base.getindex{T}(increment::Increment{Vector{T}}, i::Int64)
    return increment.data[i]
end

function Base.(:*)(d, increment::Increment)
    return d*increment.data
end

### Basic data structure for continuous field

type Basis
    basis :: Function
    dbasis :: Function
end

function Base.call(basis::Basis, xi::Vector)
    basis.basis(xi)
end

function Base.call(basis::Basis, xi::Vector, ::Type{Val{:grad}})
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

function Field{T}(data::Pair{Float64, Vector{T}}...)
    return DVTV([Increment{Vector{T}}(d[1], d[2]) for d in data])
end

function Base.convert{T}(::Type{DCTV}, data::Pair{Float64, Vector{T}}...)
    return DCTV([Increment{Vector{T}}(d[1], d[2]) for d in data])
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

function Base.getindex(field::DVTV, i::Int64)
    return field.data[i]
end

function Base.push!(field::DCTV, data::Pair)
    push!(field.data, data)
end

function Base.push!(field::DVTV, data::Pair)
#    info("field.data = \n$(field.data)")
#    info("data = \n$data")
    push!(field.data, data)
end

function Base.getindex(field::DVTV, i::Int64)
    return field.data[i]
end

function Base.getindex(field::DVTI, i::Int64)
    return field.data[i]
end

function Base.getindex(field::DCTV, i::Int64)
    return field.data[i]
end

function Base.getindex(field::Field, i::Int64)
    return field.data[i]
end

function Base.length(field::DVTI)
    return length(field.data)
end

function Base.length(field::DCTI)
    return 1
end

function Base.length(field::DVTV)
    return length(field.data)
end

function Base.length(field::DCTV)
    return length(field.data)
end

for op = (:+, :*, :/, :-)
    @eval ($op)(increment::Increment, field::DCTI) = ($op)(increment.data, field.data)
    @eval ($op)(field::DCTI, increment::Increment) = ($op)(increment.data, field.data)
    @eval ($op)(field1::DCTI, field2::DCTI) = ($op)(field1.data, field2.data)
    @eval ($op)(field::DCTI, k) = ($op)(field.data, k)
    @eval ($op)(k, field::DCTI) = ($op)(field.data, k)
end

function Base.vec(field::DVTI)
    return [field.data...;]
end

function Base.vec(field::DCTV)
    info("trying to vectorize $field")
    error("does not make sense")
end

function Base.endof(field::Field)
    return endof(field.data)
end

#function Base.similar{T}(field::DVTI, data::Vector{T})
#    return Increment(reshape(data, round(Int, length(data)/length(increment)), length(increment)))
#end

function Base.similar{T}(field::DVTI, data::Vector{T})
    n = length(field.data)
    data = reshape(data, round(Int, length(data)/n), n)
    newdata = Vector[data[:,i] for i=1:n]
    return typeof(field)(newdata)
end

function Base.start(::DVTI)
    return 1
end

function Base.next(f::DVTI, state)
    return f.data[state], state+1
end

function Base.done(f::DVTI, s)
    return s > length(f.data)
end

### Accessing continuous fields

function Base.call(field::CVTI, xi::Vector)
    field.data(xi)
end

function Base.call(field::CVTI, xi::Vector, ::Type{Val{:grad}})
    field.data(xi, Val{:grad})
end

function Base.convert(::Type{Basis}, field::CVTI)
    return field.data
end

function Base.call(field::CCTV, time::Number)
    return field.data(time)
end

### Interpolation 

""" Interpolate time-invariant field in time direction. """
function Base.call(field::DVTI, time::Float64)
    return field
end
function Base.call(field::DCTI, time::Float64)
    return field
end
function Base.call(field::CVTI, time::Float64)
    return field.data()
end
function Base.call(field::CCTI, time::Float64)
    return field.data()
end

""" Interpolate time-variant field in time direction. """
function Base.call(field::DCTV, time::Float64)
    for i=reverse(1:length(field))
        if isapprox(field[i].time, time)
            return DCTI(field[i].data)
        end
    end
    info(field.data)
    info(time)
    error("interpolate DCTV: not implemented yet")
end

function Base.call(field::DVTV, time::Float64, time_extrapolation::Symbol=:linear)
    for i=reverse(1:length(field))
        if isapprox(field[i].time, time)
            return DVTI(field[i].data)
        end
    end
    info(field.data)
    info(time)
    error("interpolate DVTV: not implemented yet")
end

""" Interpolate constant field in spatial dimension. """
function Base.call(basis::CVTI, field::DCTI, xi::Vector)
    return field.data
end

""" Interpolate variable field in spatial dimension. """
function Base.call(basis::CVTI, values::DVTI, xi::Vector)
    N = basis(xi)
    return sum([N[i]*values[i] for i=1:length(N)])
end

function Base.call(basis::CVTI, geometry::DVTI, xi::Vector, ::Type{Val{:grad}})
    dbasis = basis(xi, Val{:grad})
#    J = sum([dbasis[:,i]*geometry[i]' for i=1:length(geometry)])
    J = sum([kron(dbasis[:,i], geometry[i]') for i=1:length(geometry)])
    invJ = isa(J, Vector) ? inv(J[1]) : inv(J)
    grad = invJ * dbasis
    return grad
end

function Base.call(basis::CVTI, geometry::DVTI, values::DVTI, xi::Vector, ::Type{Val{:grad}})
    grad = call(basis, geometry, xi, Val{:grad})
#    gradf = sum([grad[:,i]*values[i]' for i=1:length(geometry)])'
    gradf = sum([kron(grad[:,i], values[i]') for i=1:length(values)])'
    return length(gradf) == 1 ? gradf[1] : gradf
end

function Base.call(basis::CVTI, xi::Vector, time::Number)
    call(basis, xi)
end

function Base.(:*)(grad::Matrix{Float64}, field::DVTI)
    return sum([kron(grad[:,i], field[i]') for i=1:length(field)])'
end

### FIELDSET ###

typealias FieldSet Dict{ASCIIString, Field}

