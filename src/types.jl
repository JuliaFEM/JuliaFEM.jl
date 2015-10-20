# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

using ForwardDiff

""" Field is a fundamental data type which holds some values in some time t """
type Field{T}
    time :: Float64
    increment :: Int64
    values :: T
end
""" Initialize field. """
function Field(time, values)
    Field(time, 1, values)
end
""" Get length of a field (number of basis functions in practice). """
function Base.length(f::Field)
    length(f.values)
end
""" Push value to field. """
function Base.push!(f::Field, value)
    push!(f.values, value)
end
""" Get field discrete value at point i. """
function Base.getindex(f::Field, i::Int64)
    f.values[i]
end
""" Multiply field with some constant k. """
function Base.(:*)(k::Number, f::Field)
    Field(f.time, k*f.values)
end
""" Multiply field with some vector x. """
function Base.(:*)(x::Vector, f::Field)
    @assert length(x) == length(f)
    sum([f[i]*x[i] for i in 1:length(f)])
end
""" Multiply field with some matrix x. """
# function Base.(:*){T}(x::Matrix, f::Field{Vector{T}})
function Base.(:*)(x::Matrix, f::Field)
    sum([f[i]*x[i,:] for i in 1:length(f)])
end
""" Sum two fields. """
function Base.(:+)(f1::Field, f2::Field)
    @assert(f1.time == f2.time, "Cannot add fields: time mismatch, $(f1.time) != $(f2.time)")
    Field(f1.time, f1.values + f2.values)
end



""" FieldSet is set of fields, each field can have different time and/or increment. """
type FieldSet
    name :: Symbol
    fields :: Array{Field, 1}
end
""" Initializer for FieldSet. """
function FieldSet(field_name)
    FieldSet(Symbol(field_name), [])
end
function FieldSet()
    FieldSet(Symbol("unknown field"), [])
end
""" Add new field to fieldset. """
function Base.push!(fs::FieldSet, field::Field)
    push!(fs.fields, field)
end
""" Multiply fieldset with some vector x. """
Base.(:*)(x::Array{Float64, 1}, fs::FieldSet) = sum(x .* fs.fields)
""" Get length of a fieldset. """
function Base.length(fieldset::FieldSet)
    length(fieldset.fields)
end
""" Return ith field from fieldset. """
function Base.getindex(fieldset::FieldSet, i::Int64)
    fieldset.fields[i]
end
#""" Return last field from fieldset. """
function Base.endof(fieldset::FieldSet)
    length(fieldset)
end



""" Basis function. """
type Basis
    basis :: Function
    dbasisdxi :: Function
end
""" Constructor of basis function. """
function Basis(basis)
    Basis(basis, ForwardDiff.jacobian(basis))
end
""" Get partial derivative of basis function. """
diff(h::Basis) = h.dbasisdxi
derivative(h::Basis) = h.dbasisdxi


# convenient functions -- maybe this is not correct place for them
""" Evaluate basis function in point ξ. """
call(b::Basis, xi) = b.basis(xi)
#""" Interpolate field (h*f)(ξ) """
#Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld
#""" Interpolate from set of fields with basis b, i.e. f(t) = b(t)*[f1, f2] """
#Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld
#""" Interpolate field f using basis b. """
#Base.(:*)(b::Basis, f::Field) = (x) -> b(x)*f
#Base.(:*)(b::Basis, f::Array{Field}) = (t) -> b(t)*f

