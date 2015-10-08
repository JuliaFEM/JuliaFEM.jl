# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

using ForwardDiff

""" Field is a fundamental type which holds some values in some time t """
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

"""
FieldSet is array of fields, each field maybe having different time and/or increment.
"""
typealias FieldSet Array{Field, 1}
""" Multiply fieldset with some vector x. """
Base.(:*)(x::Array{Float64, 1}, f::Array{Field}) = sum(x .* f)

""" Add new field to fieldset. """
function add_field!(fs::FieldSet, field::Field)
    push!(fs, field)
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


# convenient functions
""" Evaluate basis function in point ξ. """
call(b::Basis, xi) = b.basis(xi)
#""" Interpolate field (h*f)(ξ) """
#Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld
#""" Interpolate from set of fields with basis b, i.e. f(t) = b(t)*[f1, f2] """
#Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld
#""" Interpolate field f using basis b. """
#Base.(:*)(b::Basis, f::Field) = (x) -> b(x)*f
#Base.(:*)(b::Basis, f::Array{Field}) = (t) -> b(t)*f

