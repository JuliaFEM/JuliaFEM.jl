# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

using ForwardDiff


""" Field. """
type Field{T}
    time :: Float64
    increment :: Int64
    values :: Array{T, 1}
end

""" Initialize field. """
function Field(time, values)
    Field(time, 1, values)
end

""" Get length of a field (number of basis functions in practice). """
Base.length(f::Field) = length(f.values)

""" Get field discrete value at point i. """
Base.getindex(f::Field, i::Int64) = f.values[i]

""" Interpolate field h(ξ)*f = x*f """
Base.(:*)(x::Array{Float64, 1}, f::Field) = sum(x .* f.values)
Base.(:*)(x::Array{Float64, 2}, f::Field) = sum([f[i]*x[i,:] for i in 1:length(f)])

""" Interpolate field (h*f)(ξ) """
Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld

""" Multiply field with some constant k. """
Base.(:*)(k::Float64, f::Field) = Field(f.time, k*f.values)

""" Sum two fields. """
function Base.(:+)(f1::Field, f2::Field)
    @assert(f1.time == f2.time, "Cannot add fields: time mismatch, $(f1.time) != $(f2.time)")
    Field(f1.time, f1.values + f2.values)
end

""" Interpolate from set of fields x*[f1, f2] where x is evaluated basis. """
Base.(:*)(x::Array{Float64, 1}, f::Array{Field}) = sum(x .* f)

""" Interpolate from set of fields with basis b, i.e. f(t) = b(t)*[f1, f2] """
Base.(:*)(f::Function, fld::Field) = (x) -> f(x)*fld

"""
Interpolate a field from finite set of fields some time t ∈ R.
"""
function call(fields :: Array{Field, 1}, t::Float64)
    if t <= fields[1].time
        return fields[1]
    end
    if t >= fields[end].time
        return fields[end]
    end
    i = length(fields)
    while fields[i].time >= t
        i -= 1
    end
    if fields[i].time == t
        return fields[i]
    end
    #Logging.debug("doing linear interpolation between fields $i and $(i+1)")
    f1 = fields[i]
    t1 = f1.time
    f2 = fields[i+1]
    t2 = f2.time
    dt = t2 - t1
    nw = (t2-t)/dt*f1.values + (t-t1)/dt*f2.values
    f = Field(t, nw)
    return f
end
function call(field::Field, t::Float64)
    Field(t, field.increment, field.values)
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

""" Interpolate field f using basis b. """
Base.(:*)(b::Basis, f::Field) = (x) -> b(x)*f
Base.(:*)(b::Basis, f::Array{Field}) = (t) -> b(t)*f

""" Evaluate basis function in point ξ. """
call(b::Basis, xi) = b.basis(xi)

""" Get partial derivative of basis function. """
∂(h::Basis) = h.dbasisdxi
diff(h::Basis) = h.dbasisdxi
derivative(h::Basis) = h.dbasisdxi

