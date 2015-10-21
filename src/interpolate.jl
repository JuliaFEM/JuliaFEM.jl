# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM: Basis, Field, FieldSet, diff


"""
Interpolate field u using basis N in point xi.
"""
function interpolate{T}(N::Basis, u::Field{Vector{T}}, xi::Array{Float64,1})
    N(xi)*u
end
"""
Interpolate field u using basis N in set of points xi. Convenient function.
"""
function interpolate{T}(N::Basis, u::Field{Vector{T}}, xis::Array{Array{Float64,1},1})
    T[N(xi)*u for xi in xis]
end
function interpolate{T}(N::Basis, u::Field{T}, xi::Array{Float64,1})
    u.values
end

"""
Interpolate a field from fieldset for some time t.
"""
function interpolate(fields::FieldSet, t::Number)
    if length(fields) == 0
        throw("Empty set of fields: $fields")
    end
    if t <= fields[1].time
        return Field(t, fields[1].values)
    end
    if t >= fields[end].time
        return Field(t, fields[end].values)
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

function dinterpolate(N::Basis, u::Field, xi::Array{Float64, 1})
    dN = diff(N)
    dN(xi)*u
end
