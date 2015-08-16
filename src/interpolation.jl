# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Interpolate field variable using basis functions f for point ip.
This function tries to be as general as possible and allows interpolating
lot of different fields.

Parameters
----------
field :: Array{Number, dim}
  Field variable
basis :: Function
  Basis functions
ip :: Array{Number, 1}
  Point to interpolate
"""
function interpolate{T<:Real}(field::Array{T,1}, basis::Function, ip)
    result = dot(field, basis(ip))
    return result
end
function interpolate{T<:Real}(field::Array{T,2}, basis::Function, ip)
    m, n = size(field)
    bip = basis(ip)
    tmp = size(bip)
    if length(tmp) == 1
        ndim = 1
        nnodes = tmp[1]
    else
        ndim, nnodes = size(bip)
    end
    if ndim == 1
        if n == nnodes
            result = field * bip
        elseif m == nnodes
            result = field' * bip
        end
    else
      if n == nnodes
        result = bip' * field
    elseif m == nnodes
      result = bip' * field'
    end
    end
    if length(result) == 1
        result = result[1]
    end
    return result
end
function interpolate(e::Element, field::ASCIIString, x::Array{Float64,1}; derivative=false)
    return interpolate(e.attributes[field], derivative ? e.dbasis : e.basis, x)
end

