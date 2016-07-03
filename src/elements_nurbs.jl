# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using ForwardDiff
# TODO: evaluate partial derivatives of basis functions without forwarddiff

""" NURBS segment. """
type NSeg <: AbstractElement
    order :: Int
    knots :: Vector{Float64}
    weights :: Vector{Float64}
end

function NSeg()
    NSeg(1,
        [-1.0, -1.0, 1.0, 1.0],
        ones(4))
end

type NSurf <: AbstractElement
    order_u :: Int
    order_v :: Int
    knots_u :: Vector{Float64}
    knots_v :: Vector{Float64}
    weights :: Matrix{Float64}
end

function NSurf()
    NSurf(1, 1,
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        ones(2, 2))
end

type NSolid <: AbstractElement
    order_u :: Int
    order_v :: Int
    order_w :: Int
    knots_u :: Vector{Float64}
    knots_v :: Vector{Float64}
    knots_w :: Vector{Float64}
    weights :: Array{Float64, 3}
end

function NSolid()
    NSolid(1, 1, 1,
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        ones(2, 2, 2))
end

function NURBS(i, p, u, t)
    p == 0 && return t[i] <= u <= t[i+1] ? 1.0 : 0.0
    anom = u-t[i]
    adenom = t[i+p]-t[i]
    a = isapprox(adenom, 0.0) ? 0.0 : anom/adenom
    bnom = t[i+p+1]-u
    bdenom = t[i+p+1]-t[i+1]
    b = isapprox(bdenom, 0.0) ? 0.0 : bnom/bdenom
    result = a*NURBS(i,p-1,u,t) + b*NURBS(i+1,p-1,u,t)
    return result
end

function get_basis(element::Element{NSeg}, xi::Vector, time)
    pu = element.properties.order
    tu = element.properties.knots
    w = element.properties.weights
    nu = length(tu)-pu-1
    u = xi[1]
    N = vec([w[j]*NURBS(j,pu,u,tu) for j=1:nu])'
    return N/sum(N)
end

function get_basis(element::Element{NSurf}, xi::Vector, time)
    pu = element.properties.order_u
    pv = element.properties.order_v
    tu = element.properties.knots_u
    tv = element.properties.knots_v
    w = element.properties.weights
    nu = length(tu)-pu-1
    nv = length(tv)-pv-1
    u, v = xi
    N = vec([w[i,j]*NURBS(i,pu,u,tu)*NURBS(j,pv,v,tv) for i=1:nu, j=1:nv])'
    return N / sum(N)
end

function get_basis(element::Element{NSolid}, xi::Vector, time)
    pu = element.properties.order_u
    pv = element.properties.order_v
    pw = element.properties.order_w
    tu = element.properties.knots_u
    tv = element.properties.knots_v
    tw = element.properties.knots_w
    w = element.properties.weights
    nu = length(tu)
    nv = length(tv)
    nw = length(tw)
    u, v, w = xi
    N = [w[i,j,k]*NURBS(i,pu,u,tu)*NURBS(j,pv,v,tv)*NURBS(k,pw,w,tw) for i=1:nu, j=1:nv, k=1:nw]
    return N / sum(N)
end

# TODO: evaluate partial derivatives of basis functions without forwarddiff
""" Evaluate partial derivatives of basis functions using ForwardDiff. """
function get_dbasis{E<:Union{NSeg, NSurf, NSolid}}(element::Element{E}, ip, time)
    xi = isa(ip, IP) ? ip.coords : ip
    basis(xi) = vec(get_basis(element, xi, time))
    return ForwardDiff.jacobian(basis, xi)'
end

function length(element::Element{NSeg})
    nu = length(element.properties.knots) - element.properties.order - 1
    return nu
end

function size(element::Element{NSeg})
    return (1, length(element))
end

function length(element::Element{NSurf})
    nu = length(element.properties.knots_u) - element.properties.order_u - 1
    nv = length(element.properties.knots_v) - element.properties.order_v - 1
    return nu*nv
end

function size(element::Element{NSurf})
    return (2, length(element))
end

function length(element::Element{NSolid})
    nu = length(element.properties.knots_u) - element.properties.order_u - 1
    nv = length(element.properties.knots_v) - element.properties.order_v - 1
    nw = length(element.properties.knots_w) - element.properties.order_w - 1
    return nu*nv*nw
end

function size(element::Element{NSolid})
    return (3, length(element))
end

function is_nurbs(element::Element)
    return false
end

function is_nurbs{E<:Union{NSeg, NSurf, NSolid}}(element::Element{E})
    return true
end

