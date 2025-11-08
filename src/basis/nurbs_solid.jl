# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

mutable struct NSolid <: AbstractBasis{3}
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

function length(basis::NSolid)
    nu = length(basis.knots_u) - basis.order_u - 1
    nv = length(basis.knots_v) - basis.order_v - 1
    nw = length(basis.knots_w) - basis.order_w - 1
    return nu*nv*nw
end

function size(basis::NSolid)
    return (3, length(basis))
end

function eval_basis!(basis::NSolid, N::Vector, xi::Vec{3})
    pu = basis.order_u
    pv = basis.order_v
    pw = basis.order_w
    tu = basis.knots_u
    tv = basis.knots_v
    tw = basis.knots_w
    weights = basis.weights
    nu = length(tu)-pu-1
    nv = length(tv)-pv-1
    nw = length(tw)-pw-1
    u, v, w = xi
    n = 1
    for i=1:nu
        for j=1:nv
            for k=1:nw
                A = NURBS(i,pu,u,tu)
                B = NURBS(j,pv,v,tv)
                C = NURBS(k,pw,w,tw)
                N[n] = weights[i,j,k]*A*B*C
                n += 1
            end
        end
    end
    N ./= sum(N)
    return N
end
