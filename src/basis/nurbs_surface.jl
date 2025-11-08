# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

mutable struct NSurf <: AbstractBasis{2}
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

function length(basis::NSurf)
    nu = length(basis.knots_u) - basis.order_u - 1
    nv = length(basis.knots_v) - basis.order_v - 1
    return nu*nv
end

function size(basis::NSurf)
    return (2, length(basis))
end

function eval_basis!(basis::NSurf, N::Vector, xi::Vec{2})
    pu = basis.order_u
    pv = basis.order_v
    tu = basis.knots_u
    tv = basis.knots_v
    w = basis.weights
    nu = length(tu)-pu-1
    nv = length(tv)-pv-1
    u, v = xi
    n = 1
    for i=1:nu
        for j=1:nv
            N[n] = w[i,j]*NURBS(i,pu,u,tu)*NURBS(j,pv,v,tv)
            n += 1
        end
    end
    N ./= sum(N)
    return N
end
