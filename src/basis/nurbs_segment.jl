# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

""" NURBS segment. """
mutable struct NSeg <: AbstractBasis{1}
    order :: Int
    knots :: Vector{Float64}
    weights :: Vector{Float64}
end

function NSeg()
    NSeg(1,
        [-1.0, -1.0, 1.0, 1.0],
        ones(4))
end

function length(basis::NSeg)
    nu = length(basis.knots) - basis.order - 1
    return nu
end

function size(basis::NSeg)
    return (1, length(basis))
end

function eval_basis!(basis::NSeg, N::Vector, xi::Vec{1})
    pu = basis.order
    tu = basis.knots
    w = basis.weights
    nu = length(tu)-pu-1
    u = xi[1]
    for j=1:nu
        N[j] = w[j]*NURBS(j,pu,u,tu)
    end
    N ./= sum(N)
    return N
end
