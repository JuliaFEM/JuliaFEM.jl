# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM

import Base: +, -, /, *, push!, convert, getindex, setindex!, length, similar, call, vec, endof, append!

using ForwardDiff
autodiffcache = ForwardDiffCache()
# export derivative, jacobian, hessian

using JLD

""" Simple linspace extension to arrays.

Examples
--------
>>> linspace([0.0], [1.0], 3)
3-element Array{Array{Float64,1},1}:
 [0.0]
 [0.5]
 [1.0]

"""
function Base.linspace{T<:Array}(X1::T, X2::T, n)
    [1/2*(1-ti)*X1 + 1/2*(1+ti)*X2 for ti in linspace(-1, 1, n)]
end

function Base.resize!(A::SparseMatrixCSC, m::Int64, n::Int64)
    (n == A.n) && (m == A.m) && return
    @assert n >= A.n
    @assert m >= A.m
    append!(A.colptr, A.colptr[end]*ones(Int, m-A.m))
    A.n = n
    A.m = m
end


# fields, see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb
include("fields.jl")
#include("basis.jl")  # interpolation of discrete fields
include("symbolic.jl") # a thin symbolic layer for fields
include("types.jl")  # type definitions

### ELEMENTS ###
include("elements.jl")
include("lagrange.jl") # Lagrange elements
#include("hierarchical.jl") # P-elements
#include("mortar_elements.jl") # Mortar elements

### EQUATIONS ###
include("integrate.jl")  # default integration points for elements
include("sparse.jl")
include("problems.jl")
include("equations.jl")

### FORMULATIION ###
include("dirichlet.jl")
include("heat.jl")
include("elasticity.jl")
include("linear_elasticity.jl")

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solvers.jl")
include("directsolver.jl") # parallel sparse direct solver for non-linear problems

### MORTAR STUFF ###
include("mortar.jl")  # mortar projection

include("abaqus_reader_old.jl")

# rest of things
include("utils.jl")
