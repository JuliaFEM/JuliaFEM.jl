# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

module API
include("api.jl")

# export ....

end

module Preprocess
# include("abaqus_reader.jl") <-- ERROR: LoadError: LoadError: UndefVarError: Model not defined
include("aster_reader.jl")
end

module Postprocess
include("xdmf.jl")
end

""" JuliaFEM testing routines. """
module Test
include("test.jl")
end

module Core

import Base: +, -, /, *, push!, convert, getindex, setindex!, length, similar, call, vec, endof

"""
A very simple debugging macro. It prints debug message if environment variable
JULIAFEM_DEBUG is found.

Usage: instead of starting session `julia file.jl` do `JULIAFEM_DEBUG=1 julia file.jl`.
Or set `export JULIAFEM_DEBUG=1` for your `.bashrc`.
"""
macro debug(msg)
    if !haskey(ENV, "JULIAFEM_DEBUG")
        return
    end
    return :( println("DEBUG: ", $msg) )
end

function set_debug_on!()
    ENV["JULIAFEM_DEBUG"] = 1;
end

function set_debug_off!()
    pop!(ENV, "JULIAFEM_DEBUG");
end

export @debug, set_debug_on!, set_debug_off!

using ForwardDiff
autodiffcache = ForwardDiffCache()
# export derivative, jacobian, hessian

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

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solvers.jl")
include("directsolver.jl") # parallel sparse direct solver for non-linear problems

### MORTAR STUFF ###
include("mortar.jl")  # mortar projection
end

module Interfaces
include("interfaces.jl")
end

end # module
