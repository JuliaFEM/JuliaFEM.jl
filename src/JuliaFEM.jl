# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

#using Logging
#@Logging.configure(level=DEBUG)
#using Lexicon

macro debug(msg)
    return :( println("DEBUG: ", $msg) )
end

using ForwardDiff
autodiffcache = ForwardDiffCache()

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

include("types.jl")  # type definitions
#include("interpolate.jl")  # interpolation routines

### ELEMENTS ###
include("elements.jl")
include("lagrange.jl") # Lagrange elements
#include("hierarchical.jl") # P-elements

### EQUATIONS ###
include("integrate.jl")  # default integration points for elements
include("equations.jl")
include("problems.jl")

### FORMULATIION ###
include("dirichlet.jl")
include("heat.jl")
include("elasticity.jl")

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solvers.jl")

# PRE AND POSTPROCESS
include("xdmf.jl")
include("abaqus_reader.jl")

end # module

FEM = JuliaFEM

