# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

importall Base
using ForwardDiff
using JLD
autodiffcache = ForwardDiffCache()
# export derivative, jacobian, hessian

include("common.jl")
typealias Node Vector{Float64}

include("fields.jl")
export DCTI
#include("basis.jl")  # interpolation of discrete fields
#include("symbolic.jl") # a thin symbolic layer for fields
#include("types.jl")  # type definitions

### ELEMENTS ###
include("elements.jl") # common element routines
export Element
include("lagrange_macro.jl") # Continuous Galerkin (Lagrange) elements generated using macro
export Quad4

#include("hierarchical.jl") # P-elements
#include("mortar_elements.jl") # Mortar elements
#include("equations.jl")

include("integrate.jl")  # default integration points for elements
include("sparse.jl")

include("problems.jl") # common problem routines
export Problem

include("elasticity.jl") # elasticity equations
export Elasticity

include("dirichlet.jl")
include("heat.jl")
export assemble

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solver_utils.jl")
include("solvers.jl")

### MORTAR STUFF ###
include("mortar.jl")  # mortar projection

include("abaqus_reader_old.jl")

# rest of things
include("utils.jl")
include("core.jl")

module API
include("api.jl")
# export ....
end

module Preprocess
#=
macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end
=#
include("abaqus_reader.jl")
include("preprocess_aster_reader.jl")
end

module Postprocess
include("xdmf.jl")
end

""" JuliaFEM testing routines. """
module Test
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
end
export @test, @testset, @test_throws
#include("test.jl")
end

module MaterialModels
include("vonmises.jl")
end

module Interfaces
include("interfaces.jl")
end

end # module
