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

include("fields.jl")
export Field, DCTI, DVTI, DCTV, DVTV
include("types.jl")  # data types: Point, IntegrationPoint, ...
export AbstractPoint, Point, IntegrationPoint, IP, Node
#include("basis.jl")  # interpolation of discrete fields
#include("symbolic.jl") # a thin symbolic layer for fields

### ELEMENTS ###
include("elements.jl") # common element routines
export Node, AbstractElement, Element, update!, get_connectivity
include("lagrange_macro.jl") # Continuous Galerkin (Lagrange) elements generated using macro
export Seg2, Seg3, Tri3, Tri6, Quad4, Hex8, Tet4, Tet10

#include("hierarchical.jl") # P-elements
#include("mortar_elements.jl") # Mortar elements
#include("equations.jl")

include("integrate.jl")  # default integration points for elements
export get_integration_points

include("sparse.jl")

include("problems.jl") # common problem routines
export Problem, AbstractProblem, FieldProblem, BoundaryProblem,
       get_unknown_field_dimension

include("elasticity.jl") # elasticity equations
export Elasticity

include("dirichlet.jl")
export Dirichlet

include("heat.jl")
export Heat

export assemble, assemble!

function assemble!(problem::Problem, element::Element, time=0.0)
    assemble!(problem.assembly, problem, element, time)
end

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solver_utils.jl")
include("solvers.jl")
export Solver

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
include("abaqus_reader.jl")
include("preprocess_aster_reader.jl")
export aster_create_elements, parse_aster_med_file
end

module Postprocess
include("xdmf.jl")
export xdmf_new_temporal_collection, xdmf_new_grid,
       xdmf_new_mesh!, xdmf_new_nodal_field!,
       xdmf_save_model, xdmf_new_model, xdmf_dump
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
