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
export Node, AbstractElement, Element, update!, get_connectivity, get_basis, get_dbasis
include("lagrange_macro.jl") # Continuous Galerkin (Lagrange) elements generated using macro
export Seg2, Seg3, Tri3, Tri6, Quad4, Hex8, Tet4, Tet10
include("nurbs.jl")
export NSeg, NSurf, NSolid, is_nurbs

#include("hierarchical.jl") # P-elements
#include("mortar_elements.jl") # Mortar elements
#include("equations.jl")

include("integrate.jl")  # default integration points for elements
export get_integration_points

include("sparse.jl")
export add!, SparseMatrixCOO, get_nonzero_rows

include("problems.jl") # common problem routines
export Problem, AbstractProblem, FieldProblem, BoundaryProblem,
       get_unknown_field_dimension, get_gdofs, Assembly,
       get_parent_field_name, get_elements

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
export AbstractSolver, Solver, Nonlinear, 
       get_unknown_field_name, get_formulation_type,
       get_field_problems, get_boundary_problems,
       get_field_assembly, get_boundary_assembly
include("modal.jl")
export Modal

include("optics.jl")
export find_intersection, calc_reflection, calc_normal

### Mortar methods ###
include("mortar.jl")
export calculate_normals,
       calculate_normals!,
       project_from_slave_to_master,
       project_from_master_to_slave,
       Mortar

### Contact mechanics ###
#include("contact.jl")

# rest of things
include("utils.jl")
include("core.jl")

module API
include("api.jl")
# export ....
end

module Preprocess
include("preprocess.jl")
export create_elements
include("preprocess_abaqus_reader.jl")
include("preprocess_abaqus_reader_old.jl")
include("preprocess_aster_reader.jl")
export aster_create_elements, parse_aster_med_file, is_aster_mail_keyword,
       parse_aster_header, aster_parse_nodes, aster_renumber_nodes!,
       aster_renumber_elements!, aster_combine_meshes, aster_read_mesh,
       filter_by_element_set, filter_by_element_id
end

module Postprocess
include("postprocess_utils.jl")
export calc_nodal_values!
include("postprocess_xdmf.jl")
export XDMF, xdmf_new_result!, xdmf_save_field!, xdmf_save!
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
