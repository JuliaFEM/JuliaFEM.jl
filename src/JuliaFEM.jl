# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# __precompile__()

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

using FEMBase
import FEMBase: get_unknown_field_name, get_unknown_field_dimension,
                assemble!, update!, initialize!


# from other packages TimerOutputs.jl and Logging.jl
using TimerOutputs
export @timeit, print_timer
using Logging
export info, debug

import Base: getindex, setindex!, convert, length, size, isapprox,
             similar, start, first, next, done, last, endof, vec,
             ==, +, -, *, /, haskey, copy, push!, isempty, empty!,
             append!, sparse, full, read

module Testing
using Base.Test
export @test, @testset, @test_throws
end

using AbaqusReader
using AsterReader
include("problems_elasticity.jl")
export Elasticity
include("materials_plasticity.jl")
export plastic_von_mises
include("problems_dirichlet.jl")
export Dirichlet
include("problems_heat.jl")
export Heat
export assemble!, postprocess!
### Mortar methods ###
include("problems_mortar.jl")
include("problems_mortar_2d.jl")
include("problems_mortar_3d.jl")
include("problems_mortar_2d_autodiff.jl")
export calculate_normals, calculate_normals!, project_from_slave_to_master,
       project_from_master_to_slave, Mortar, get_slave_elements,
       get_polygon_clip
include("io.jl")
export Xdmf, h5file, xmffile, xdmf_filter, new_dataitem, update_xdmf!, save!
include("solvers.jl")
export AbstractSolver, Solver, Nonlinear, NonlinearSolver, Linear, LinearSolver,
       get_unknown_field_name, get_formulation_type, get_problems,
       get_field_problems, get_boundary_problems,
       get_field_assembly, get_boundary_assembly,
       initialize!, create_projection, eliminate_interior_dofs,
       is_field_problem, is_boundary_problem
include("solvers_modal.jl")
export Modal
include("problems_contact.jl")
include("problems_contact_2d.jl")
include("problems_contact_3d.jl")
include("problems_contact_2d_autodiff.jl")
#include("problems_contact_3d_autodiff.jl")
export Contact

# Preprocess module

module Preprocess
using FEMBase
include("preprocess.jl")
export create_elements, Mesh, add_node!, add_nodes!, add_element!,
       add_elements!, add_element_to_element_set!, add_node_to_node_set!,
       find_nearest_nodes, find_nearest_node, reorder_element_connectivity!,
       create_node_set_from_element_set!, filter_by_element_set
include("preprocess_abaqus_reader.jl")
export abaqus_read_mesh, create_surface_elements, create_nodal_elements
include("preprocess_aster_reader.jl")
export aster_read_mesh
end

# Postprocess module

module Postprocess
using FEMBase
using FEMBase: get_elements
include("postprocess_utils.jl")
export calc_nodal_values!, get_nodal_vector, get_nodal_dict, copy_field!,
       calculate_area, calculate_center_of_mass,
       calculate_second_moment_of_mass, extract
end

include("deprecations.jl")

export SparseMatrixCOO, SparseVectorCOO, optimize!, resize_sparse
export DCTI, DVTI, DCTV, DVTV, CCTI, CVTI, CCTV, CVTV, Increment
export FieldProblem, BoundaryProblem, Problem, Node, Element, Assembly
export Poi1, Seg2, Seg3, Tri3, Tri6, Tri7, Quad4, Quad8, Quad9,
       Tet4, Tet10, Pyr5, Wedge6, Wedge15, Hex8, Hex20, Hex27
export update!, add_elements!, get_unknown_field_name, add!,
       is_field_problem, is_boundary_problem, get_gdofs,
       initialize!, get_integration_points, group_by_element_type,
       get_unknown_field_dimension, get_connectivity
export get_nonzero_rows, get_local_coordinates, inside, IP, get_element_type,
       get_elements, AbstractProblem, IntegrationPoint, filter_by_element_type,
       get_element_id, get_nonzero_columns, resize_sparse, resize_sparsevec

end
