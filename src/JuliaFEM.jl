# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

importall Base

include("fields.jl")
export Field, DCTI, DVTI, DCTV, DVTV, CCTI, CVTI, CCTV, CVTV
include("types.jl")  # data types: Point, IntegrationPoint, ...
export AbstractPoint, Point, IntegrationPoint, IP, Node

### ELEMENTS ###
include("elements.jl") # common element routines
export Node, AbstractElement, Element, update!, get_connectivity, get_basis, get_dbasis, inside, get_local_coordinates
include("elements_lagrange.jl") # Continuous Galerkin (Lagrange) elements
export get_reference_coordinates, get_interpolation_polynomial
export Poi1,
       Seg2, Seg3,
       Tri3, Tri6, Tri7,
       Quad4, Quad8, Quad9,
       Tet4, Tet10,
       Wedge6,
       Hex8, Hex20, Hex27

include("elements_nurbs.jl")
export NSeg, NSurf, NSolid, is_nurbs

#include("hierarchical.jl") # P-elements

include("integrate.jl")  # default integration points for elements
export get_integration_points

include("sparse.jl")
export add!, SparseMatrixCOO, get_nonzero_rows

include("problems.jl") # common problem routines
export Problem, AbstractProblem, FieldProblem, BoundaryProblem,
       get_unknown_field_dimension, get_gdofs, Assembly,
       get_parent_field_name, get_elements

include("problems_elasticity.jl")
export Elasticity

include("problems_dirichlet.jl")
export Dirichlet

include("problems_heat.jl")
export Heat

export assemble!, postprocess!

function assemble!(problem::Problem, element::Element, time=0.0)
    assemble!(problem.assembly, problem, element, time)
end

### ASSEMBLY + SOLVE ###
include("assembly.jl")
include("solvers_utils.jl")
include("solvers.jl")
export AbstractSolver, Solver, Nonlinear, NonlinearSolver, Linear, LinearSolver,
       get_unknown_field_name, get_formulation_type, get_problems,
       get_field_problems, get_boundary_problems,
       get_field_assembly, get_boundary_assembly,
       initialize!, create_projection, eliminate_interior_dofs
include("solvers_modal.jl")
export Modal

include("optics.jl")
export find_intersection, calc_reflection, calc_normal

### Mortar methods ###
include("problems_mortar.jl")
include("problems_mortar_2d.jl")
include("problems_mortar_3d.jl")
include("problems_mortar_2d_autodiff.jl")
export calculate_normals,
       calculate_normals!,
       project_from_slave_to_master,
       project_from_master_to_slave,
       Mortar, get_slave_elements,
       get_polygon_clip

### Mortar methods, contact mechanics extension ###
include("problems_contact.jl")
include("problems_contact_2d.jl")
include("problems_contact_3d.jl")
include("problems_contact_2d_autodiff.jl")
export Contact

#=
module API
include("api.jl")
# export ....
end
=#

module Preprocess
include("preprocess.jl")
export create_elements, Mesh,
       add_node!, add_nodes!,
       add_element!, add_elements!,
       add_element_to_element_set!,
       add_node_to_node_set!,
       find_nearest_nodes,
       reorder_element_connectivity!
include("preprocess_abaqus_reader.jl")
export parse_abaqus, parse_section, parse_element_section,
       abaqus_read_mesh, abaqus_read_model
include("preprocess_aster_reader.jl")
export aster_create_elements, parse_aster_med_file, is_aster_mail_keyword,
       parse_aster_header, aster_parse_nodes, aster_renumber_nodes!,
       aster_renumber_elements!, aster_combine_meshes, aster_read_mesh,
       filter_by_element_set, filter_by_element_id, MEDFile
end

function get_mesh(mesh_name::AbstractString, args...; kwargs...)
    return get_mesh(Val{Symbol(mesh_name)}, args...; kwargs...)
end

function get_model(model_name::AbstractString, args...; kwargs...)
    return get_model(Val{Symbol(model_name)}, args...; kwargs...)
end

export get_mesh, get_model

module Postprocess
include("postprocess_utils.jl")
export calc_nodal_values!, 
       get_nodal_vector,
       get_nodal_dict,
       copy_field!
include("postprocess_xdmf.jl")
export XDMF, xdmf_new_result!, xdmf_save_field!, xdmf_save!
end
export Postprocessor

# This connects model from preprocess_abaqus_reader to 
# other JuliaFEM ecosystem and solves problem.
module Abaqus
include("abaqus.jl")
export abaqus_read_model
end

""" JuliaFEM testing routines. """
module Testing
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
end

export @test, @testset, @test_throws
#include("test.jl")
end

#=
module MaterialModels
include("vonmises.jl")
end
=#

#=
module Interfaces
include("interfaces.jl")
end
=#

end # module
