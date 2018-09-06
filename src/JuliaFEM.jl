# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    JuliaFEM.jl - an open source solver for both industrial and academia usage

The JuliaFEM software library is a framework that allows for the distributed
processing of large Finite Element Models across clusters of computers using
simple programming models. It is designed to scale up from single servers to
thousands of machines, each offering local computation and storage. The basic
design principle is: everything is nonlinear. All physics models are nonlinear
from which the linearization are made as a special cases.

# Examples

Typical workflow to use JuliaFEM to solve a partial differential equations,
is 1) read mesh, 2) create elements, 3) update element properties, 4) create
problems, 5) create analysis, and 6) run analysis. A simple linear static
analysis of elastic block is clarifying these steps.

```julia
using JuliaFEM
```

1. Usually the first thing to do is to create a geometry of domain. Typically
   this is done by reading a mesh file from disk. Currently JuliaFEM supports
   reading a mesh from Code Aster file format (using `aster_read_mesh`) and
   from ABAQUS file format (using `abaqus_read_mesh`).

```julia
mesh = aster_read_mesh("mesh.med")
```

2. Next step is to create one or several sets of elements from a mesh. This is
   done using a function `create_elements(mesh, set_name)`.

```julia
body_elements = create_elements(mesh, "body")
traction_elements = create_elements(mesh, "traction")
bc_elements = create_elements(mesh, "bc")
```

3. In JuliaFEM, all properties of elements are given using so called fields.
   Fields can depend from time or spatial coordinates elements. In special cases
   field is constant in time, spatial or both directions. Updating fields to
   element is done using `update!`-function.

```julia
update!(body_elements, "youngs modulus", 210.0e3)
update!(body_elements, "poissons ratio", 0.3)
update!(traction_elements, "surface pressure", 100.0)
update!(bc_elements, "displacement 1", 0.0)
update!(bc_elements, "displacement 2", 0.0)
update!(bc_elements, "displacement 3", 0.0)
```

4. The physics considered to be solve is given using `Problem`. Problem type can
   be e.g. `Elasticity` for hyperelasticity, `Heat` for solving the heat equation
   and so on. Problems are defined by giving the problem type as first argument,
   problem name in second argument and the last argument is giving the dimension
   of problem, meaning degrees of freedom connected to each node. After problems
   are created, elements are added to them by using function `add_elements!`.

```julia
body = Problem(Elasticity, "body", 3)
traction = Problem(Elasticity, "traction", 3)
bc = Problem(Dirichlet, "bc", 3, "displacement")
add_elements!(body, body_elements)
add_elements!(traction, traction_elements)
add_elements!(bc, bc_elements)
```

5. After geometry and physics is defined, we next define what kind of analysis
   are we going to perform. Analysis can be, for example, quasistatic analysis,
   analysis of dynamics of system, natural frequency analysis and so on. Some
   other special analysis types also exists, like performing model dimension
   reduction by creating super-elements or running optimization loop, given
   geometry, another analysis and initial conditions. For simplicity, we now
   create a linear quasistatic analysis of given problems. Problems are added
   to analysis using `add_problems!`.

```julia
analysis = Analysis(Linear)
add_problems!(analysis, body, traction, bc)
```

6. The last thing to do is to request the results of analysis to be written to
   disk for later use and actually perform the analysis. Currently, Xdmf output
   is supported, which can then be read using ParaView.

```julia
xdmf_output = Xdmf("analysis_results")
add_results_writer!(analysis, xdmf_output)
run!(analysis)
close(xdmf)
```

After analysis is ready, types and variables can be accessed using REPL or
Jupyter notebook for further postprocessing. Simulation can also be written
into a function to be a part of a larger analysis process. For more information
about JuliaFEM, please visit our website at

    www.juliafem.org

"""
module JuliaFEM

using SparseArrays, LinearAlgebra, Statistics
using Reexport, ForwardDiff, LightXML, HDF5

@reexport using FEMBase
import FEMBase: get_unknown_field_name, get_unknown_field_dimension,
                assemble!, update!, initialize!
using FEMBase: get_problems

using TimerOutputs
export @timeit, print_timer

import Base: getindex, setindex!, convert, length, size, isapprox,
             similar, start, first, next, done, last, endof, vec,
             ==, +, -, *, /, haskey, copy, push!, isempty, empty!,
             append!, read, copy

using AbaqusReader
using AsterReader

@reexport using HeatTransfer
include("problems_elasticity.jl")
export Elasticity
include("materials_plasticity.jl")
export plastic_von_mises
include("problems_dirichlet.jl")
export Dirichlet

export assemble!, postprocess!

# Structural elements: beams
@reexport using FEMBeam

### Mortar methods ###

@reexport using MortarContact2D
@reexport using MortarContact2DAD

include("problems_mortar.jl")
include("problems_mortar_3d.jl")
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
include("problems_contact_3d.jl")
#include("problems_contact_3d_autodiff.jl")
export Contact

module Preprocess
end

using FEMBase, SparseArrays, LinearAlgebra
include("preprocess.jl")
export create_elements, Mesh, add_node!, add_nodes!,
       add_element_to_element_set!, add_node_to_node_set!,
       find_nearest_nodes, find_nearest_node, reorder_element_connectivity!,
       create_node_set_from_element_set!, filter_by_element_set
include("preprocess_abaqus_reader.jl")
export abaqus_read_mesh, create_surface_elements, create_nodal_elements
include("preprocess_aster_reader.jl")
export aster_read_mesh

# Postprocess module

module Postprocess
end

include("postprocess_utils.jl")
export calc_nodal_values!, get_nodal_vector, get_nodal_dict, copy_field!,
       calculate_area, calculate_center_of_mass, calculate_second_moment_of_mass,
       extract

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
