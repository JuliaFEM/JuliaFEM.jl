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

# Import Base functions FIRST before defining any methods
import Base: getindex, setindex!, convert, length, size, isapprox,
   similar, first, last, vec,
   ==, +, -, *, /, haskey, copy, push!, isempty, empty!,
   append!, read

using SparseArrays, LinearAlgebra
using Logging  # For mesh readers
using Tensors  # For basis functions (Vec type)

# Removed for minimal dependency approach:
# - Calculus: Symbolic basis generation (basis/create_basis.jl, basis/subs.jl commented out)
# - ForwardDiff: Only used in tutorial notebooks for plasticity
# - HDF5/LightXML: I/O functionality (io.jl, AsterReader already commented out)
# - Arpack: Modal analysis (solvers_modal.jl commented out)

# No-op timing macro (TimerOutputs removed for minimal deps)
macro timeit(args...)
   return esc(args[end])
end

# import FEMSparse  # Consolidated into src/sparse/
# import FEMQuad   # Consolidated into src/quadrature.jl

# Note: Consolidating FEMBase and FEMBasis into JuliaFEM
# Previously: @reexport using FEMBase
# Now: Include files directly below

# ============================================================================
# TOPOLOGY: Reference element geometries (NEW - separation of concerns)
# ============================================================================
include("topology/topology.jl")       # Abstract topology interface
# 1D elements
include("topology/seg2.jl")
include("topology/seg3.jl")
# 2D triangular elements
include("topology/tri3.jl")
include("topology/tri6.jl")
include("topology/tri7.jl")
# 2D quadrilateral elements
include("topology/quad4.jl")
include("topology/quad8.jl")
include("topology/quad9.jl")
# 3D tetrahedral elements
include("topology/tet4.jl")
include("topology/tet10.jl")
# 3D hexahedral elements
include("topology/hex8.jl")
include("topology/hex20.jl")
include("topology/hex27.jl")
# 3D pyramid elements
include("topology/pyr5.jl")
# 3D wedge/prism elements
include("topology/wedge6.jl")
include("topology/wedge15.jl")

# Export topology types (new names without hardcoded node counts)
export AbstractTopology, dim, reference_coordinates, edges, faces
export Point  # 0D
export Segment  # 1D (was Seg2/Seg3, but those had node counts)
export Triangle, Quadrilateral  # 2D (was Tri3/Tri6/Tri7, Quad4/Quad8/Quad9)
export Tetrahedron, Hexahedron, Pyramid, Wedge  # 3D (was Tet4/Tet10, Hex8/Hex20/Hex27, Pyr5, Wedge6/Wedge15)

# Export old names as deprecated aliases (for backwards compatibility)
export Seg2, Seg3
export Tri3, Tri6, Tri7  # All map to Triangle
export Quad4, Quad8, Quad9  # All map to Quadrilateral
export Tet4, Tet10  # Both map to Tetrahedron
export Hex8, Hex20, Hex27  # All map to Hexahedron
export Pyr5  # Maps to Pyramid
export Wedge6, Wedge15  # Both map to Wedge

# Note: Node count is no longer in topology! It comes from basis:
# - Triangle + Lagrange{Triangle, 1} → 3 nodes
# - Triangle + Lagrange{Triangle, 2} → 6 nodes
# - Quadrilateral + Lagrange{Quadrilateral, 1} → 4 nodes
# - Quadrilateral + Lagrange{Quadrilateral, 2} → 9 nodes

# ============================================================================
# QUADRATURE: Low-level integration point data (consolidated from FEMQuad.jl)
# ============================================================================
include("quadrature.jl")

# ============================================================================
# INTEGRATION: High-level integration schemes (NEW - separation of concerns)
# ============================================================================
include("integration/integration.jl")  # Abstract integration interface, IntegrationPoint
include("integration/gauss.jl")        # Gauss-Legendre quadrature

export AbstractIntegration, IntegrationPoint, integration_points, npoints
export Gauss

# ============================================================================
# BASIS: Interpolation schemes (consolidated from FEMBasis.jl)
# ============================================================================
include("basis/abstract.jl")
include("basis/subs.jl")  # Symbolic substitution (includes minimal simplify from SymDiff.jl)
include("basis/vandermonde.jl")

# Lagrange basis functions - auto-generated file contains all basis types
# Now generates methods for parametric Lagrange{T,P} type instead of separate types
include("basis/lagrange_generator.jl")
include("basis/lagrange_generated.jl")  # Auto-generated by: julia --project=. src/basis/lagrange_generator.jl

# Export basis types and functions
export AbstractBasis, Lagrange, Serendipity
export nnodes, get_reference_element_coordinates, eval_basis!, eval_dbasis!

include("basis/nurbs.jl")
# OLD NURBS basis files - commented out during AbstractBasis refactoring
# These use AbstractBasis{dim} syntax which conflicts with new non-parametric AbstractBasis
# include("basis/nurbs_segment.jl")  # NSeg <: AbstractBasis{1}
# include("basis/nurbs_surface.jl")  # NSurf <: AbstractBasis{2}
# include("basis/nurbs_solid.jl")    # NSolid <: AbstractBasis{3}
# TODO: Rewrite for new AbstractBasis (non-parametric)
# include("basis/math.jl")  # Uses AbstractBasis{dim} throughout (jacobian, grad, interpolate, etc.)
# TODO: Rewrite math functions for new AbstractBasis
# TEMPORARY: Define minimal jacobian function for testing
function jacobian(B::AbstractBasis, X::Vector{<:Vec}, xi::Vec)
   dB = eval_dbasis!(B, xi)
   @assert length(X) == length(dB)
   # Compute J = dX/dξ: rows are physical dims, columns are parametric dims
   # J[i,j] = ∂X_i/∂ξ_j = sum_k X_k[i] * dN_k/dξ_j
   dim_physical = length(first(X))
   dim_parametric = length(xi)

   # Build Jacobian matrix manually for embedding case (e.g., 1D element in 3D space)
   # Result is a dim_physical × dim_parametric matrix
   J_data = zeros(dim_physical, dim_parametric)
   @inbounds for k in 1:length(X)
      for i in 1:dim_physical
         for j in 1:dim_parametric
            J_data[i, j] += X[k][i] * dB[k][j]
         end
      end
   end

   # Convert to Tensor (note: Tensor{2,N} is N×N, but we need dim_physical×dim_parametric)
   # For now, return as Matrix
   return J_data
end

# Consolidate FEMBase.jl into src/ (Phase 1 continued)
# Order matters: fields → types → sparse → elements → integrate → problems → assembly
include("fields/fields.jl")          # Field system (DCTI, DVTI, etc.)
include("core_types.jl")             # Node, IP, IntegrationPoint

# Compatibility shim: Create FEMBase module for vendor packages EARLY
# This must come before preprocess.jl or any code that uses FEMBase.something
include("fembase_compat.jl")

include("sparse/sparse.jl")          # SparseMatrixCOO, SparseVectorCOO
include("elements/elements.jl")      # Element type and interface
include("elements/elements_lagrange.jl")  # OLD - uses AbstractBasis{0} (Poi1)
# include("elements/integrate.jl")     # OLD - references NSeg, Poi1, etc.
include("assembly/problems.jl")      # Problem types
include("assembly/assembly.jl")      # Assembly framework
include("solvers/solvers_base.jl")   # Base solver types
include("analysis.jl")               # Analysis and AbstractResultsWriter
include("deprecated_fembase.jl")     # Deprecated/legacy methods from FEMBase (length, size, etc.)

# Mesh readers (consolidated from AbaqusReader.jl and AsterReader.jl)
include("readers.jl")

# Graph algorithms (RCM bandwidth minimization from GraphOrdering.jl)
include("graph/graph_ordering.jl")

# TODO: Consolidate these vendor packages later
# using AbaqusReader  # Consolidated into src/readers.jl
# using AsterReader   # Consolidated into src/readers.jl

# Problem types (OLD - all reference old basis types like Seg2, Tri3, Poi1, etc.)
# include("problems_heat.jl")
# export Heat, PlaneHeat
# include("problems_truss.jl")
# export Truss
# include("problems_elasticity.jl")
# export Elasticity
# include("materials_plasticity.jl")  # Requires ForwardDiff for automatic differentiation
# export plastic_von_mises
include("problems_dirichlet.jl")
export Dirichlet

# export assemble!, postprocess!

# TODO: Consolidate vendor packages (FEMBeam, Mortar) later
# Structural elements: beams
# @reexport using FEMBeam

### Mortar methods ###

# @reexport using MortarContact2D
# @reexport using MortarContact2DAD

# include("problems_mortar.jl")
# include("problems_mortar_3d.jl")
# export calculate_normals, calculate_normals!, project_from_slave_to_master,
#    project_from_master_to_slave, Mortar, get_slave_elements,
#    get_polygon_clip, calculate_polygon_area
# include("io.jl")  # Requires HDF5 and LightXML - skip for minimal deps
# export Xdmf, h5file, xmffile, xdmf_filter, new_dataitem, update_xdmf!, save!
include("solvers.jl")
export AbstractSolver, Solver, Nonlinear, NonlinearSolver, Linear, LinearSolver,
   get_unknown_field_name, get_formulation_type, get_problems,
   get_field_problems, get_boundary_problems,
   get_field_assembly, get_boundary_assembly,
   initialize!, create_projection, eliminate_interior_dofs,
   is_field_problem, is_boundary_problem
# include("solvers_modal.jl")  # Requires Arpack for eigenvalue problems
# export Modal

# Re-export Analysis and related types from FEMBase (needed by tests)
export Analysis, AbstractAnalysis, add_problems!, run!
include("problems_contact.jl")
include("problems_contact_3d.jl")
#include("problems_contact_3d_autodiff.jl")
export Contact

module Preprocess
end

using SparseArrays, LinearAlgebra
include("preprocess.jl")
export create_elements, Mesh, add_node!, add_nodes!,
   add_element_to_element_set!, add_node_to_node_set!,
   find_nearest_nodes, find_nearest_node, reorder_element_connectivity!,
   create_node_set_from_element_set!, filter_by_element_set

# IO submodule for mesh readers and result writers
include("io/io.jl")
using .IO
export abaqus_read_mesh, create_surface_elements, create_nodal_elements
export aster_read_mesh  # Requires HDF5 - add when optional deps are set up

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
export update!, add_element!, add_elements!, get_unknown_field_name, add!,
   is_field_problem, is_boundary_problem, get_gdofs,
   initialize!, get_integration_points, group_by_element_type,
   get_unknown_field_dimension, get_connectivity
export get_nonzero_rows, get_local_coordinates, inside, IP, get_element_type,
   get_elements, AbstractProblem, IntegrationPoint, filter_by_element_type,
   get_element_id, get_nonzero_columns, resize_sparse, resize_sparsevec

end
