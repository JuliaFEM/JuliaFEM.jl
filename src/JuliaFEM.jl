# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    JuliaFEM.jl - Modern Finite Element Method Library for Julia

JuliaFEM is an open-source FEM library focused on **contact mechanics**, with a modern
architecture designed for GPU acceleration and educational transparency.

**Project Status:** Revived November 2025 (original: 2015-2019) with complete architectural overhaul.

# Key Features

- **Contact Mechanics Focus**: Primary differentiation from general-purpose FEM libraries
- **Nodal Assembly**: GPU-friendly architecture without atomic operations
- **Tensors.jl Integration**: Natural tensor notation for stress, strain, and stiffness
- **Type-Stable Fields**: NamedTuple-based fields for 100× performance vs Dict
- **Matrix-Free Solvers**: Krylov methods (GMRES) for large-scale problems
- **Backend Abstraction**: Identical user code runs on CPU or GPU

# Modern API (November 2025)

## Physics-Based Interface

```julia
using JuliaFEM

# Create physics problem
physics = Physics(Elasticity, "cantilever beam", 3)

# Add elements with type-stable fields
add_elements!(physics, body_elements)

# Apply boundary conditions
add_dirichlet!(physics, fixed_nodes, [1,2,3], 0.0)  # Fix all DOFs
add_neumann!(physics, surface_elements, traction)    # Surface load

# Solve (automatically selects backend)
solution = solve!(physics, backend=GPU())  # or backend=CPU()
```

## Element Creation (Immutable, Type-Stable)

```julia
# Modern approach: NamedTuple fields (type-stable!)
element = Element(Lagrange{Triangle,1}, (1,2,3),
                 fields=(E=210e3, ν=0.3, ρ=7850.0))

# Update immutably (returns new element)
element2 = update(element, :displacement => u_values)

# Access fields (compile-time type known!)
E = element.fields.E  # Float64, no Dict lookup!
```

# Architecture Highlights

**Nodal Assembly** (GPU-friendly):
- Loop over nodes (not elements) → no atomic operations on GPU
- 3×3 stiffness blocks using Tensor{2,3} from Tensors.jl
- Matrix-free K*v without forming global matrix

**Separation of Concerns**:
- Topology: Reference element geometry (Triangle, Quadrilateral, Tetrahedron, etc.)
- Integration: Gauss quadrature rules (zero-allocation tuple-based)
- Basis: Shape functions (Lagrange, Serendipity)
- Materials: Stress computation (LinearElastic, NeoHookean, PerfectPlasticity)

**Contact Mechanics**:
- Mortar methods for interface coupling
- 2D and 3D contact with friction
- GPU-accelerated contact detection

# Documentation

Comprehensive guides in `docs/book/`:

- `element_architecture.md` - Element composition philosophy
- `nodal_assembly_concept.md` - GPU-friendly assembly
- `multigpu_nodal_assembly.md` - Multi-GPU architecture (in progress)

# Legacy API Support

The old `Problem` API is maintained for backward compatibility:

```julia
problem = Problem(Elasticity, "body", 3)
add_elements!(problem, elements)  # Still works
```

**Migration:** Gradually transitioning to `Physics` API for new code.

# More Information

Website: www.juliafem.org
GitHub: github.com/JuliaFEM/JuliaFEM.jl

"""
module JuliaFEM

# Import Base functions FIRST before defining any methods
# Only import what we actually use - removed unused: similar, first, last, vec, 
# +, -, *, /, isempty, empty!, push!
import Base: getindex, setindex!, convert, length, size, isapprox,
   ==, haskey, copy, read, append!

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

# Consolidated topology files (one per shape family)
include("topology/segments.jl")       # Segment (1D)
include("topology/triangles.jl")      # Triangle (2D simplex)
include("topology/quadrilaterals.jl") # Quadrilateral (2D quad)
include("topology/tetrahedra.jl")     # Tetrahedron (3D simplex)
include("topology/hexahedra.jl")      # Hexahedron (3D hex)
include("topology/pyramids.jl")       # Pyramid (3D)
include("topology/wedges.jl")         # Wedge (3D prism)

# Export topology types (shape names, NOT node counts!)
export AbstractTopology, dim, reference_coordinates, edges, faces
export Segment        # 1D line
export Triangle       # 2D simplex
export Quadrilateral  # 2D quad
export Tetrahedron    # 3D simplex
export Hexahedron     # 3D hex
export Pyramid        # 3D pyramid
export Wedge          # 3D prism

# Export deprecated aliases (backward compatibility)
# These resolve to topology types, NOT separate types!
export Seg2, Seg3                    # → Segment
export Tri3, Tri6, Tri7             # → Triangle
export Quad4, Quad8, Quad9          # → Quadrilateral
export Tet4, Tet10                  # → Tetrahedron
export Hex8, Hex20, Hex27           # → Hexahedron
export Pyr5                         # → Pyramid
export Wedge6, Wedge15              # → Wedge

# Note: Node count is NO LONGER in topology name! It comes from basis:
# Examples:
#   Triangle + Lagrange{Triangle, 1} → 3 nodes (Tri3 → Triangle)
#   Triangle + Lagrange{Triangle, 2} → 6 nodes (Tri6 → Triangle)
#   Quadrilateral + Lagrange{Quadrilateral, 1} → 4 nodes (Quad4 → Quadrilateral)
#   Quadrilateral + Serendipity{Quadrilateral, 2} → 8 nodes (Quad8 → Quadrilateral)
#   Quadrilateral + Lagrange{Quadrilateral, 2} → 9 nodes (Quad9 → Quadrilateral)

# ============================================================================
# QUADRATURE: Low-level integration point data (consolidated from FEMQuad.jl)
# ============================================================================
include("quadrature.jl")

# ============================================================================
# INTEGRATION: High-level integration schemes (NEW - separation of concerns)
# ============================================================================
include("integration/integration.jl")  # Abstract integration interface, IntegrationPoint
include("integration/gauss.jl")        # Gauss-Legendre quadrature
include("integration/gauss_points.jl") # NEW: Compile-time integration points (zero-allocation)

export AbstractIntegration, IntegrationPoint, integration_points, npoints
export Gauss
export get_gauss_points!  # NEW: Zero-allocation integration point API

# ============================================================================
# GEOMETRY: Jacobian computation and coordinate transformations
# ============================================================================
include("geometry/jacobian.jl")
export compute_jacobian, physical_derivatives

include("geometry/strain.jl")
export compute_strain

# ============================================================================
# BASIS: Interpolation schemes (consolidated from FEMBasis.jl)
# ============================================================================
include("basis/abstract.jl")
include("basis/subs.jl")  # Symbolic substitution (includes minimal simplify from SymDiff.jl)
include("basis/vandermonde.jl")

# NEW BASIS API (November 2025 - see docs/book/adr-003-basis-function-api.md)
include("basis/basis_api.jl")           # New API: get_basis_functions, get_basis_derivatives

# Lagrange basis functions - auto-generated file contains all basis types
# Now generates methods for parametric Lagrange{T,P} type with BOTH old and new APIs
include("basis/lagrange_generator.jl")
include("basis/lagrange_generated.jl")  # Auto-generated by: julia --project=. src/basis/lagrange_generator.jl

# Export basis types and functions (OLD API)
export AbstractBasis, Lagrange, Serendipity
export nnodes, get_reference_element_coordinates, eval_basis!, eval_dbasis!

# Export new API functions (NEW API - November 2025)
export get_basis_functions, get_basis_derivatives
export get_basis_function, get_basis_derivative

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

# GPU Physics (new architecture - pure GPU, all BCs in device code)
# Note: CUDA is loaded by the demo script, not here
# The gpu_physics_elasticity.jl module should be included directly by demos

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

export assemble!, postprocess!

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
# Backend-transparent physics API
include("physics_api.jl")
export Physics, Elasticity, DirichletBC, NeumannBC
export add_elements!, add_dirichlet!, add_neumann!

# Assembly structures (element and nodal)
include("element_assembly_structures.jl")
export ElementAssemblyData, ElementContribution
export scatter_to_global!, compute_residual!, apply_dirichlet_bc!
export matrix_vector_product, get_dof_indices

include("nodal_assembly_structures.jl")
export NodeToElementsMap, get_node_spider

# Backend abstraction (CPU/GPU selection)
include("backend/abstract.jl")
export solve!, Auto, GPU, CPU
export ElasticitySolution

# CPU backend (fallback for now)
include("backend/cpu.jl")

# GPU backend is now loaded via extension (ext/JuliaFEMCUDAExt.jl)
# Extension automatically loads when user does 'using CUDA'
# No need to manually include anymore!

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
export update!, update, add_element!, add_elements!, get_unknown_field_name, add!,
   is_field_problem, is_boundary_problem, get_gdofs,
   initialize!, get_integration_points, group_by_element_type,
   get_unknown_field_dimension, get_connectivity
export get_nonzero_rows, get_local_coordinates, inside, IP, get_element_type,
   get_elements, AbstractProblem, IntegrationPoint, filter_by_element_type,
   get_element_id, get_nonzero_columns, resize_sparse, resize_sparsevec

end
