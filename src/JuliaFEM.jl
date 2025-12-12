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

# Problem API

The `Problem` type provides element-based assembly:

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

# Import Base functions for extension
import Base: getindex, setindex!, convert, length, size, isapprox,
   ==, haskey, copy, read, append!

using SparseArrays, LinearAlgebra
using Logging  # For mesh readers
using Tensors  # For basis functions (Vec type)
using StaticArrays: SVector  # Static coordinate storage for topology nodes

# No-op timing macro
macro timeit(args...)
   return esc(args[end])
end

# ============================================================================
# CORE API - Include FIRST (all abstract types and interfaces)
# ============================================================================

# This is the Julia equivalent of C/C++ header files.
# All abstract types and lightweight structs are defined here.
# This MUST be included before any concrete implementations.
include("api.jl")  # Documentation-only (no type definitions)

# Continuum domain abstract types
include("domains/continuum/abstract.jl")

# Continuum domain concrete types
include("domains/continuum/types.jl")

# Formulation domain API (discretization strategies)
include("domains/continuum/formulations.jl")
export AbstractFormulation, ContinuumFormulation
export AbstractContinuumTheory, FullThreeD, PlaneStress, PlaneStrain, Axisymmetric

# Field domain API (field variable types and DOF counting)
include("fields/api.jl")
export AbstractField, Displacement, Temperature, DisplacementRotation
export dofs_per_node, get_dof_mapping!, get_field

# Local field evaluation (field quantities at a point)
include("fields/local_field.jl")
export LocalField

# Material domain API
include("materials/api.jl")
export AbstractMaterial, AbstractElasticMaterial, AbstractPlasticMaterial
export AbstractMaterialState, EmptyState
export compute_stress, elasticity_tensor
# Material behavior traits (for generic integration)
export MaterialBehavior, StatelessConstantTangent, StatelessStrainDependent, StatefulStrainDependent
export material_behavior, needs_deformation, needs_state

# Mesh domain API  
include("mesh/api.jl")
export AbstractMesh, AbstractRefineStrategy
export nnodes_total, nelements, get_node, connectivity_matrix
export get_elements_for_node, get_element_set, get_node_set
export refine

# Subdomain API files (define their own abstractions and formulations)
include("domains/beams/api.jl")    # AbstractBeamTheory, BeamFormulation{Theory}, EulerBernoulli, Timoshenko
export AbstractBeamTheory, BeamFormulation, EulerBernoulli, Timoshenko

include("domains/shells/api.jl")   # AbstractShellTheory, ShellFormulation{Theory}, ReissnerMindlin, KirchhoffLove
export AbstractShellTheory, ShellFormulation, ReissnerMindlin, KirchhoffLove

include("domains/trusses/api.jl")  # AbstractTrussTheory, TrussFormulation{Theory}, SimpleTruss
export AbstractTrussTheory, TrussFormulation, SimpleTruss

# Plate formulations
include("domains/plates/api.jl")
export AbstractPlateFormulation, DKTFormulation, PlateDisplacement
export constitutive_matrix_plate, get_thickness

# Note: domains/plates/dkt.jl is included later after Mesh is defined (line ~430)

# Physics domain API (abstract type, interface functions, and concrete implementations)
include("physics/abstract.jl")       # AbstractPhysics abstract type
include("physics/api.jl")            # Interface functions (assemble!, solve!, etc.)
include("physics/types.jl")          # Physics category types (Elasticity, Thermal, etc.)
# include("physics/boundary_conditions.jl")  # BC method implementations (for old Physics struct - not needed)
export AbstractPhysics, assemble!, solve!, add_dirichlet!, add_neumann!
export Elasticity, Thermal, required_field_type

# Physics utilities (strain extraction from displacement gradients)
include("physics/strain.jl")
export extract_strain, extract_strain_rate

export AbstractTopology

# import FEMSparse  # Consolidated into src/sparse/
# import FEMQuad   # Consolidated into src/quadrature.jl

# Note: Consolidating FEMBase and FEMBasis into JuliaFEM
# Previously: @reexport using FEMBase
# Now: Include files directly below

# ============================================================================
# TOPOLOGY: Reference element geometries (NEW - separation of concerns)
# ============================================================================
# Topology domain API (abstract type and interface)
include("topology/api.jl")
export AbstractTopology, nnodes, dim, reference_coordinates, edges, faces

# Export topological entity types
export TopologicalEntity, Vertex, Edge, Face, Cell
export entities, nentities, vertices, cells
export nvertices, nedges, nfaces

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

# Export aliases. These resolve to topology types, NOT separate types!
export Seg2, Seg3                   # → Segment
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
# QUADRATURE: Integration schemes and quadrature rules
# ============================================================================
include("quadrature/api.jl")          # Core API types and interfaces (NEW)
# integration.jl removed - OLD AbstractIntegration API no longer needed

# Gauss-Legendre quadrature point tables (must be loaded before gauss.jl)
include("quadrature/quaddata.jl")     # Quadrature data constants
include("quadrature/gl_tetrahedra.jl")    # Tetrahedron quadrature points
include("quadrature/gl_triangles.jl")     # Triangle quadrature points
include("quadrature/gl_wedges.jl")        # Wedge quadrature points
include("quadrature/gl_pyramids.jl")      # Pyramid quadrature points
include("quadrature/gl_tensor_product.jl") # Segment, Quadrilateral, Hexahedron quadrature points

include("quadrature/gauss.jl")        # Gauss-Legendre quadrature
# gauss_points.jl removed - OLD compile-time integration point API no longer needed

export IntegrationPoint, integration_points, npoints
export get_quadrature_points  # Export quadrature point tables
# New quadrature API
export AbstractQuadratureRule, GaussLegendre, GaussLobatto, QuadraturePoint
export default_quadrature

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
include("basis/api.jl")
include("basis/subs.jl")  # Symbolic substitution (includes minimal simplify from SymDiff.jl)
include("basis/vandermonde.jl")

# Lagrange basis functions - auto-generated file contains all basis types
# Now generates methods for parametric Lagrange{T,P} type with BOTH old and new APIs
include("basis/basis_generator.jl")
include("basis/basis_generated.jl")  # Auto-generated by: julia --project=. src/basis/basis_generator.jl

# Plate element basis functions (DKT, DST, etc.)
include("domains/plates/dkt_basis.jl")  # AbstractPlateBasis, DKT

# Export basis types and functions
export AbstractBasis, Lagrange, Serendipity
export eval_basis!, eval_dbasis!
export ndofs
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

# ============================================================================
# DOF SYSTEM: Unified field specification system
# ============================================================================
include("dofs/dofs.jl")

# Modern unified field system (multi-field is fundamental!)
export AbstractDOF, DOFSet, @DOFSet
export element_id, n_element_dofs, element_dofs, basis_type, dof_type  # Element accessors
export local_dof_count, global_dof_indices, local_to_global_map, field_dof_range  # Coupled assembly
export field_ndofs, field_names, field_count, is_single_field
export quantity_type, entity_type, single_field
export ndofs, dof_size  # dof_size needed by fields.jl
export extract_element_dofs, extract_element_dofs_structured  # DOF extraction from global vector
export interpolate_fields, interpolate_field, interpolate_field_value  # Field interpolation at quadrature points
export interpolate_local_fields  # LocalField interpolation with rates

# DOF manager and element creation
export DOFManager, get_element_ids, create_elements!
export get_node_dofs, allocate_dofs!
export register_fields!, count_field_dofs, count_entities  # NEW: Multi-field registration
export ElementDescription

# DOF connectivity (inverse mapping for DOF-based assembly)
export DOFElementConnection, DOFConnectivity, DOFConnectivityGPU
export build_dof_connectivity, build_dof_connectivity_gpu
export connection_count, is_empty
export elem_id, local_dof_idx

# Legacy DOF type (DOF type for field variables)
export DOF  # Old entity-based DOF{Quantity, Entity}

# Consolidate FEMBase.jl into src/ (Phase 1 continued)
# Order matters: fields → types → sparse → elements → integrate → problems → assembly
include("fields/fields.jl")          # LEGACY field system (DCTI, DVTI - to be removed)
include("legacy/core_types.jl")      # Node, IP, IntegrationPoint (LEGACY - Dict-based)

# Compatibility shim: Create FEMBase module for vendor packages EARLY
# NOTE: The compatibility shim file was removed to avoid a missing-file
# include during package load. If a full FEMBase compatibility layer is
# required later, add a proper shim file and include it here.

include("sparse/sparse.jl")          # SparseMatrixCOO, SparseVectorCOO
include("elements/elements.jl")      # Element type and interface
include("elements/extract_element_dofs.jl")  # DOF extraction from global vector
include("elements/interpolate.jl")           # Field interpolation at quadrature points
include("elements/elements_lagrange.jl")  # OLD - uses AbstractBasis{0} (Poi1)

include("legacy/assembly_problems.jl")      # Problem types
include("solvers/solvers_base.jl")   # Base solver types
include("legacy/analysis.jl")               # Analysis and AbstractResultsWriter
include("legacy/deprecated_fembase.jl")     # Deprecated/legacy methods from FEMBase

# GPU Physics (new architecture - pure GPU, all BCs in device code)
# Note: CUDA is loaded by the demo script, not here
# The gpu_physics_elasticity.jl module should be included directly by demos

# Mesh readers (consolidated readers/ → io/)
# AbaqusReader - ABAQUS .inp file format
include("io/keyword_register.jl")
include("io/parse_mesh.jl")
include("io/parse_model.jl")
include("io/create_surface_elements.jl")
include("io/abaqus_download.jl")

# AsterReader - Code Aster .med file format (requires HDF5)
# include("io/read_aster_mesh.jl")
# include("io/read_aster_results.jl")

# Graph algorithms (RCM bandwidth minimization from GraphOrdering.jl)
include("mesh/graph_ordering.jl")

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
include("legacy/problems_dirichlet.jl") 
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

# Note: Physics API now defined in api.jl (included at top of file)
# physics_api.jl is deprecated and will be removed

# Material models
include("materials/abstract_material.jl")
# Note: abstract_material.jl redefines AbstractMaterial (already in api.jl)
# TODO: Remove duplicate from abstract_material.jl

# Material state variables and traits (new compositional design)
include("materials/state_variables.jl")
export AbstractStateVariable
export PlasticStrain, Backstress, EquivalentPlasticStrain, DamageVariable
export state_variable_type, default_symbol

include("materials/traits.jl")
include("materials/field_traits.jl")
export supported_physics, required_field_types, required_state_variables
export required_material_fields, material_field_type, create_zero_field
export get_state_variable_types, get_state_variable_symbols, is_stateful

# Material cache for state variable storage (global, compositional)
include("materials/material_cache.jl")
export GlobalMaterialCache
export create_global_material_cache
export get_state, get_old_state, set_state!
export update_cache!, reset_cache!
export get_state_variable, set_state_variable

include("materials/linear_elastic.jl")
export LinearElastic

include("materials/neo_hookean.jl")
export NeoHookean

include("materials/perfect_plasticity.jl")
export PerfectPlasticity

# NOTE: state_bridge.jl removed - materials now use compositional NamedTuple directly
# No more conversion between monolithic structs and compositional states needed!

# ============================================================================
# LEVEL 4: ASSEMBLY FRAMEWORK (Generic assembly strategies)
# ============================================================================

# 
# Old files: element_structures.jl, nodal_structures.jl, framework.jl
# These were NOT used anywhere in the codebase

# Generic assemblers (domain-agnostic)
include("assemblers/abstract.jl")
export AbstractAssembler, AbstractAssemblerCache, AbstractKernel
export ElementBasedAssembler
export COOAssembler, CSCAssembler

include("assemblers/element_cache.jl")
export ElementCache
export create_element_cache

include("assemblers/geometry_cache.jl")
export GeometryCache, ImmutableGeometryCache

include("assemblers/material_cache.jl")
export AssemblyMaterialWorkspace, ImmutableMaterialStateCache
export create_material_cache, create_assembly_workspace
export get_stress, get_tangent, get_field, set_fields!
export get_tangent_vector, get_stress_vector  # Zero-allocation field extraction

include("assemblers/caches.jl")
export COOCache, CSCCache
export reset!, extract_system, build_sparsity_pattern

include("assemblers/kernel_interface.jl")
export compute_element_stiffness!, dofs_per_node, get_dof_mapping!
export compute_b_matrix, compute_jacobian, validate_kernel

include("assemblers/element_based_coo.jl")
export assemble!, create_cache

include("assemblers/element_based_csc.jl")
# assemble!, create_cache already exported

# Microkernel architecture for multi-field DOF-by-DOF assembly
# (Included after assemblers since it depends on AbstractKernel, GeometryCache, AssemblyMaterialWorkspace)
include("physics/microkernels.jl")   # evaluate() interface, traits
include("physics/formulations.jl")   # Helper utilities for Element{K,P,S}
export evaluate, requires_basis_values, requires_basis_gradients, requires_basis_second_derivatives
export field_type_for_dispatch, ThermoelasticityFields

# ============================================================================
# LEVEL 5: PHYSICAL DOMAINS (Domain-specific assembly kernels)
# ============================================================================

# ============================================================================
# COMMON DOMAIN UTILITIES (Shared across all domain types)
# ============================================================================

# Boundary conditions (generic - work with any kernel/domain)
# include("domains/common/boundary_conditions.jl")  # Uses old Physics struct - not needed for now
# export apply_neumann_bcs!, apply_dirichlet_bcs!  # Explicit BC application

# ============================================================================
# CONTINUUM MECHANICS DOMAIN
# ============================================================================

# Continuum mechanics kernel (weak form only)
include("domains/continuum/kernel.jl")
export ContinuumKernel
export compute_block_at_point  # Weak form (atomic operation)

# Continuum cache update functions (three-phase API)
include("domains/continuum/update_geometry_cache.jl")
include("domains/continuum/update_element_cache.jl")
include("domains/continuum/update_material_cache.jl")
export compute_block, compute_block!  # Integration API (compute_block! deprecated)
export update_geometry_cache!, update_element_cache!, update_material_cache!  # Cache update functions

# Continuum assembly (uses generic assemblers)
export compute_element_stiffness  # For testing and advanced use

# Backend abstraction (CPU/GPU selection)
# TODO: These need to be updated to work with new Physics API
# Temporarily commented out until backend dispatch is updated
# include("backend/abstract.jl")
# export solve!, Auto, GPU, CPU
# export ElasticitySolution

# CPU backend (fallback for now)
# include("backend/cpu.jl")

# GPU backend is now loaded via extension (ext/JuliaFEMCUDAExt.jl)
# Extension automatically loads when user does 'using CUDA'
# No need to manually include anymore!

include("legacy/solvers.jl") 
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
# Commented out during Element{K,P,S} redesign (uses old Element API)
# include("legacy/problems_contact.jl") 
# include("legacy/problems_contact_3d.jl") 
#include("legacy/problems_contact_3d_autodiff.jl") 
# export Contact

module Preprocess
end

using SparseArrays, LinearAlgebra

# NEW: Include modern parametric Mesh{T<:AbstractTopology} infrastructure
include("mesh/mesh.jl")
export Mesh, topology_type, nnodes_per_element, nelements, nnodes_total

# DOF manager requires Mesh type - include after Mesh is defined
include("dofs/dof_manager.jl")

# DOF connectivity (inverse mapping: DOF → Elements) - needs Element and DOFManager
include("dofs/dof_connectivity.jl")

# DOF-based assembler (needs DOF connectivity)
include("assemblers/dof_based_coo.jl")
export DOFBasedCOOAssembler, DOFBasedCOOCache

# Now that Mesh is defined, include plate elements that depend on it
# include("domains/plates/dkt.jl")  # Uses old Physics struct - experimental, not in main test suite
export get_elements_for_node, connectivity_matrix, get_node
export find_nearest_nodes, find_nearest_node
export get_element_set, get_elements_in_set
export get_node_set, get_nodes_in_set, create_node_set_from_element_set!
export extract_surface, validate, info
export set_node_id!, get_node_by_id, set_element_id!, get_element_by_id
export set_node_color!, get_node_color, set_element_color!, get_element_color, get_elements_with_color
export mark_ghost_node!, is_ghost_node, mark_ghost_element!, is_ghost_element
export get_local_nodes, get_local_elements
export apply_node_permutation!, apply_element_permutation!

# Mesh refinement strategies
include("mesh/refine.jl")
export AbstractRefineStrategy, LongestEdgeBisection, refine

# Structured mesh generation utilities
include("mesh/structured.jl")
export create_structured_box_mesh, create_unit_cube_mesh
export create_cantilever_mesh, create_thin_plate_mesh

# Polar/circular mesh utilities
include("mesh/circular.jl")
export create_circular_plate_mesh

# OLD: Comment out Dict-based Mesh (conflicts with new Mesh{T})
# include("preprocess.jl")
# export create_elements, Mesh, add_node!, add_nodes!,
#    add_element_to_element_set!, add_node_to_node_set!,
#    find_nearest_nodes, find_nearest_node, reorder_element_connectivity!,
#    create_node_set_from_element_set!, filter_by_element_set

# IO submodule for mesh readers and result writers
include("io/io.jl")
include("io/gmsh_reader.jl")  # GMSH reader (moved from root)
using .IO
export abaqus_read_mesh, create_surface_elements, create_nodal_elements
export aster_read_mesh  # Requires HDF5 - add when optional deps are set up

# Postprocess module

module Postprocess
end

# Postprocess utilities (moved to trash - needs refactoring)
# include("postprocess_utils.jl")
# export calc_nodal_values!, get_nodal_vector, get_nodal_dict, copy_field!,
#    calculate_area, calculate_center_of_mass, calculate_second_moment_of_mass,
#    extract

include("legacy/deprecations.jl") 

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
