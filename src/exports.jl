# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

# ============================================================================
# Public-API exports for JuliaFEM. One file, grouped by domain, no duplicates.
#
# Pure name-binding reservations: this file does not depend on any symbol
# being defined yet. All exported names are defined by `include`s in
# `src/JuliaFEM.jl`. Verified against the live module by
# `julia --project=. -e 'using JuliaFEM; for s in names(JuliaFEM) ... end'`.
# ============================================================================

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
export AbstractTopology
export TopologicalEntity, Vertex, Edge, Face, Cell
export entities, nentities, vertices, cells
export nvertices, nedges, nfaces
export nnodes, dim, reference_coordinates, edges, faces

# Topology shape types (one per family) and their node-count aliases.
export Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Wedge
export Seg2, Seg3
export Tri3, Tri6, Tri7, Tri10
export Quad4, Quad8, Quad9
export Tet4, Tet10
export Hex8, Hex20, Hex27
export Pyr5
export Wedge6, Wedge15

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------
export AbstractQuadratureRule, GaussLegendre, GaussLobatto, QuadraturePoint
export integration_points, get_quadrature_points, default_quadrature, basis_quadrature_order

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
export compute_jacobian, physical_derivatives, compute_strain
export piola_covariant, piola_contravariant

# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------
export Vec  # Re-export from Tensors.jl, used by basis function APIs.
export AbstractBasis, Lagrange, Serendipity
export get_basis_functions, get_basis_derivatives
export get_basis_function, get_basis_derivative
export nbasis
export nedelec_whitney_tet_reference,
    nedelec_hierarchical_tet_edge_reference, rt0_tet_reference_basis, rt0_hex8_reference_basis

# ---------------------------------------------------------------------------
# Formulations and field tags
# ---------------------------------------------------------------------------
export AbstractFormulation, ContinuumFormulation
export AbstractContinuumTheory, ThreeDimensional, PlaneStress, PlaneStrain, Axisymmetric

export AbstractField, Displacement, Temperature, PorePressure, MoistureContent, PressurePotential, RT0FaceFlux, Nedelec1Edge, DisplacementRotation
export LocalField

# ---------------------------------------------------------------------------
# DOF system (DOF{Quantity, Entity}, DOFSet, layout, extraction, interpolation)
# ---------------------------------------------------------------------------
export AbstractDOF, DOF, DOFSet, @DOFSet
export ndofs, dof_size, quantity_type, entity_type
export field_ndofs, field_names, field_count, is_single_field
export field_dof_range, field_idx, entity_local, component
export DOFLayoutEntry, local_dof_layout

export element_id, n_element_dofs, element_dofs, basis_type, dof_type
export local_dof_count, global_dof_indices, local_to_global_map
export extract_element_dofs, extract_element_dofs_structured
export interpolate_fields, interpolate_field, interpolate_field_value
export interpolate_local_fields

export DOFHandler,
    DOFManager,
    create_elements!,
    global_field_ranges,
    global_facet_dof,
    saddle_point_blocks,
    saddle_point_matrix_blocks
export get_node_dofs, get_element_ids
export default_pressure_gauge_dof
export DOFElementConnection, DOFConnectivity
export build_dof_connectivity
export connection_count, is_empty
export elem_id, local_dof_idx

export dofs_per_node, get_dof_mapping!

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
export AbstractMesh, Mesh, topology_type

# Interface mesh (coupled boundaries)
export AbstractInterfaceMesh,
    InterfaceMesh,
    InterfaceVolumeCoupling,
    InterfaceDOFHandler,
    create_interface_elements!,
    interface_nnodes,
    interface_nelements
export nnodes_total, nelements, nnodes_per_element
export get_node
export connectivity_matrix, get_elements_for_node
export get_element_set, get_node_set
export get_elements_in_set, get_nodes_in_set
export extract_surface, surface_topology, validate, info
export AbstractRefineStrategy, LongestEdgeBisection, refine
export create_structured_box_mesh, create_unit_cube_mesh, create_cook_membrane_mesh
export create_structured_line_mesh
export read_gmsh_msh, mesh_from_current_gmsh_model, write_vtu_mesh
export create_cantilever_mesh, create_thin_plate_mesh
export FEDiscretization, linear_system, total_dofs
export AbstractFacetConnectivityMaps
export Hex8FacetMaps, build_hex8_facet_maps, build_hex20_facet_maps, hex8_face_area_physical,
       hex8_face_outward_sign, hex8_edge_length_physical, hex8_edge_orientation_sign,
       hex20_edge_orientation_sign
export Tet4FacetMaps, build_tet4_facet_maps, build_tet10_facet_maps,
       tet_facet_gid_from_corners, tet_face_area_physical,
       tet_edge_length_physical, tet_face_outward_sign, tet4_edge_orientation_sign,
       tet10_edge_orientation_sign
export Wedge6FacetMaps, build_wedge6_facet_maps, wedge_face_area_physical, wedge_edge_length_physical,
       wedge_face_outward_sign, wedge6_edge_orientation_sign
export Pyr5FacetMaps, build_pyr5_facet_maps, pyr_face_area_physical, pyr_edge_length_physical,
       pyr_face_outward_sign, pyr5_edge_orientation_sign

# ---------------------------------------------------------------------------
# Materials (abstract types, traits, state-variable system, concrete models)
# ---------------------------------------------------------------------------
export AbstractMaterial, AbstractElasticMaterial, AbstractPlasticMaterial
export compute_stress, elasticity_tensor

export MaterialBehavior
export StatelessConstantTangent, StatelessStrainDependent, StatefulStrainDependent
export material_behavior, needs_deformation, needs_state

export AbstractStateVariable
export PlasticStrain, Backstress, EquivalentPlasticStrain, DamageVariable
export DamageEquivalentStrain, Eigenstrain, CreepStrain, ChabocheAlpha
export state_variable_type, default_symbol

export supported_physics, required_field_types, required_state_variables
export required_material_fields, material_field_type, create_zero_field
export get_state_variable_types, get_state_variable_symbols, is_stateful

export GlobalMaterialCache, create_global_material_cache
export get_state, get_old_state, set_state!
export update_cache!, reset_cache!
export get_state_variable, set_state_variable

export LinearElastic, OrthotropicLinearElastic, NeoHookean, PerfectPlasticity, J2LinearIsotropicPlasticity
export AbstractContinuumKinematics, SmallStrainKinematics, GreenLagrangeKinematics, continuum_kinematics
export MooneyRivlin, Yeoh3, Gent
export StVenantKirchhoffJ2Plasticity, ScalarDamageLinearElastic, ChabocheJ2Plasticity
export NortonCreepElastic, LinearElasticWithEigenstrain
export HeatConductivity, compute_heat_flux, conductivity_tensor, scalar_diffusion_tensor
export HydraulicConductivity, hydraulic_conductivity_tensor, compute_seepage_flux
export MoistureDiffusivity
export ElementWiseScalarDiffusion

# ---------------------------------------------------------------------------
# Physics tags (used by material traits; not the legacy `Physics{...}` struct)
# ---------------------------------------------------------------------------
export AbstractPhysics
export Elasticity, Thermal, required_field_type
export extract_strain, extract_strain_rate

# Reserved names. The default 0.x build does not implement methods on these; they
# exist so that `JuliaFEM.Legacy` can hang its methods on the same generic
# functions, and so that the assemblers can extend `assemble!`.
export assemble!
export solve!, add_dirichlet!, add_neumann!

# ---------------------------------------------------------------------------
# Assemblers (caches, kernel contract, element-based, DOF-based, matrix-free)
# ---------------------------------------------------------------------------
export AbstractAssembler, AbstractAssemblerCache, AbstractKernel
export ElementBasedAssembler, COOAssembler

export ContinuumElementCache, ElementCache, create_element_cache
export GeometryCache
export AssemblyMaterialWorkspace
export create_material_cache
export get_field, set_fields!
export get_stress, get_tangent
export get_stress_vector, get_tangent_vector

export ContinuumCOOCache, COOCache
export reset!, extract_system

export create_cache

# Microkernel contract (DOF-based / matrix-free path)
export qpoint_buffer_eltype, update_qpoint_buffer!, prepare_dof_based_material_workspace!
export evaluate_entry, evaluate_mass_entry, compute_internal_force_value
export reference_fields

# DOF-based assembler (CPU + KernelAbstractions)
export DOFBasedCOOAssembler, DOFBasedCOOCache
export UniformKernelColumn, PerElementKernelColumn
export prototype_kernel, kernel_at, assert_homogeneous_dof_based_kernel_column!
export ka_per_element_kernel_column_supported
export DOFBasedCOOCacheKA, sync_from_cpu!, to_float32
export apply_K!, apply_K_masked_rows!, apply_K_owned_rows!, apply_K_owned_rows_from_packed!, apply_K_contributions!, apply_M!, assemble_M!, equilibrium_residual!
export apply_f_int_owned_rows!, apply_f_int_owned_rows_from_packed!
export assemble_internal_force!, nonlinear_equilibrium_residual!

export AbstractMultiplyGhostLayout, LocalMultiplyLayout, ReferenceMaskMultiplyLayout
export prepare_multiply_workspace!
export MeshPartitionLayout, uniform_single_partition
export element_indices_for_part, validate_partition
export brick_hex_slab_upper, brick_hex_partition_slabs
export element_counts_by_part, referenced_global_dofs
export sum_element_dof_slots
export mark_referenced_dofs!, collect_true_indices!, fill_referenced_dof_indices!
export ghost_dof_mask!, node_partition_owner_min!, mark_owned_vertex_field_dofs!
export PartitionAdjacency, build_partition_adjacency
export RankHaloExchange, build_rank_halo_exchanges, build_matvec_halo_exchanges, referenced_dof_mask_for_part!
export matvec_halo_mpi_request_count
export PartitionPackedLayout, build_partition_packed_layout, build_partition_packed_layout_for_matvec
export mark_matvec_stencil_closure!
export gather_from_global_to_packed!, expand_packed_to_global!, gather_owned_from_global_to_packed!
export copy_owned_subset_to_packed_owned_prefix!, extract_owned_subset_from_global!, gather_owned_rows_from_workspace!
export gather_ghosts_from_global_to_packed!
export unpack_halo_recv_to_packed!, pack_halo_send_from_packed!
export owned_dot_packed, owned_norm²_packed, owned_dot_global_vecs
export allocate_halo_recv_buffers, allocate_halo_send_buffers, partitioned_matvec_workspace
export partitioned_mpi_owned_matvec_workspace, allocate_exchange_matvec_halo_mpi_requests
export simulate_halo_recv_from_global!, partitioned_owned_matvec!, partitioned_owned_internal_force!, exchange_matvec_halos_mpi!
export mpi_owned_dot_global, mpi_owned_dot_local, mpi_partitioned_operator_matvec!, mpi_partitioned_operator_matvec_owned!, mpi_partitioned_internal_force_owned!

# Matrix-free operator + constraints + loads + preconditioners + eigensolve
export AbstractMatrixFreeOperator, MatrixFreeOperator, MatrixFreeMassOperator,
    MatrixFreeOperatorKA, MatrixFreeMassOperatorKA, matrix_free_op_ka,
    matrix_free_mass_op_ka
export InternalForceOperator, NonlinearResidualOperator
export matrix_free_op, internal_force_op, nonlinear_residual_op

export AbstractDirichletConstraint, PenaltyDirichlet, EliminatedDirichlet
export AbstractMultipointConstraint, LinearMPC
export apply_constraint!, apply_constraint_pre!, apply_constraint_post!, apply_penalty_dirichlet_post_owned!, apply_penalty_dirichlet_post_ap_owned!
export eliminated_dirichlet_free_indices,
       extract_eliminated_dirichlet_subsystem,
       prolongate_eliminated_dirichlet_solution
export apply_constraint_diag!, apply_constraint_block_diag!

export AbstractNeumannLoad, NodalForce, UniformBodyForce, UniformMixedDarcySource,
       MixedDarcyTet4BoundaryNormalFluxLoad, MixedDarcyHex8BoundaryNormalFluxLoad,
       SurfaceLoad, SurfaceScalarFluxOnField
export apply_load!

export JacobiPreconditioner,
    BlockJacobiPreconditioner,
    ApproxSchurDiagBlockPreconditioner,
    ApproxSchurICholDiagBlockPreconditioner,
    ICholPreconditioner
export compute_diagonal!, compute_block_diagonal!

export lowest_eigenpairs, solve_eigenproblem

# ---------------------------------------------------------------------------
# Domain kernels and per-domain integration helpers
# ---------------------------------------------------------------------------
export ContinuumKernel
export material_lab_single_hex8_brick,
    hex8_symmetric_uniaxial_eliminated_dirichlet,
    material_lab_linear_elastic_uniaxial_solve
export MixedUPKernel
export StokesMixedKernel
export FacetMassKernel, EdgeMassKernel
export HellingerReissnerKernel
export HuWashizuKernel
export compute_block!, compute_block_at_point
export update_geometry_cache!, update_element_cache!, update_material_cache!, update_material_cache_stateless_strain!

export HeatKernel, DarcyPotentialKernel,
    AbstractDarcyMixedRT0P0Kernel, DarcyMixedRT0P0Kernel, DarcyMixedHex8RT0P0Kernel
export ThermoElasticKernel, ThermoElasticQPBuffer, BiotPoroelasticKernel
export ThermoPoroelasticKernel, ThermoPoroelasticQPBuffer

# ---------------------------------------------------------------------------
# Element template
# ---------------------------------------------------------------------------
export Element

# ---------------------------------------------------------------------------
# Sparse matrix scratch types
# ---------------------------------------------------------------------------
# `SparseMatrixCOO` / `SparseVectorCOO` are still consumed by the legacy
# pipeline (`JuliaFEM.Legacy`), so they remain part of the default export
# surface. The CSC-only convenience helpers (`get_nonzero_rows`,
# `get_nonzero_columns`, `resize_sparse`, `resize_sparsevec`) used to live
# next to them but have moved into `JuliaFEM.Legacy.SparseHelpers` since
# they are not used by the modern path.
export SparseMatrixCOO, SparseVectorCOO
