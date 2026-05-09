# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
    JuliaFEM

Open-source finite element framework written in Julia.

The package is **0.x**; the repository is in the middle of a deliberate
architectural reset toward a **stable 1.0** with a type-stable, zero-allocation,
GPU-friendly assembly pipeline.
The authoritative current-architecture summary lives in `AGENTS.md` in
the repository root; the `docs/src/` tree contains a current-API quick
start and the auto-generated API reference.

Older functionality (`Problem` / `Assembly` / `Solver` / `Analysis`
hierarchy, Dict-based fields, the Abaqus mesh reader stack, ...) is
preserved in the optional `JuliaFEM.Legacy` submodule. Set the
environment variable `JULIAFEM_ENABLE_LEGACY=1` before `using JuliaFEM`
to load it.

Because the legacy gate is evaluated at module-load time, the precompile
cache encodes whichever mode was active when the cache was built.
Switching between modes requires re-precompilation: either run
`Pkg.precompile()` after changing the variable, or delete the cache
under `~/.julia/compiled/v\$VERSION/JuliaFEM/`.
"""
module JuliaFEM

import Base: getindex, setindex!, convert, length, size, isapprox,
   ==, haskey, copy, read, append!

using SparseArrays, LinearAlgebra
using Logging
using Tensors
using StaticArrays: SVector

# No-op timing macro retained for compatibility with annotated hot paths.
macro timeit(args...)
    return esc(args[end])
end

# ============================================================================
# Public API surface
#
# All `export` statements live in a single, grouped, deduplicated file. They
# are reservations of names; the symbols themselves are defined by the
# `include`s below.
# ============================================================================
include("exports.jl")

# ============================================================================
# Level 1 - Field, formulation, material, mesh, physics, topology API tags
# ============================================================================
include("domains/continuum/abstract.jl")
include("domains/continuum/types.jl")

include("fields/api.jl")
include("fields/local_field.jl")

include("materials/api.jl")

include("mesh/api.jl")

include("physics/abstract.jl")
include("physics/api.jl")
include("physics/types.jl")
include("physics/strain.jl")

# ============================================================================
# Level 2 - Topology shape types
# ============================================================================
include("topology/api.jl")
include("topology/segments.jl")
include("topology/triangles.jl")
include("topology/quadrilaterals.jl")
include("topology/tetrahedra.jl")
include("topology/hexahedra.jl")
include("topology/pyramids.jl")
include("topology/wedges.jl")

# ============================================================================
# Level 3 - Quadrature, geometry, basis functions
# ============================================================================
include("quadrature/api.jl")
include("quadrature/quaddata.jl")
include("quadrature/gl_tetrahedra.jl")
include("quadrature/gl_triangles.jl")
include("quadrature/gl_wedges.jl")
include("quadrature/gl_pyramids.jl")
include("quadrature/gl_tensor_product.jl")
include("quadrature/gauss.jl")

include("geometry/jacobian.jl")
include("geometry/piola.jl")
include("geometry/strain.jl")

include("basis/api.jl")
include("quadrature/integration_basis.jl")
# basis_generated.jl is auto-generated. The generator
# (`src/basis/basis_generator.jl`) is a standalone script — do not
# `include` it from the package; it carries `__precompile__(false)` and
# would disable the entire module's precompile cache. Regenerate with:
#   julia --project=. src/basis/basis_generator.jl
include("basis/basis_generated.jl")
include("basis/nedelec_reference.jl")
include("basis/rt0_reference.jl")

# ============================================================================
# Level 4 - DOF system, sparse scratch, element template, materials
# ============================================================================
include("dofs/dofs.jl")

include("sparse/sparse.jl")

include("elements/elements.jl")
include("elements/extract_element_dofs.jl")
include("elements/interpolate.jl")

include("materials/state_variables.jl")
include("materials/traits.jl")
include("materials/field_traits.jl")
include("materials/global_material_cache.jl")
include("materials/continuum_kinematics.jl")
include("materials/symmetric_fourth_identity.jl")
include("materials/linear_elastic.jl")
include("materials/orthotropic_linear_elastic.jl")
include("materials/neo_hookean.jl")
include("materials/hyperelastic_models.jl")
include("materials/perfect_plasticity.jl")
include("materials/j2_isotropic_plasticity.jl")
include("materials/stvenant_kirchhoff_j2.jl")
include("materials/scalar_damage.jl")
include("materials/chaboche_j2.jl")
include("materials/norton_creep.jl")
include("materials/eigenstrain_elastic.jl")
include("materials/heat_conductivity.jl")
include("materials/moisture_diffusivity.jl")
include("materials/hydraulic_conductivity.jl")
include("materials/element_wise_scalar_diffusion.jl")

# ============================================================================
# Level 5 - Assembly framework
# ============================================================================
include("assemblers/abstract.jl")

# Per-element scratch (geometry / element / material) plus the global COO
# scratch shared by every assembler.
include("assemblers/caches/element_cache.jl")
include("assemblers/caches/geometry_cache.jl")
include("assemblers/caches/material_cache.jl")
include("assemblers/caches/coo_cache.jl")

# Element-based assembler (COO) and its scatter routines.
include("assemblers/element_based/element_based_coo.jl")

# Microkernel contract for the DOF-based / matrix-free path. Implementations
# live in `domains/<physics>/kernel.jl`.
include("assemblers/microkernel.jl")

# ============================================================================
# Level 6 - Physical domains (assembly kernels per discipline)
# ============================================================================
include("domains/continuum/kernel.jl")
include("domains/continuum/update_geometry_cache.jl")
include("domains/continuum/update_element_cache.jl")
include("domains/continuum/update_material_cache.jl")
include("domains/continuum/mixed_up_kernel.jl")
include("domains/continuum/stokes_mixed_kernel.jl")
include("domains/continuum/hellinger_reissner_kernel.jl")
include("domains/continuum/hu_washizu_kernel.jl")

include("domains/heat/kernel.jl")
include("domains/darcy/potential.jl")
include("domains/darcy/mixed_rt0.jl")
include("domains/darcy/mixed_rt0_hex.jl")

include("domains/thermo_elastic/kernel.jl")

# ============================================================================
# Level 7 - Concrete mesh + DOFHandler + DOF-based assembler + matrix-free
# ============================================================================
include("mesh/mesh.jl")
include("mesh/hex8_facet_maps.jl")
include("mesh/tet4_facet_maps.jl")
include("mesh/wedge6_facet_maps.jl")
include("mesh/pyr5_facet_maps.jl")

# Connectivity value types must be available before DOFHandler so the
# handler can store `Union{Nothing, DOFConnectivity}` instead of `Any`.
# The builders still live after DOFHandler because they reference it.
include("dofs/dof_connectivity_types.jl")
include("dofs/dof_handler.jl")
include("domains/continuum/facet_mass_kernel.jl")
include("domains/continuum/edge_mass_kernel.jl")
include("dofs/dof_connectivity.jl")

include("interface/interface_mesh.jl")
include("interface/interface_dof_handler.jl")

# DOF-based assembler — CPU plus the backend-agnostic KernelAbstractions
# port (CPU(), CUDABackend(), MetalBackend(), AMDGPUBackend(), oneAPIBackend();
# locally validated against CPU()).
include("assemblers/dof_based/dof_based_coo.jl")
include("assemblers/dof_based/dof_based_coo_ka.jl")

# Partition metadata + multiply-buffer layouts (MPI/GPU hooks; default is
# single-process identity).
include("assemblers/partitioning.jl")
include("assemblers/halo_exchange.jl")
include("assemblers/packed_layout.jl")
include("assemblers/partitioned_matvec.jl")

# Matrix-free path: declarative Dirichlet / MPC constraints, declarative
# Neumann loads, matrix-free preconditioners and the generalized eigensolver.
# All operate on `DOFBasedCOOCache` + `apply_K!` / `apply_M!` and share the
# `apply_constraint_*` hook protocol so they compose freely.
include("assemblers/matrix_free/dirichlet.jl")
include("assemblers/matrix_free/mpc.jl")
include("assemblers/matrix_free/operator.jl")
include("assemblers/matrix_free/preconditioners.jl")
include("assemblers/matrix_free/eigensolve.jl")
include("assemblers/matrix_free/loads.jl")

# ============================================================================
# Level 8 - Mesh utilities and I/O
# ============================================================================
include("mesh/refine.jl")
include("mesh/structured.jl")

include("domains/continuum/material_element_lab.jl")

# Self-contained Gmsh mesh reader (defines its own `JuliaFEM.GmshReader`
# submodule with `read_gmsh_mesh` / `GmshMesh`).
include("io/gmsh_reader.jl")

# ============================================================================
# Optional older API surface (Problem / Assembly / Solver / Analysis,
# Dict-based fields, Abaqus reader, ...). Loaded only when the user opts in
# via `JULIAFEM_ENABLE_LEGACY=1`. Names are accessible as
# `JuliaFEM.Legacy.<name>` and are not re-exported at the top level.
# ============================================================================
const ENABLE_LEGACY = get(ENV, "JULIAFEM_ENABLE_LEGACY", "0") == "1"
if ENABLE_LEGACY
    include("legacy/Legacy.jl")
end

"""
    exchange_matvec_halos_mpi!(recv_vals, send_vals, packed, layout, exchange, comm; mpi_requests = nothing)

Pack halo sends from `packed`, complete non-blocking MPI exchanges into `recv_vals`, then
return. Requires `using MPI` after `JuliaFEM` (loads the `JuliaFEMMPIExt` weak dependency).

Posts all [`MPI.Irecv!`](@ref) before [`MPI.Isend`](@ref); tags encode the directed partition
pair `(sender_part, receiver_part)`. Expects `MPI.Comm_rank(comm) + 1 == exchange.part`.

Keyword `mpi_requests`: optional persistent [`Vector{MPI.Request}`](@ref) from
[`allocate_exchange_matvec_halo_mpi_requests`](@ref), length
[`matvec_halo_mpi_request_count`](@ref)`(exchange)`. When `nothing`, a fresh vector is allocated each
call (fine for setup; Krylov inner loops should reuse a buffer).

Typical sequence (full trial workspace): [`gather_owned_from_global_to_packed!`](@ref), this function,
[`unpack_halo_recv_to_packed!`](@ref), [`expand_packed_to_global!`](@ref),
[`apply_K_owned_rows!`](@ref). Lean owned-only path: [`copy_owned_subset_to_packed_owned_prefix!`](@ref),
this function, [`unpack_halo_recv_to_packed!`](@ref), [`apply_K_owned_rows_from_packed!`](@ref).
"""
function exchange_matvec_halos_mpi! end

"""
    allocate_exchange_matvec_halo_mpi_requests(exchange::RankHaloExchange)

Preallocate storage for [`exchange_matvec_halos_mpi!`](@ref)`(; mpi_requests=…)`.
Requires `using MPI`. Length equals [`matvec_halo_mpi_request_count`](@ref)`(exchange)`.
"""
function allocate_exchange_matvec_halo_mpi_requests end

"""
    mpi_owned_dot_global(a, b, layout, comm) -> Float64

Parallel inner product `a'b` counting each global DOF once via [`owned_dot_global_vecs`](@ref)
on the local partition, summed with [`MPI.Allreduce`](@ref). Requires `using MPI` after
`JuliaFEM`.
"""
function mpi_owned_dot_global end

"""
    mpi_partitioned_operator_matvec!(
        Ap, p, packed, work, recv_vals, send_vals,
        layout, exchange, cache, assembler, kernel, mesh, comm;
        dirichlet = nothing, mpi_requests = nothing,
    )

Replicated-global matrix-free product including optional [`PenaltyDirichlet`](@ref) post-hook.
Loads `JuliaFEMMPIExt` when both `JuliaFEM` and `MPI` are imported.

Keyword `mpi_requests` is forwarded to [`exchange_matvec_halos_mpi!`](@ref).
"""
function mpi_partitioned_operator_matvec! end

"""
    mpi_owned_dot_local(a_owned, b_owned, comm) -> Float64

Local dot `sum_k a_owned[k]*b_owned[k]` reduced with [`MPI.Allreduce`](@ref)`(SUM)` — global inner
product when each global DOF is owned on exactly one rank and `a_owned`/`b_owned` hold owned
entries only (aligned with [`copy_owned_subset_to_packed_owned_prefix!`](@ref)).
"""
function mpi_owned_dot_local end

"""
    mpi_partitioned_operator_matvec_owned!(
        Ap_owned, p_owned, packed, recv_vals, send_vals,
        layout, exchange, cache, assembler, kernel, mesh, comm;
        dirichlet = nothing, mpi_requests = nothing,
    )

Lean MPI matvec: [`copy_owned_subset_to_packed_owned_prefix!`](@ref), halo exchange,
[`apply_K_owned_rows_from_packed!`](@ref), optional [`apply_penalty_dirichlet_post_ap_owned!`](@ref).
No `ndofs_global`-length trial workspace — trial DOFs stay in `packed`. **No** full-vector
[`MPI.Allreduce!`](@ref) on `Ap`.

Keyword `mpi_requests` is forwarded to [`exchange_matvec_halos_mpi!`](@ref).

Supports [`PenaltyDirichlet`](@ref) only on this path.
"""
function mpi_partitioned_operator_matvec_owned! end

end # module JuliaFEM
