<!--
SPDX-FileCopyrightText: 2015-2026 Jukka Aho
SPDX-License-Identifier: MIT
-->

# src/assemblers/dof_based/

DOF-by-DOF assembler. The driver loops over global DOFs (rather than
elements) and writes one row of the global system at a time. This
removes the per-element gather/scatter overhead, eliminates atomic
adds for GPU execution, and lets the same kernel run on CPU and on
every KernelAbstractions backend without code duplication.

## Files

- `kernel_column.jl` — `UniformKernelColumn` (one kernel for all elements) and `PerElementKernelColumn` (one kernel per element id). `assert_homogeneous_dof_based_kernel_column!` validates compatible kernels before cache construction. `ka_per_element_kernel_column_supported` gates whether `DOFBasedCOOCacheKA` may be built for a per-element column (KA `apply_K!` passes a single prototype kernel; see below).
- `dof_based_coo.jl` — CPU implementation. Defines `DOFBasedCOOAssembler`, `DOFBasedCOOCache`, and the matrix-free entry points (`apply_K!`, `apply_K_contributions!`, `apply_M!`, `assemble_M!`, `extract_system`). Built around the microkernel contract in `assemblers/microkernel.jl` and the DOF connectivity in `dofs/dof_connectivity.jl`.
- `dof_based_coo_ka.jl` — Backend-agnostic GPU port via `KernelAbstractions.jl`. Defines `DOFBasedCOOCacheKA`, `sync_from_cpu!`, and the precision helpers used by the Float32 GPU pipeline. The same kernel runs on `CPU()`, `CUDABackend()`, `MetalBackend()`, `AMDGPUBackend()`, `oneAPIBackend()`; the in-tree CI validates the CPU() backend.

## Kernel columns and KA `apply_K!`

The KA matvec (`apply_K!` on `DOFBasedCOOCacheKA`) passes one prototype kernel
object into `evaluate_entry` for every element row, together with per-element
views of `qp_buffers` filled during Pass 1 on the CPU cache. That matches the
CPU path only if every element’s stiffness contribution either reads material
data from `qp_buffers` alone, or reads extra scalars that are already forced
identical across the column (density, Biot `α` / `storage_S`, thermo `β`,
thermo-poro coupling scalars, and so on). For a `PerElementKernelColumn{K}`,
`ka_per_element_kernel_column_supported` is `true` when `K` is one of
`ContinuumKernel`, `HeatKernel`, `ThermoElasticKernel`, `BiotPoroelasticKernel`,
or `ThermoPoroelasticKernel`. Other kernel types still work on the CPU column
path with `kernel_at(cache, eid)`; extending the KA gate requires checking that
`evaluate_entry` does not depend on per-element fields in the kernel object
beyond what Pass 1 copies into `qp_buffers`.

`apply_M!(y, cache_ka, kernel, x)` mirrors the same layout, calling
`evaluate_mass_entry` on the chosen backend (same per-element column gate as
`apply_K!`).

For Krylov-style use, `MatrixFreeMassOperatorKA` / `matrix_free_mass_op_ka` in
`assemblers/matrix_free/operator.jl` wraps the same matvec with device-matched
scratch buffers (`test/assemblers/test_matrix_free_mass_operator_ka.jl`).

## Design notes

The hot path is intentionally a tight loop over global DOFs; a Pass 1
fills shape-function values, derivatives and detJ for every quadrature
point in the element batch, and Pass 2 evaluates the per-row entries
via `evaluate_entry`. The constraint and load infrastructure in
`assemblers/matrix_free/` plugs into this same Pass 1 / Pass 2 split.

Zero-allocation invariants are enforced by
`test/assemblers/test_dof_based_zero_alloc.jl`.

## Partitioned element sets (distributed matvec stepping stone)

`apply_K!` visits each global DOF once and sums every incident element;
`apply_K_contributions!` loops only over a subset of elements and adds into
`y` (`+=`). If two subsets are disjoint and their union is all elements,
summing the partial vectors equals `apply_K!` on the same `x` (reference
test: `test/assemblers/test_dof_based_partitioned_matvec.jl`). If subsets
overlap, contributions double-count until the overlap is handled elsewhere.

Lightweight partition metadata (`MeshPartitionLayout`,
`element_indices_for_part`, `brick_hex_partition_slabs`, `referenced_global_dofs`,
`element_counts_by_part`, ...) lives in `assemblers/partitioning.jl`. For MPI-style
staging without GC traffic, preallocate a DOF-sized [`BitVector`](@ref) and use
`mark_referenced_dofs!`, `collect_true_indices!`, `fill_referenced_dof_indices!`,
`ghost_dof_mask!`, `node_partition_owner_min!`, and `mark_owned_vertex_field_dofs!`
(zero allocations once buffers exist — see `test/assemblers/test_partitioning_zero_alloc.jl`).
Neighbor partitions and MPI-style send/recv DOF lists live in `assemblers/halo_exchange.jl`
(`build_partition_adjacency`, `build_rank_halo_exchanges`, `ReferenceMaskMultiplyLayout`).
For `MatrixFreeOperator`, the buffer filled before `apply_K!` is controlled
by `AbstractMultiplyGhostLayout` / `prepare_multiply_workspace!` (default:
copy `x`), which is the extension point for ghost DOFs under MPI.

MPI helpers (weak dependency `MPI`, extension `ext/JuliaFEMMPIExt.jl`): halo exchange
`exchange_matvec_halos_mpi!`, owned-row product `mpi_partitioned_operator_matvec_owned!`,
workspace `partitioned_mpi_owned_matvec_workspace`, and reusable request buffers via
`allocate_exchange_matvec_halo_mpi_requests` / keyword `mpi_requests` (see the MPI subsection in `AGENTS.md`).
