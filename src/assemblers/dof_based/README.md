# src/assemblers/dof_based/

DOF-by-DOF assembler. The driver loops over global DOFs (rather than
elements) and writes one row of the global system at a time. This
removes the per-element gather/scatter overhead, eliminates atomic
adds for GPU execution, and lets the same kernel run on CPU and on
every KernelAbstractions backend without code duplication.

## Files

- `dof_based_coo.jl` — CPU implementation. Defines `DOFBasedCOOAssembler`, `DOFBasedCOOCache`, and the matrix-free entry points (`apply_K!`, `apply_K_contributions!`, `apply_M!`, `assemble_M!`, `extract_system`). Built around the microkernel contract in `assemblers/microkernel.jl` and the DOF connectivity in `dofs/dof_connectivity.jl`.
- `dof_based_coo_ka.jl` — Backend-agnostic GPU port via `KernelAbstractions.jl`. Defines `DOFBasedCOOCacheKA`, `sync_from_cpu!`, and the precision helpers used by the Float32 GPU pipeline. The same kernel runs on `CPU()`, `CUDABackend()`, `MetalBackend()`, `AMDGPUBackend()`, `oneAPIBackend()`; the in-tree CI validates the CPU() backend.

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
`allocate_exchange_matvec_halo_mpi_requests` / keyword `mpi_requests` (see `AGENTS.md` §3.6).
