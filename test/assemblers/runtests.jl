using Test
using JuliaFEM

@testset "Assemblers" begin
    # Locks in zero-allocation + correctness + 0 LLVM allocation sites
    # for the element-template-driven DOF-based assembler.
    include("test_dof_based_zero_alloc.jl")

    # Matrix-free `apply_K!` correctness vs assembled K, zero
    # allocations, 0 LLVM gc-alloc sites, and a Krylov.cg validation
    # via LinearOperators + IterativeSolvers.
    include("test_dof_based_apply_K.jl")

    # Disjoint element subsets + `apply_K_contributions!` sum to `apply_K!`
    # (fake two-rank reference); partition metadata + multiply-buffer hook.
    include("test_dof_based_partitioned_matvec.jl")

    # Full matvec + row mask; disjoint vertex-owned rows sum to `apply_K!`.
    include("test_apply_K_masked_rows.jl")

    # Structured Hex8 slab partitions, DOF closures, contribution sums.
    include("test_partitioning_helpers.jl")

    include("test_partitioning_zero_alloc.jl")

    # Partition adjacency, halo DOF lists, ReferenceMaskMultiplyLayout.
    include("test_halo_exchange.jl")

    # Per-partition packed DOF layout + gather/expand + matvec glue.
    include("test_packed_layout.jl")

    # partitioned_owned_matvec! orchestration (serial halo replica).
    include("test_partitioned_matvec.jl")

    # Backend-agnostic apply_K! via KernelAbstractions: same kernel
    # on CPU(), CUDABackend(), MetalBackend(), AMDGPUBackend(),
    # oneAPIBackend(). Locally validates the CPU() backend produces
    # bit-equivalent output to the direct CPU apply_K!.
    include("test_dof_based_apply_K_ka.jl")

    # Mass-matrix microkernel through the DOF-based assembler.
    # Validates `evaluate_mass_entry`, `apply_M!`, and `assemble_M!`
    # for both `ContinuumKernel` and `HeatKernel`, including
    # row-sum (= rho * V), block-diagonal-in-components structure for
    # elasticity, density linearity, and the same zero-alloc
    # contract as `apply_K!` / `assemble!`.
    include("test_dof_based_mass.jl")

    # Neumann loads (NodalForce + UniformBodyForce) through
    # `apply_load!`. Locks in row-sum identity (int b dV = b dot V),
    # additive composition, end-to-end Poisson with body source vs
    # the analytical T(x) = Q x (L - x) / (2 k) solution, and
    # zero allocations for both load types.
    include("test_dof_based_loads.jl")

    # `BlockJacobiPreconditioner{N}` for vector problems where the
    # 3x3 nodal block has full off-diagonal coupling. Validates
    # `compute_block_diagonal!` against assembled K, that
    # `ldiv!(P, x)` is the exact block-diag inverse, that
    # PenaltyDirichlet + BlockJacobi CG matches the direct solve,
    # and that BlockJacobi reaches the same residual in <= as many
    # iterations as scalar Jacobi.
    include("test_block_jacobi.jl")

    # Float32 (single-precision) `apply_K!` through the
    # precision-parametric KernelAbstractions cache. Locks in the
    # storage typing of `to_float32(cache)`, F32-vs-F64 single-
    # precision agreement on both `ContinuumKernel` and
    # `HeatKernel`, the precision-mismatch guard, and the round-
    # trip back-compat of the default Float64 KA path.
    include("test_dof_based_apply_K_f32.jl")

    # `SurfaceLoad` distributed-traction integration via
    # `apply_load!`. Validates the row-sum identity
    # `Sigma f = t * area` for both quad (Hex8 face) and tri (Tet4
    # face) faces, end-to-end pull (3D elasticity) and 1D heat-
    # conduction problems, additive composition with
    # `UniformBodyForce`, and zero-allocation hot path.
    include("test_surface_load.jl")

    # `ICholPreconditioner` (IC(0) -- incomplete Cholesky with
    # zero fill-in). Locks in algebraic correctness on
    # tridiagonal/dense SPD matrices, the diagonal-shift retry
    # for near-indefinite inputs (without aliasing the input
    # `K`), `ldiv!` agreement with `(L * L')^{-1} * x`, fewer
    # PCG iterations than scalar Jacobi on a stiff elasticity
    # cantilever, end-to-end PenaltyDirichlet matrix-free PCG
    # via the `(cache, asm, kernel, mesh; dirichlet)` factory,
    # and zero-allocation `ldiv!`.
    include("test_ichol_preconditioner.jl")

    # `LinearMPC` -- penalty-enforced linear multipoint
    # constraints sharing the `apply_constraint_*` hook
    # protocol with the Dirichlet types. Locks in tuple ->
    # flat-CSR packing, assembled-vs-matrix-free agreement on
    # heat (periodic) and elasticity (multi-master),
    # end-to-end periodic-heat PCG matches the direct solve,
    # composition with `PenaltyDirichlet` for an
    # inhomogeneous-offset rigid-link elasticity solve, and
    # zero-alloc `apply_constraint_post!` / `_diag!`.
    include("test_linear_mpc.jl")

    # `lowest_eigenpairs` / `solve_eigenproblem` -- matrix-free
    # generalized eigensolve `K phi = lambda M phi` via subspace
    # iteration with Rayleigh-Ritz. Locks in algebraic
    # correctness on dense SPD test problems, matrix-free
    # apply_K!/apply_M! agreement with assembled K, M, recovery
    # of the analytical 1D heat spectrum, and the high-level
    # wrapper's shift-invert path for free-free systems with
    # rigid-body / null-space modes.
    include("test_eigensolve.jl")

    # Typed `MatrixFreeOperator` / `MatrixFreeMassOperator` --
    # `LinearAlgebra.mul!`, `eltype`, `size`, `op(y, x)` callable,
    # `op * x` allocating mat-vec, optional Dirichlet folding into
    # every mat-vec, plug-in via `LinearOperators.LinearOperator`
    # into `IterativeSolvers.cg!`, and zero allocations on `mul!`
    # after warmup.
    include("test_matrix_free_operator.jl")
end
