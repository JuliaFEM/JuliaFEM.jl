# src/assemblers/matrix_free/

Declarative constraints, loads, preconditioners and the eigensolver
that operate on the DOF-based cache without ever forming the global
stiffness matrix. Everything here builds on `apply_K!` /
`apply_M!` and the `apply_constraint_*` hook protocol so that
Dirichlet, MPC and load contributions compose freely with both the
assembled and the matrix-free solves.

## Files

- `dirichlet.jl` — `AbstractDirichletConstraint`, `PenaltyDirichlet`, `EliminatedDirichlet`. Drive both the assembled solve (`apply_constraint!(K, c)` / `apply_constraint!(K, b, c)`) and the matrix-free path (`apply_constraint_pre!` / `apply_constraint_post!` wrapped around `apply_K!`). Also re-exports `matrix_free_op` as a thin factory for `MatrixFreeOperator`.
- `mpc.jl` — `AbstractMultipointConstraint`, `LinearMPC`. Penalty-enforced affine `u_s = sum(c_k * u_{m_k}) + g` constraints sharing the same hook protocol as Dirichlet, so they compose with both solve paths. The MPC contribution is folded into `MatrixFreeOperator` via the `mpc =` keyword.
- `operator.jl` — `AbstractMatrixFreeOperator`, `MatrixFreeOperator{C, A, K, M, D, P, L}`, `MatrixFreeMassOperator{C, A, K, M}`. Typed linear operators that implement `LinearAlgebra.mul!`, `eltype`, `size`, `*`, and a callable form (`op(y, x)`) so they plug into `LinearOperators.LinearOperator(...)`. Each operator owns its work buffer, so every mat-vec is allocation-free after warmup. The `MatrixFreeOperator` constructor accepts `dirichlet =`, `mpc =`, and `multiply_layout =` keywords; the layout drives `prepare_multiply_workspace!` before `apply_K!` (`LocalMultiplyLayout` copies `x`; future MPI layouts may fill ghost DOFs). Constraint hooks still compose as before.
- `loads.jl` — `AbstractNeumannLoad`, `NodalForce`, `UniformBodyForce`, `SurfaceLoad`, plus `apply_load!`. Reuses the cache's SoA `N_data` / `detJ_w` batches so body-force integration is allocation-free and shares Pass 1 with `apply_K!` / `apply_M!`.
- `preconditioners.jl` — `JacobiPreconditioner`, `BlockJacobiPreconditioner`, `ICholPreconditioner` plus `compute_diagonal!` / `compute_block_diagonal!` and the `apply_constraint_*` diagonal hooks. Closes the conditioning gap of the penalty-Dirichlet matrix-free path so unpreconditioned CG still converges.
- `eigensolve.jl` — `lowest_eigenpairs`, `solve_eigenproblem`. Subspace iteration with Rayleigh-Ritz built directly on `MatrixFreeOperator` and `MatrixFreeMassOperator`. Supports closure-style operators (for backward compatibility) and assembled `K`, `M` matrices.

## Design notes

`MatrixFreeOperator` is the thread that ties this directory together.
Every consumer (the eigensolver, the Krylov solves in
`test/assemblers/`, downstream user code) goes through the same typed
operator, which means:

- The `dirichlet =` and `mpc =` constraint hooks compose without ad-
  hoc closures at the call site.
- The work buffer is owned by the operator, so the operator is
  reusable across many mat-vecs without re-allocating.
- The standard `LinearAlgebra.mul!` / `size` / `eltype` interface
  makes the operator a drop-in for any solver that takes an abstract
  linear operator (Krylov, eigensolvers, ...).

Partition metadata and multiply-buffer layouts (`AbstractMultiplyGhostLayout`,
`LocalMultiplyLayout`, `ReferenceMaskMultiplyLayout`, `MeshPartitionLayout`, …)
live in `assemblers/partitioning.jl`; MPI-style neighbor / send / recv DOF lists
are built in `assemblers/halo_exchange.jl`. See `assemblers/dof_based/README.md`
for how these tie to `apply_K_contributions!`.
