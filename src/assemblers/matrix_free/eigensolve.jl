# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

import IterativeSolvers
import LinearOperators

"""
Matrix-free generalized eigensolve `K φ = λ M φ`.

Computes the smallest few eigenpairs of an SPD generalized
eigenproblem using simultaneous inverse iteration with Rayleigh-Ritz
projection (a.k.a. subspace iteration). The algorithm is matrix-free:
it only requires linear operators that compute `K * x` and `M * x`,
plus an inner Krylov solve (`K * X = M * V`) per outer iteration.

Algorithm sketch (per outer iteration on a subspace `V ∈ ℝⁿˣᵖ`):

  1. `Z = M * V`                  (matrix-free `apply_M!` per column)
  2. `X = K \\ Z`                  (matrix-free CG per column)
  3. M-orthonormalize `X`         (block Gram-Schmidt in the M inner product)
  4. Rayleigh-Ritz on `X`:
        K̂ = X' (K X), M̂ = X' (M X)
        solve `K̂ Q = M̂ Q Λ` (small dense generalized eigenproblem)
  5. `V ← X * Q`, sort by Λ
  6. check `|Λ_k − Λ_k_prev| / |Λ_k| < tol` for the lowest `nev` modes

The subspace size `p ≥ nev` provides oversampling that improves
robustness; `p = nev + 4` is a sensible default.

# API

```julia
λ, V = lowest_eigenpairs(op_K, op_M, n; nev = 1, tol = 1e-8,
                                            maxiter = 200, p = nothing)

λ, V = lowest_eigenpairs(K::AbstractMatrix, M::AbstractMatrix; nev = 1, …)
```

`op_K(y, x) -> y` and `op_M(y, x) -> y` are in-place mat-vec closures.
The returned `λ::Vector{Float64}` has length `nev` (ascending), and
`V::Matrix{Float64}` is `n × nev` with `M`-orthonormal columns
(`V' * (M * V) ≈ I`).

The returned `V` satisfies the orthogonality identity but the
*eigenvector residuals* `‖K v_k − λ_k M v_k‖` are governed by the
inner-CG tolerance `cg_tol` (default = `tol²`, so the eigenproblem
converges before the CG solves limit accuracy).
"""

# ---------------------------------------------------------------------------
# Low-level: matrix-free / matrix-aware eigensolver
# ---------------------------------------------------------------------------

"""
    lowest_eigenpairs(op_K, op_M, n; nev = 1, tol = 1e-8,
                                       maxiter = 200, p = nothing,
                                       cg_tol = nothing,
                                       cg_maxiter = nothing,
                                       preconditioner = nothing,
                                       verbose = false) -> (λ, V)

Compute the lowest `nev` eigenpairs of `K φ = λ M φ` using subspace
iteration. `op_K(y, x)` and `op_M(y, x)` are in-place matrix-free
operators. `n` is the problem dimension.

Inner CG solves use `LinearOperators.LinearOperator` over `op_K` with
tolerance `cg_tol` (defaults to `tol^2`). Optional `preconditioner`
plugs into the inner CG (any IterativeSolvers-compatible `Pl`).
"""
const _OpKind = Union{Function,AbstractMatrixFreeOperator}

function lowest_eigenpairs(op_K::_OpKind, op_M::_OpKind, n::Int;
                           nev::Int = 1,
                           tol::Real = 1e-8,
                           maxiter::Int = 200,
                           p::Union{Nothing,Int} = nothing,
                           cg_tol::Union{Nothing,Real} = nothing,
                           cg_maxiter::Union{Nothing,Int} = nothing,
                           preconditioner = nothing,
                           verbose::Bool = false)
    @assert nev > 0      "lowest_eigenpairs: nev must be ≥ 1 (got $nev)"
    @assert n   >= nev   "lowest_eigenpairs: n ($n) must be ≥ nev ($nev)"

    p_eff      = p === nothing ? min(n, nev + 4) : p
    @assert p_eff >= nev  "lowest_eigenpairs: subspace size p ($p_eff) must be ≥ nev"

    cg_tol_eff      = cg_tol === nothing ? max(tol * tol, 1e-14) : cg_tol
    cg_maxiter_eff  = cg_maxiter === nothing ? max(2 * n, 200) : cg_maxiter

    # Pre-allocated workspace.
    V    = randn(n, p_eff)         # current subspace
    Z    = zeros(n, p_eff)         # M * V
    X    = zeros(n, p_eff)         # K^{-1} * (M * V)
    KX   = zeros(n, p_eff)
    MX   = zeros(n, p_eff)
    yvec = zeros(n)                # column scratch

    K_op_lin = LinearOperators.LinearOperator(Float64, n, n, true, true, op_K)

    λ_prev = fill(Inf, p_eff)
    λ      = zeros(p_eff)
    Q      = zeros(p_eff, p_eff)

    for it in 1:maxiter
        # 1. Z = M V (column-wise)
        @inbounds for j in 1:p_eff
            op_M(yvec, view(V, :, j))
            copyto!(view(Z, :, j), yvec)
        end

        # 2. X = K \ Z (column-wise CG)
        @inbounds for j in 1:p_eff
            xj = view(X, :, j); fill!(xj, 0.0)
            zj = copy(view(Z, :, j))
            if preconditioner === nothing
                IterativeSolvers.cg!(xj, K_op_lin, zj;
                                     abstol = cg_tol_eff, reltol = cg_tol_eff,
                                     maxiter = cg_maxiter_eff)
            else
                IterativeSolvers.cg!(xj, K_op_lin, zj;
                                     Pl = preconditioner,
                                     abstol = cg_tol_eff, reltol = cg_tol_eff,
                                     maxiter = cg_maxiter_eff)
            end
        end

        # 3. K X, M X for Rayleigh-Ritz
        @inbounds for j in 1:p_eff
            op_K(yvec, view(X, :, j)); copyto!(view(KX, :, j), yvec)
            op_M(yvec, view(X, :, j)); copyto!(view(MX, :, j), yvec)
        end

        # Small projected matrices. We don't wrap them in
        # `Symmetric(...)` even though they are mathematically symmetric:
        # `eigen(::Symmetric, ::Symmetric)` dispatches to LAPACK's
        # `sygv`, which requires *both* arguments to be SPD. For our
        # generalized FEM eigenproblems `M̂` is SPD but `K̂` is only
        # positive *semi*-definite (it has zero eigenvalues for
        # rigid-body / constant-mode null spaces, with floating-point
        # noise occasionally pushing them slightly negative inside the
        # subspace). The unwrapped `eigen` falls back to the
        # `ggev`/`ggev3` general path which handles this cleanly.
        K̂ = X' * KX
        M̂ = X' * MX
        K̂ = (K̂ + K̂') / 2     # numerical symmetrization
        M̂ = (M̂ + M̂') / 2

        # 4. Small dense generalized eigenproblem.
        F   = LinearAlgebra.eigen(K̂, M̂)
        idx = sortperm(real.(F.values))
        λ  .= real.(F.values[idx])
        Q  .= real.(F.vectors[:, idx])

        # 5. Update subspace V ← X * Q.
        LinearAlgebra.mul!(V, X, Q)

        # 6. Convergence on the lowest `nev` Λ.
        rel_err = 0.0
        @inbounds for k in 1:nev
            denom = max(abs(λ[k]), tol)
            rel_err = max(rel_err, abs(λ[k] - λ_prev[k]) / denom)
        end
        if verbose
            @info "lowest_eigenpairs: subspace iteration" iter=it nev=nev λ=λ[1:nev] rel_err=rel_err
        end

        if rel_err < tol
            # M-normalize columns of V (Q from generalized eigen is
            # already M-orthonormal up to the floating-point error of
            # the small eigensolve, so this is a final cleanup).
            _M_normalize_cols!(V, op_M, yvec)
            return (λ[1:nev], V[:, 1:nev])
        end

        copyto!(λ_prev, λ)
    end

    error("lowest_eigenpairs: subspace iteration did not converge in " *
          "$maxiter outer iterations (last rel_err on the lowest $nev " *
          "eigenvalues = $(λ - λ_prev)). Try increasing `maxiter`, " *
          "loosening `tol`, or enlarging the subspace via `p`.")
end

# In-place mat-vec closure for an assembled matrix `A`, matching the
# `op(y, x) -> y` calling convention used by `lowest_eigenpairs`.
_dense_mul_op(A::AbstractMatrix) = (y, x) -> (LinearAlgebra.mul!(y, A, x); y)

# Convenience overload: assembled real-valued `K`, `M` matrices. The
# subspace iteration itself uses Float64 workspaces, so matrix-vector
# products from other real element types are converted through the output
# vector supplied by `mul!`.
function lowest_eigenpairs(K::AbstractMatrix{<:Real},
                           M::AbstractMatrix{<:Real};
                           kwargs...)
    n = size(K, 1)
    @assert size(K) == (n, n) "lowest_eigenpairs: K must be square"
    @assert size(M) == (n, n) "lowest_eigenpairs: M must be square"
    return lowest_eigenpairs(_dense_mul_op(K), _dense_mul_op(M), n; kwargs...)
end

# In-place M-normalization of the columns of `V`. Each column is scaled
# so that `vᵀ M v = 1`; columns with `vᵀ M v ≤ 0` (numerical drift) are
# left unchanged so the caller can detect them.
function _M_normalize_cols!(V::AbstractMatrix{Float64},
                            op_M::_OpKind,
                            scratch::Vector{Float64})
    @inbounds for j in 1:size(V, 2)
        vj = view(V, :, j)
        op_M(scratch, vj)
        nrm2 = LinearAlgebra.dot(vj, scratch)
        if nrm2 > 0.0
            s = 1.0 / sqrt(nrm2)
            @simd for i in eachindex(vj)
                vj[i] *= s
            end
        end
    end
    return V
end


# ---------------------------------------------------------------------------
# High-level wrapper: build matrix-free K and M operators from the
# DOF-based assembler and solve directly. Currently supports the
# unconstrained case; constraint handling is left to the caller.
# ---------------------------------------------------------------------------

"""
    solve_eigenproblem(cache, asm, mesh;
                       nev = 1, tol = 1e-8, maxiter = 200,
                       p = nothing, verbose = false,
                       dirichlet = nothing, mpc = nothing,
                       shift = 0.0) -> (λ, V)
    solve_eigenproblem(cache, asm, kernel, mesh; …)

Convenience wrapper around `lowest_eigenpairs`: assembles matrix-free
`K` and `M` operators (via `apply_K!` / `apply_M!`) and runs subspace
iteration to extract the lowest `nev` generalized eigenpairs of
`K φ = λ M φ`. Volume kernels are read from `cache.kernel_column`.

The four-argument form ignores `kernel` (backward compatibility; emits
`Base.depwarn` once per session, same as the matrix-free operator overloads).

`dirichlet` and `mpc` are forwarded to [`MatrixFreeOperator`](@ref) so the
constrained operator is solved directly. `shift` adds `σ M` to `K`
internally and subtracts `σ` from the returned eigenvalues — useful
for problems with rigid-body / null-space modes (free-free elasticity,
unconstrained heat) where the unshifted `K` is singular and the inner
CG cannot invert it. A shift slightly larger than the smallest
non-trivial eigenvalue is sufficient.
"""
function solve_eigenproblem(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    nev::Int = 1,
    tol::Real = 1e-8,
    maxiter::Int = 200,
    p::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
    dirichlet = nothing,
    mpc = nothing,
    shift::Real = 0.0,
)
    n = cache.ndofs
    op_K_base = MatrixFreeOperator(cache, asm, mesh;
                                   dirichlet = dirichlet, mpc = mpc)
    op_M      = MatrixFreeMassOperator(cache, asm, mesh)

    # Optional shift: K_shift = K + σ M  ⇒  λ_shift = λ + σ.
    op_K = if shift == 0.0
        op_K_base
    else
        σ = Float64(shift)
        scratch = zeros(Float64, n)
        function (y, x)
            LinearAlgebra.mul!(y, op_K_base, x)
            LinearAlgebra.mul!(scratch, op_M, x)
            @inbounds @simd for i in eachindex(y)
                y[i] += σ * scratch[i]
            end
            return y
        end
    end

    λ, V = lowest_eigenpairs(op_K, op_M, n;
                             nev = nev, tol = tol, maxiter = maxiter,
                             p = p, verbose = verbose)
    if shift != 0.0
        λ = λ .- Float64(shift)
    end
    return (λ, V)
end

@inline function solve_eigenproblem(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    kwargs...,
)
    _depwarn_redundant_kernel_arg!(:solve_eigenproblem)
    return solve_eigenproblem(cache, asm, mesh; kwargs...)
end
