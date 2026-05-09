# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Dirichlet boundary conditions for the DOF-based assembler.

The matrix-free path (`apply_K!`) has no notion of constrained DOFs by
itself — it just computes `y = K * x`. This module provides a small set
of reusable constraint types that let the same declarative description
of the boundary serve both the assembled and the matrix-free
solve paths.

Two flavours are provided:

* `PenaltyDirichlet` — adds `λ` to the diagonal of the fixed DOFs (and
  `λ * T̂` to the RHS for inhomogeneous BCs). Cheap and trivially
  invertible by `Kbc \\ b`. Pure CG without a preconditioner can stall
  on large `λ` because of the resulting condition number; pair it with
  a Jacobi preconditioner or use `EliminatedDirichlet` instead.

* `EliminatedDirichlet` — proper row/column elimination. Conditioning
  follows the free-DOF block `K_ff` of the original system, not the
  penalty parameter, so unpreconditioned CG on the matrix-free operator
  converges as if the BCs weren't there. Inhomogeneous BCs are
  supported via the standard "lift the RHS" trick (`b ← b − K · û`).

Each constraint type knows how to:

```julia
c = PenaltyDirichlet(fixed_dofs)                # or EliminatedDirichlet
c = PenaltyDirichlet(fixed_dofs, T̂ ; penalty)   # inhomogeneous

apply_constraint!(K::AbstractMatrix, c)         # before a direct solve
apply_constraint!(K::AbstractMatrix, b, c)      # K + b together (inhomogeneous)
apply_constraint!(b::AbstractVector, c)         # RHS contribution
apply_constraint_pre!(workbuf, x, c)            # matrix-free hook (before apply_K!)
apply_constraint_post!(y, x, c)                 # matrix-free hook (after apply_K!)
```

`matrix_free_op` glues `apply_K!` + the two hooks into a closure
suitable for `LinearOperators.LinearOperator` / Krylov solvers — no
hard dependency on `LinearOperators.jl`.

Penalty `λ` defaults to `1e16`, large enough to drive the solution at
the fixed DOFs to within ~1/λ of the prescribed value.
"""

abstract type AbstractDirichletConstraint end

# -- Matrix-free hooks (defaults: no-op) ------------------------------
#
# Each constraint type can override these to inject behaviour around the
# `apply_K!` call inside `matrix_free_op`. The default is a no-op so a
# brand-new constraint type works immediately on the assembled path
# without touching the matrix-free machinery.
@inline apply_constraint_pre!(workbuf::AbstractVector{Float64},
                              x::AbstractVector{Float64},
                              ::AbstractDirichletConstraint) = workbuf

@inline apply_constraint_post!(y::AbstractVector{Float64},
                               x::AbstractVector{Float64},
                               ::AbstractDirichletConstraint) = y

# Apply the constraint's modification to a *diagonal vector* — used by
# `compute_diagonal!` to make the matrix-free Jacobi preconditioner
# match the operator returned by `matrix_free_op`. Default: leave the
# diagonal untouched.
@inline apply_constraint_diag!(d::AbstractVector{Float64},
                               ::AbstractDirichletConstraint) = d

"""
    PenaltyDirichlet(fixed_dofs)
    PenaltyDirichlet(fixed_dofs, values; penalty = 1e16)

Penalty-method Dirichlet boundary condition for the DOF-based
assembler. Stores the *fixed DOF indices* and the *prescribed values*
(zeros for the homogeneous case), plus the penalty weight.

Same struct serves both the assembled and the matrix-free path — see
`apply_constraint!` for usage.
"""
struct PenaltyDirichlet{IT<:AbstractVector{<:Integer},
                        VT<:AbstractVector{Float64}} <: AbstractDirichletConstraint
    fixed_dofs::IT
    values::VT
    penalty::Float64

    function PenaltyDirichlet(fixed_dofs::IT, values::VT, penalty::Float64) where {
            IT<:AbstractVector{<:Integer}, VT<:AbstractVector{Float64}}
        @assert length(fixed_dofs) == length(values) (
            "PenaltyDirichlet: fixed_dofs ($(length(fixed_dofs))) and " *
            "values ($(length(values))) must have the same length")
        @assert penalty > 0 "PenaltyDirichlet: penalty must be positive (got $penalty)"
        return new{IT, VT}(fixed_dofs, values, penalty)
    end
end

# Convenience: homogeneous Dirichlet (all prescribed values = 0)
PenaltyDirichlet(fixed_dofs::AbstractVector{<:Integer}; penalty::Float64 = 1e16) =
    PenaltyDirichlet(fixed_dofs, zeros(Float64, length(fixed_dofs)), penalty)

PenaltyDirichlet(fixed_dofs::AbstractVector{<:Integer},
                 values::AbstractVector{<:Real};
                 penalty::Float64 = 1e16) =
    PenaltyDirichlet(fixed_dofs, Float64.(values), penalty)


"""
    apply_constraint!(y::AbstractVector, x::AbstractVector, c::PenaltyDirichlet)

Matrix-free Dirichlet contribution: `y[d] += λ * x[d]` for every
`d` in `c.fixed_dofs`. Call after `apply_K!(y, …, x)` so the
operator effectively becomes `y = (K + λ Σ eᵈ ⊗ eᵈ) * x`.
"""
@inline function apply_constraint!(y::AbstractVector{Float64},
                                   x::AbstractVector{Float64},
                                   c::PenaltyDirichlet)
    λ = c.penalty
    @inbounds for d in c.fixed_dofs
        y[d] += λ * x[d]
    end
    return y
end

# Matrix-free post-hook: same as the explicit 3-arg apply_constraint!.
@inline apply_constraint_post!(y::AbstractVector{Float64},
                               x::AbstractVector{Float64},
                               c::PenaltyDirichlet) = apply_constraint!(y, x, c)

"""
    apply_penalty_dirichlet_post_owned!(
        y_workspace, p_owned, layout::PartitionPackedLayout, bc::PenaltyDirichlet,
    ) -> y_workspace

Add `λ * p_owned[pk]` to `y_workspace[d]` for each fixed global DOF `d` owned on this partition
(`layout.global_to_packed[d]` is in `1:layout.n_owned`). Use after [`apply_K_owned_rows!`](@ref) on
`y_workspace` when `p` is stored only in owned-packed order.

Allocation-free.
"""
function apply_penalty_dirichlet_post_owned!(
    y_workspace::AbstractVector{Float64},
    p_owned::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    bc::PenaltyDirichlet,
)
    λ = bc.penalty
    nd = layout.ndofs_global
    no = layout.n_owned
    length(y_workspace) ≥ nd ||
        throw(DimensionMismatch("y_workspace length $(length(y_workspace)) < ndofs_global $nd"))
    length(p_owned) == no ||
        throw(DimensionMismatch("p_owned length $(length(p_owned)), n_owned $no"))
    g2p = layout.global_to_packed
    @inbounds for idx in eachindex(bc.fixed_dofs)
        d = bc.fixed_dofs[idx]
        (1 ≤ d ≤ nd) || continue
        pk = g2p[d]
        (pk != 0 && pk ≤ no) || continue
        y_workspace[d] += λ * p_owned[pk]
    end
    return y_workspace
end

"""
    apply_penalty_dirichlet_post_ap_owned!(Ap_owned, packed, layout, bc::PenaltyDirichlet) -> Ap_owned

Same diagonal penalty as [`apply_penalty_dirichlet_post_owned!`](@ref), but accumulates into
`Ap_owned[pk]` instead of a global-length workspace (`pk` is the owned packed slot for fixed DOF `d`).

Allocation-free.
"""
function apply_penalty_dirichlet_post_ap_owned!(
    Ap_owned::AbstractVector{Float64},
    packed::AbstractVector{Float64},
    layout::PartitionPackedLayout,
    bc::PenaltyDirichlet,
)
    λ = bc.penalty
    no = layout.n_owned
    nd = layout.ndofs_global
    n_packed = layout.n_packed
    length(Ap_owned) == no ||
        throw(DimensionMismatch("Ap_owned length $(length(Ap_owned)), n_owned $no"))
    length(packed) ≥ n_packed ||
        throw(DimensionMismatch("packed length $(length(packed)) < n_packed $n_packed"))
    g2p = layout.global_to_packed
    @inbounds for idx in eachindex(bc.fixed_dofs)
        d = bc.fixed_dofs[idx]
        (1 ≤ d ≤ nd) || continue
        pk = g2p[d]
        (pk != 0 && pk ≤ no) || continue
        Ap_owned[pk] += λ * packed[pk]
    end
    return Ap_owned
end

# Diagonal-of-the-constrained-operator hook: K_op[d, d] = K[d, d] + λ
# for every fixed DOF, exactly mirroring `apply_constraint!(K, c)`.
@inline function apply_constraint_diag!(d::AbstractVector{Float64},
                                        c::PenaltyDirichlet)
    λ = c.penalty
    @inbounds for fdof in c.fixed_dofs
        d[fdof] += λ
    end
    return d
end

"""
    apply_constraint!(K::AbstractMatrix, c::PenaltyDirichlet)

Bake the penalty Dirichlet contribution into an assembled `K`:
`K[d,d] += λ` for every `d` in `c.fixed_dofs`. Use this on the
direct-solve path so it produces the same system as the matrix-free
operator.

Works with `Matrix{Float64}`. For `SparseMatrixCSC`, the diagonal entry
must already be structurally non-zero (it always is for a normal FEM
stiffness matrix); otherwise `setindex!` will resize the sparsity
pattern, which is correct but slow.
"""
function apply_constraint!(K::AbstractMatrix{Float64}, c::PenaltyDirichlet)
    λ = c.penalty
    @inbounds for d in c.fixed_dofs
        K[d, d] += λ
    end
    return K
end

"""
    apply_constraint!(b::AbstractVector, c::PenaltyDirichlet)

Inhomogeneous-Dirichlet contribution to the RHS: `b[d] += λ * T̂[d]` for
every `d` in `c.fixed_dofs`. For a homogeneous constraint
(`PenaltyDirichlet(fixed_dofs)`) this is a no-op.
"""
function apply_constraint!(b::AbstractVector{Float64}, c::PenaltyDirichlet)
    λ = c.penalty
    @inbounds for k in eachindex(c.fixed_dofs)
        d = c.fixed_dofs[k]
        b[d] += λ * c.values[k]
    end
    return b
end


"""
    EliminatedDirichlet(fixed_dofs)
    EliminatedDirichlet(fixed_dofs, values)

Row/column-elimination Dirichlet boundary condition. Unlike the penalty
variant, eliminating preserves the conditioning of the underlying
free-DOF block `K_ff`, so unpreconditioned Krylov solvers converge as
they would on the original (unconstrained) free system.

For an assembled matrix the `apply_constraint!(K, b, c)` overload
performs the standard three-step lift:

  1. `b ← b − K · û`             (lift the RHS by the prescribed values)
  2. `K[d, :] = K[:, d] = 0`      (zero rows and columns of fixed DOFs)
  3. `K[d, d] = 1`,  `b[d] = û_d` (identity on fixed DOFs)

For the matrix-free path the same effect is achieved by the two hooks:

  * `apply_constraint_pre!`:   `workbuf[d] = 0`              (mask `x` before `K * x`)
  * `apply_constraint_post!`:  `y[d] = x[d]`                  (identity on fixed rows)

Combining them, the resulting operator satisfies

    op(x)[free]  = K_ff x[free]      — independent of x[fixed]
    op(x)[fixed] = x[fixed]          — identity row

so a Krylov solve on `op` against the lifted RHS recovers the eliminated
solution exactly. Use it instead of `PenaltyDirichlet` when CG/GMRES
without a preconditioner needs to converge cleanly.
"""
struct EliminatedDirichlet{IT<:AbstractVector{<:Integer},
                           VT<:AbstractVector{Float64}} <: AbstractDirichletConstraint
    fixed_dofs::IT
    values::VT

    function EliminatedDirichlet(fixed_dofs::IT, values::VT) where {
            IT<:AbstractVector{<:Integer}, VT<:AbstractVector{Float64}}
        @assert length(fixed_dofs) == length(values) (
            "EliminatedDirichlet: fixed_dofs ($(length(fixed_dofs))) and " *
            "values ($(length(values))) must have the same length")
        return new{IT, VT}(fixed_dofs, values)
    end
end

EliminatedDirichlet(fixed_dofs::AbstractVector{<:Integer}) =
    EliminatedDirichlet(fixed_dofs, zeros(Float64, length(fixed_dofs)))

EliminatedDirichlet(fixed_dofs::AbstractVector{<:Integer},
                    values::AbstractVector{<:Real}) =
    EliminatedDirichlet(fixed_dofs, Float64.(values))


"""
    apply_constraint!(K::AbstractMatrix, c::EliminatedDirichlet)

Zero rows and columns of the fixed DOFs and place a `1.0` on the
diagonal. Use this only for the homogeneous case (or when the
caller has already lifted the RHS) — for the inhomogeneous case the
joint `apply_constraint!(K, b, c)` overload performs the lift first.
"""
function apply_constraint!(K::AbstractMatrix{Float64}, c::EliminatedDirichlet)
    n = size(K, 1)
    @inbounds for d in c.fixed_dofs
        for i in 1:n
            K[d, i] = 0.0
            K[i, d] = 0.0
        end
        K[d, d] = 1.0
    end
    return K
end

"""
    apply_constraint!(K::AbstractMatrix, b::AbstractVector, c::EliminatedDirichlet)

Inhomogeneous row/column elimination on the assembled system. Performs:

  1. `b ← b − K · û`         (lift)
  2. zero rows and columns of fixed DOFs in `K`
  3. `K[d, d] = 1`, `b[d] = û_d`

Mutates `K` and `b` in place; idempotent (calling twice is safe but the
second call sees a system that has already been eliminated).
"""
function apply_constraint!(K::AbstractMatrix{Float64},
                           b::AbstractVector{Float64},
                           c::EliminatedDirichlet)
    n = size(K, 1)
    fixed = c.fixed_dofs
    vals  = c.values

    @inbounds for k in eachindex(fixed)
        d  = fixed[k]
        ud = vals[k]
        if ud == 0.0
            continue
        end
        for i in 1:n
            b[i] -= K[i, d] * ud
        end
    end
    apply_constraint!(K, c)
    @inbounds for k in eachindex(fixed)
        b[fixed[k]] = vals[k]
    end
    return K, b
end

"""
    apply_constraint!(b::AbstractVector, c::EliminatedDirichlet)

Force `b[d] = û_d` on every fixed DOF. Used after `K` has already been
eliminated (e.g. when reusing an eliminated `K` across multiple RHSes).
"""
function apply_constraint!(b::AbstractVector{Float64}, c::EliminatedDirichlet)
    @inbounds for k in eachindex(c.fixed_dofs)
        b[c.fixed_dofs[k]] = c.values[k]
    end
    return b
end


# Matrix-free pre-hook: zero the fixed entries of the work copy of `x`
# *before* `apply_K!` is called, so `K · workbuf` only spans the free
# block. Equivalent to multiplying x by the projector onto the free
# subspace.
@inline function apply_constraint_pre!(workbuf::AbstractVector{Float64},
                                       x::AbstractVector{Float64},
                                       c::EliminatedDirichlet)
    @inbounds for d in c.fixed_dofs
        workbuf[d] = 0.0
    end
    return workbuf
end

# Matrix-free post-hook: identity row on the fixed DOFs. Combined with
# the pre-hook, the operator is exactly the eliminated free-DOF system
# extended with a 1×1 identity in each fixed row.
@inline function apply_constraint_post!(y::AbstractVector{Float64},
                                        x::AbstractVector{Float64},
                                        c::EliminatedDirichlet)
    @inbounds for k in eachindex(c.fixed_dofs)
        d = c.fixed_dofs[k]
        y[d] = x[d]
    end
    return y
end

# Diagonal of the eliminated operator: 1.0 on every fixed DOF (identity
# row), unchanged everywhere else. Mirrors `apply_constraint!(K, c)`.
@inline function apply_constraint_diag!(d::AbstractVector{Float64},
                                        c::EliminatedDirichlet)
    @inbounds for fdof in c.fixed_dofs
        d[fdof] = 1.0
    end
    return d
end


