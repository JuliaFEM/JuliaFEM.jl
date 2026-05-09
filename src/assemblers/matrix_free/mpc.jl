# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Linear multipoint constraints (MPC) for the DOF-based assembler.

A *linear MPC* prescribes a slave DOF as an affine combination of one
or more master DOFs:

    u_s = Σ_k c_k · u_{m_k} + g            (one constraint)

equivalently

    R = u_s − Σ_k c_k · u_{m_k} − g = 0    (residual form)

This module follows the same hook protocol as `dirichlet.jl` so MPCs
serve both the assembled and the matrix-free solve paths from a
single declarative description:

```julia
mpc = LinearMPC([(slave, master_dofs, coeffs, offset), …]; penalty)

apply_constraint!(K, mpc)          # bake λ · Cᵀ C contribution into K
apply_constraint!(b, mpc)          # bake λ · Cᵀ g contribution into b
apply_constraint_post!(y, x, mpc)  # matrix-free: y += λ Cᵀ C · x
apply_constraint_diag!(d, mpc)     # diagonal hook for Jacobi precond
```

Two enforcement strategies are supported, mirroring Dirichlet:

* `LinearMPC`        — penalty enforcement.  Adds `λ · Σ_k Rᵀ R / 2` to
                       the energy.  Cheap, drop-in, but conditioning of
                       the resulting system grows with `λ`.

`matrix_free_op(...; mpc = …)` composes naturally with
`dirichlet = …`, so both can be active in the same solve.

# Assembled-K contribution (per constraint)

Given `R = u_s − Σ_k c_k u_{m_k} − g`,
`E_pen = (λ/2) Rᵀ R`, and the Hessian `K_pen = λ · Cᵀ C`:

```
K[s, s]      += λ
K[s, m_k]    += −λ · c_k
K[m_k, s]    += −λ · c_k
K[m_k, m_l]  += λ · c_k · c_l
b[s]         += λ · g
b[m_k]       += −λ · c_k · g
```

# Matrix-free contribution (per constraint)

Compute the residual `R0(x) = x[s] − Σ_k c_k · x[m_k]` once, then:

```
y[s]      += λ · R0
y[m_k]    -= λ · c_k · R0
```

These two views agree, so the assembled and matrix-free paths produce
identical operators.
"""

abstract type AbstractMultipointConstraint end

"""
    LinearMPC(constraints; penalty = 1e10)

Penalty-enforced linear multipoint constraint set.  `constraints` is a
vector of `(slave, masters, coeffs, offset)` tuples — one per
constraint:

  * `slave`   — the constrained DOF (Int)
  * `masters` — vector of master DOFs (Vector{Int})
  * `coeffs`  — vector of coefficients of the same length as `masters`
                (Vector{Float64})
  * `offset`  — additive offset `g` in `R = u_s − Σ c_k u_{m_k} − g`
                (Float64)

The data are stored in flat CSR arrays for zero-allocation matrix-free
hooks; the `(constraints; penalty)` constructor takes care of packing.

`penalty = λ` defaults to `1e10`. Larger `λ` enforces the constraint
more strictly at the cost of conditioning. Pair with `JacobiPreconditioner`
or `ICholPreconditioner` for matrix-free CG.
"""
struct LinearMPC <: AbstractMultipointConstraint
    slaves::Vector{Int}              # length = N (number of constraints)
    offsets::Vector{Float64}         # length = N
    master_offsets::Vector{Int}      # length = N + 1 (CSR row pointers)
    master_dofs::Vector{Int}         # length = nnz
    master_coeffs::Vector{Float64}   # length = nnz
    penalty::Float64

    function LinearMPC(slaves::Vector{Int},
                       offsets::Vector{Float64},
                       master_offsets::Vector{Int},
                       master_dofs::Vector{Int},
                       master_coeffs::Vector{Float64},
                       penalty::Float64)
        N = length(slaves)
        @assert length(offsets) == N (
            "LinearMPC: slaves ($N) and offsets ($(length(offsets))) must agree")
        @assert length(master_offsets) == N + 1 (
            "LinearMPC: master_offsets must have length $(N + 1) " *
            "(got $(length(master_offsets)))")
        @assert master_offsets[1] == 1 "LinearMPC: master_offsets[1] must be 1"
        @assert master_offsets[end] == length(master_dofs) + 1 (
            "LinearMPC: master_offsets[end] = $(master_offsets[end]) but " *
            "master_dofs has $(length(master_dofs)) entries " *
            "(expected $(length(master_dofs) + 1))")
        @assert length(master_coeffs) == length(master_dofs) (
            "LinearMPC: master_dofs ($(length(master_dofs))) and " *
            "master_coeffs ($(length(master_coeffs))) must have the same length")
        @assert penalty > 0 "LinearMPC: penalty must be positive (got $penalty)"
        return new(slaves, offsets, master_offsets,
                   master_dofs, master_coeffs, penalty)
    end
end

# Convenience constructor: list of (slave, masters, coeffs, offset) tuples.
function LinearMPC(constraints::AbstractVector; penalty::Real = 1e10)
    N = length(constraints)
    @assert N > 0 "LinearMPC: at least one constraint required"

    slaves         = Vector{Int}(undef, N)
    offsets        = Vector{Float64}(undef, N)
    master_offsets = Vector{Int}(undef, N + 1)
    master_offsets[1] = 1

    # First pass: lengths.
    total_nnz = 0
    @inbounds for k in 1:N
        c = constraints[k]
        masters = c[2]
        coeffs  = c[3]
        @assert length(masters) == length(coeffs) (
            "LinearMPC[$k]: masters ($(length(masters))) and coeffs " *
            "($(length(coeffs))) must have the same length")
        total_nnz += length(masters)
        master_offsets[k + 1] = total_nnz + 1
    end

    master_dofs   = Vector{Int}(undef, total_nnz)
    master_coeffs = Vector{Float64}(undef, total_nnz)

    # Second pass: pack.
    @inbounds for k in 1:N
        c = constraints[k]
        slaves[k]  = Int(c[1])
        masters    = c[2]
        coeffs     = c[3]
        offsets[k] = length(c) >= 4 ? Float64(c[4]) : 0.0
        base = master_offsets[k] - 1
        for j in eachindex(masters)
            master_dofs[base + j]   = Int(masters[j])
            master_coeffs[base + j] = Float64(coeffs[j])
        end
    end

    return LinearMPC(slaves, offsets, master_offsets,
                     master_dofs, master_coeffs, Float64(penalty))
end

"Number of constraints in `c`."
@inline n_constraints(c::LinearMPC) = length(c.slaves)


# ---------------------------------------------------------------------------
# Matrix-free hooks (default: no-op; LinearMPC overrides post / diag)
# ---------------------------------------------------------------------------

@inline apply_constraint_pre!(workbuf::AbstractVector{Float64},
                              x::AbstractVector{Float64},
                              ::AbstractMultipointConstraint) = workbuf

@inline apply_constraint_post!(y::AbstractVector{Float64},
                               x::AbstractVector{Float64},
                               ::AbstractMultipointConstraint) = y

@inline apply_constraint_diag!(d::AbstractVector{Float64},
                               ::AbstractMultipointConstraint) = d


# ---------------------------------------------------------------------------
# LinearMPC: matrix-free hooks
# ---------------------------------------------------------------------------

"""
    apply_constraint_post!(y, x, mpc::LinearMPC)

Add the penalty-MPC contribution `y += λ · Cᵀ C · x` to `y`. The
constraint residual is computed once per constraint and scattered back:
`y[s] += λ R0`, `y[m_k] -= λ c_k R0`.
"""
@inline function apply_constraint_post!(y::AbstractVector{Float64},
                                        x::AbstractVector{Float64},
                                        c::LinearMPC)
    λ = c.penalty
    N = n_constraints(c)
    @inbounds for k in 1:N
        s   = c.slaves[k]
        lo  = c.master_offsets[k]
        hi  = c.master_offsets[k + 1] - 1

        R0 = x[s]
        for jp in lo:hi
            R0 -= c.master_coeffs[jp] * x[c.master_dofs[jp]]
        end

        λR0  = λ * R0
        y[s] += λR0
        for jp in lo:hi
            y[c.master_dofs[jp]] -= c.master_coeffs[jp] * λR0
        end
    end
    return y
end

"""
    apply_constraint!(y, x, mpc::LinearMPC)

Standalone form of the matrix-free MPC contribution (alias of
`apply_constraint_post!`). Provided so the constraint can be used outside
the `matrix_free_op` machinery (e.g. when stacking custom operators).
"""
@inline function apply_constraint!(y::AbstractVector{Float64},
                                   x::AbstractVector{Float64},
                                   c::LinearMPC)
    return apply_constraint_post!(y, x, c)
end

"""
    apply_constraint_diag!(d, mpc::LinearMPC)

Diagonal-of-the-constrained-operator hook. Adds the diagonal entries
of `λ · Cᵀ C`:

    d[s]    += λ
    d[m_k]  += λ · c_k²

Used by `JacobiPreconditioner(...; mpc)` so the diagonal preconditioner
matches the operator returned by `matrix_free_op(...; mpc)`.
"""
@inline function apply_constraint_diag!(d::AbstractVector{Float64},
                                        c::LinearMPC)
    λ = c.penalty
    N = n_constraints(c)
    @inbounds for k in 1:N
        s   = c.slaves[k]
        d[s] += λ
        lo   = c.master_offsets[k]
        hi   = c.master_offsets[k + 1] - 1
        for jp in lo:hi
            ck = c.master_coeffs[jp]
            d[c.master_dofs[jp]] += λ * ck * ck
        end
    end
    return d
end


# ---------------------------------------------------------------------------
# Assembled-K hooks
# ---------------------------------------------------------------------------

"""
    apply_constraint!(K::AbstractMatrix, mpc::LinearMPC)

Bake the penalty-MPC contribution `λ · Cᵀ C` into an assembled `K` so the
direct solve sees the same constrained operator as the matrix-free path.
"""
function apply_constraint!(K::AbstractMatrix{Float64}, c::LinearMPC)
    λ = c.penalty
    N = n_constraints(c)
    @inbounds for k in 1:N
        s  = c.slaves[k]
        lo = c.master_offsets[k]
        hi = c.master_offsets[k + 1] - 1

        K[s, s] += λ
        for jp in lo:hi
            mk = c.master_dofs[jp]
            ck = c.master_coeffs[jp]
            K[s, mk] -= λ * ck
            K[mk, s] -= λ * ck
            for jq in lo:hi
                ml = c.master_dofs[jq]
                cl = c.master_coeffs[jq]
                K[mk, ml] += λ * ck * cl
            end
        end
    end
    return K
end

"""
    apply_constraint!(b::AbstractVector, mpc::LinearMPC)

RHS contribution of an inhomogeneous penalty-MPC: `b[s] += λ g_k` and
`b[m_k] -= λ c_k g_k`. For homogeneous constraints (`g = 0` everywhere)
this is a no-op.
"""
function apply_constraint!(b::AbstractVector{Float64}, c::LinearMPC)
    λ = c.penalty
    N = n_constraints(c)
    @inbounds for k in 1:N
        g = c.offsets[k]
        if g == 0.0
            continue
        end
        s = c.slaves[k]
        b[s] += λ * g
        lo = c.master_offsets[k]
        hi = c.master_offsets[k + 1] - 1
        for jp in lo:hi
            b[c.master_dofs[jp]] -= λ * c.master_coeffs[jp] * g
        end
    end
    return b
end


# `matrix_free_op(...; dirichlet, mpc)` lives in `operator.jl` and is
# the single factory for the typed `MatrixFreeOperator` (which folds
# both Dirichlet and MPC hooks into every mat-vec). Composition order:
#
#   1. workbuf .= x
#   2. apply_constraint_pre!(workbuf, x, dirichlet)
#   3. apply_K!(y, …, workbuf)
#   4. apply_constraint_post!(y, x, dirichlet)
#   5. apply_constraint_post!(y, x, mpc)
#
# Dirichlet acts first (zeros fixed DOFs in the input, identity-row in
# the output) and the MPC then *adds* its `λ · Cᵀ C · x` contribution.
