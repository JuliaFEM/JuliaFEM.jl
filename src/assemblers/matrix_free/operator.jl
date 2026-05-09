# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractMatrixFreeOperator

Common supertype for typed matrix-free linear operators built on the
DOF-based assembler. Subtypes wrap a `DOFBasedCOOCache`, the kernel
they evaluate (`apply_K!` for the stiffness, `apply_M!` for the mass)
and any constraint hooks that should be folded into every mat-vec.

The contract is:

* `LinearAlgebra.mul!(y, op, x)` — in-place mat-vec.
* `Base.size(op)` / `Base.size(op, d)` / `Base.eltype(op)`.
* `op(y, x)` — operator-as-closure call style, kept so the type plugs
  into `LinearOperators.LinearOperator(Float64, n, n, true, true, op)`
  and any other Krylov interface that takes a callable.
* `op * x` — allocating mat-vec.
"""
abstract type AbstractMatrixFreeOperator end

# Default closure-style call: forwards to mul!.
@inline (op::AbstractMatrixFreeOperator)(y::AbstractVector{Float64},
                                         x::AbstractVector{Float64}) =
    LinearAlgebra.mul!(y, op, x)

Base.eltype(::AbstractMatrixFreeOperator) = Float64

function Base.:*(op::AbstractMatrixFreeOperator, x::AbstractVector{Float64})
    y = similar(x, Float64, size(op, 1))
    LinearAlgebra.mul!(y, op, x)
    return y
end

# ---------------------------------------------------------------------------
# Stiffness operator: K (with optional Dirichlet + MPC constraints folded
# into each mat-vec).
# ---------------------------------------------------------------------------

"""
    MatrixFreeOperator{C, A, K, M, D, P, L}

Typed matrix-free stiffness operator.

Encapsulates the `(cache, asm, kernel, mesh)` four-tuple required by
`apply_K!`, an optional `Dirichlet` constraint and an optional `MPC`
constraint, a length-`ndofs` work buffer that absorbs any non-
`Vector{Float64}` input column, and a second buffer for the 5-argument
`mul!` interface. Each `mul!` performs

  1. `prepare_multiply_workspace!(workbuf, x, multiply_layout)`
  2. `apply_constraint_pre!(workbuf, x, dirichlet)`   (if dirichlet)
  3. `apply_K!(y, cache, asm, kernel, mesh, workbuf)`
  4. `apply_constraint_post!(y, x, dirichlet)`        (if dirichlet)
  5. `apply_constraint_post!(y, x, mpc)`              (if mpc)

so the constrained operator
`(K + λ·diag(eᵈ)) x`,  `K_ff x[free] ⊕ x[fixed]`, etc. is materialised
without ever forming `K`.

# Examples

```julia
op = MatrixFreeOperator(cache, asm, kernel, mesh; dirichlet = bc)
mul!(y, op, x)                       # in-place
y2 = op * x                          # allocating
linop = LinearOperators.LinearOperator(Float64, size(op, 1), size(op, 2),
                                       true, true, op)   # Krylov plug-in
```
"""
struct MatrixFreeOperator{C<:DOFBasedCOOCache,
                          A<:DOFBasedCOOAssembler,
                          K<:AbstractKernel,
                          M<:AbstractMesh,
                          D, P,
                          L<:AbstractMultiplyGhostLayout} <: AbstractMatrixFreeOperator
    cache::C
    asm::A
    kernel::K
    mesh::M
    dirichlet::D
    mpc::P
    workbuf::Vector{Float64}
    mulbuf::Vector{Float64}
    multiply_layout::L
end

MatrixFreeOperator(cache::DOFBasedCOOCache,
                   asm::DOFBasedCOOAssembler,
                   kernel::AbstractKernel,
                   mesh::AbstractMesh,
                   dirichlet,
                   mpc,
                   workbuf::Vector{Float64}) =
    MatrixFreeOperator(cache, asm, kernel, mesh, dirichlet, mpc, workbuf, similar(workbuf))

MatrixFreeOperator(cache::DOFBasedCOOCache,
                   asm::DOFBasedCOOAssembler,
                   kernel::AbstractKernel,
                   mesh::AbstractMesh,
                   dirichlet,
                   mpc,
                   workbuf::Vector{Float64},
                   mulbuf::Vector{Float64}) =
    MatrixFreeOperator(cache, asm, kernel, mesh, dirichlet, mpc, workbuf, mulbuf, LocalMultiplyLayout())

function MatrixFreeOperator(cache::DOFBasedCOOCache,
                            asm::DOFBasedCOOAssembler,
                            kernel::AbstractKernel,
                            mesh::AbstractMesh;
                            dirichlet = nothing,
                            mpc = nothing,
                            multiply_layout::L = LocalMultiplyLayout()) where L <: AbstractMultiplyGhostLayout
    workbuf = zeros(Float64, cache.ndofs)
    mulbuf = similar(workbuf)
    return MatrixFreeOperator(cache, asm, kernel, mesh, dirichlet, mpc, workbuf, mulbuf, multiply_layout)
end

@inline Base.size(op::MatrixFreeOperator) = (op.cache.ndofs, op.cache.ndofs)
@inline Base.size(op::MatrixFreeOperator, d::Integer) =
    (d == 1 || d == 2) ? op.cache.ndofs : 1

LinearAlgebra.issymmetric(::MatrixFreeOperator) = true
LinearAlgebra.ishermitian(::MatrixFreeOperator) = true

# Whether `K` is SPD is a property of the kernel; mixed / saddle-point
# kernels override `operator_is_posdef` to `false` so Krylov stacks (CG, …)
# do not pick the SPD branch.
@inline LinearAlgebra.isposdef(op::MatrixFreeOperator) = operator_is_posdef(op.kernel)

function LinearAlgebra.mul!(y::AbstractVector{Float64},
                            op::MatrixFreeOperator,
                            x::AbstractVector{Float64})
    workbuf = op.workbuf
    prepare_multiply_workspace!(workbuf, x, op.multiply_layout)
    if op.dirichlet !== nothing
        apply_constraint_pre!(workbuf, x, op.dirichlet)
    end
    apply_K!(y, op.cache, op.asm, op.kernel, op.mesh, workbuf)
    if op.dirichlet !== nothing
        apply_constraint_post!(y, x, op.dirichlet)
    end
    if op.mpc !== nothing
        apply_constraint_post!(y, x, op.mpc)
    end
    return y
end

# 5-arg `mul!(y, op, x, α, β)` for completeness — the
# `LinearAlgebra.mul!(C, A, B, α, β)` interface that some solvers
# (and `*` fallbacks) expect.
function LinearAlgebra.mul!(y::AbstractVector{Float64},
                            op::MatrixFreeOperator,
                            x::AbstractVector{Float64},
                            α::Number, β::Number)
    n = size(op, 1)
    if iszero(β)
        fill!(y, 0)
    elseif !isone(β)
        @inbounds @simd for i in 1:n
            y[i] *= β
        end
    end
    if !iszero(α)
        scratch = op.mulbuf
        LinearAlgebra.mul!(scratch, op, x)
        @inbounds @simd for i in 1:n
            y[i] += α * scratch[i]
        end
    end
    return y
end

"""
    matrix_free_op(cache, asm, kernel, mesh; dirichlet = nothing,
                                              mpc = nothing)
        -> MatrixFreeOperator

Convenience wrapper that builds a `MatrixFreeOperator` for `K` (with
the optional `dirichlet` and `mpc` constraints folded into every
mat-vec). Returns the typed operator directly; the operator is callable
(`op(y, x)` does an in-place mat-vec) so existing call sites that
treat `matrix_free_op(...)` as a closure keep working.

The constraint type controls the constrained operator:

  * `PenaltyDirichlet`     →  `op(x) = K x + λ · diag(eᵈ) x`
  * `EliminatedDirichlet`  →  `op(x)[free]  = K_ff x[free]`,
                              `op(x)[fixed] = x[fixed]`   (identity)

# Example

```julia
using LinearOperators, IterativeSolvers, JuliaFEM

c       = EliminatedDirichlet(fixed_dofs, û)
op      = matrix_free_op(cache, asm, kernel, mesh; dirichlet = c)
linop   = LinearOperator(Float64, size(op, 1), size(op, 2), true, true, op)

T_mf = zeros(cache.ndofs)
cg!(T_mf, linop, b_lifted)
```

For new code prefer constructing `MatrixFreeOperator` directly; this
helper is kept for backward compatibility and as a single-call factory
that mirrors the kwargs the constraint and load APIs expect.
"""
@inline function matrix_free_op(cache::DOFBasedCOOCache,
                                asm::DOFBasedCOOAssembler,
                                kernel::AbstractKernel,
                                mesh::AbstractMesh;
                                dirichlet::Union{AbstractDirichletConstraint,Nothing} = nothing,
                                mpc = nothing,
                                multiply_layout::AbstractMultiplyGhostLayout = LocalMultiplyLayout())
    return MatrixFreeOperator(cache, asm, kernel, mesh;
                              dirichlet = dirichlet, mpc = mpc,
                              multiply_layout = multiply_layout)
end

# ---------------------------------------------------------------------------
# Mass operator: M (no constraint hooks; mass is already symmetric and
# has no rigid-body / fixed-DOF semantics in our matrix-free path).
# ---------------------------------------------------------------------------

"""
    MatrixFreeMassOperator{C, A, K, M}

Typed matrix-free mass operator wrapping the `(cache, asm, kernel,
mesh)` four-tuple plus a work buffer; each `mul!(y, op, x)` evaluates
`apply_M!(y, cache, asm, kernel, mesh, workbuf)`.

Used by `solve_eigenproblem` so the lowest-eigenpair routine no longer
needs ad-hoc `(y, x) -> apply_M!(...)` closures. A second buffer keeps
the 5-argument `mul!` interface allocation-free.
"""
struct MatrixFreeMassOperator{C<:DOFBasedCOOCache,
                              A<:DOFBasedCOOAssembler,
                              K<:AbstractKernel,
                              M<:AbstractMesh} <: AbstractMatrixFreeOperator
    cache::C
    asm::A
    kernel::K
    mesh::M
    workbuf::Vector{Float64}
    mulbuf::Vector{Float64}
end

MatrixFreeMassOperator(cache::DOFBasedCOOCache,
                       asm::DOFBasedCOOAssembler,
                       kernel::AbstractKernel,
                       mesh::AbstractMesh,
                       workbuf::Vector{Float64}) =
    MatrixFreeMassOperator(cache, asm, kernel, mesh, workbuf, similar(workbuf))

function MatrixFreeMassOperator(cache::DOFBasedCOOCache,
                                asm::DOFBasedCOOAssembler,
                                kernel::AbstractKernel,
                                mesh::AbstractMesh)
    workbuf = zeros(Float64, cache.ndofs)
    mulbuf = similar(workbuf)
    return MatrixFreeMassOperator(cache, asm, kernel, mesh, workbuf, mulbuf)
end

@inline Base.size(op::MatrixFreeMassOperator) = (op.cache.ndofs, op.cache.ndofs)
@inline Base.size(op::MatrixFreeMassOperator, d::Integer) =
    (d == 1 || d == 2) ? op.cache.ndofs : 1

LinearAlgebra.issymmetric(::MatrixFreeMassOperator) = true
LinearAlgebra.ishermitian(::MatrixFreeMassOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{Float64},
                            op::MatrixFreeMassOperator,
                            x::AbstractVector{Float64})
    workbuf = op.workbuf
    @inbounds @simd for i in eachindex(workbuf)
        workbuf[i] = x[i]
    end
    apply_M!(y, op.cache, op.asm, op.kernel, op.mesh, workbuf)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{Float64},
                            op::MatrixFreeMassOperator,
                            x::AbstractVector{Float64},
                            α::Number, β::Number)
    n = size(op, 1)
    if iszero(β)
        fill!(y, 0.0)
    elseif !isone(β)
        @inbounds @simd for i in 1:n
            y[i] *= β
        end
    end
    if !iszero(α)
        scratch = op.mulbuf
        LinearAlgebra.mul!(scratch, op, x)
        @inbounds @simd for i in 1:n
            y[i] += α * scratch[i]
        end
    end
    return y
end
