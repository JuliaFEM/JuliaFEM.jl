# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
    AbstractMatrixFreeOperator

Common supertype for typed matrix-free linear operators built on the
DOF-based assembler. Subtypes wrap a `DOFBasedCOOCache` (volume kernel
from `cache.kernel_column`), call [`apply_K!`](@ref) or [`apply_M!`](@ref),
and may attach constraint hooks folded into every mat-vec.

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
    MatrixFreeOperator{C, A, M, D, P, L}

Typed matrix-free stiffness operator.

Encapsulates `cache`, assembler tag, and `mesh` for [`apply_K!`](@ref).
The volume kernel is always read from `cache.kernel_column` (never from a
separate redundant field). Optional `Dirichlet` and `MPC` constraints,
length-`ndofs` work buffers for the multiply layout and the 5-argument
`mul!` interface, and nonlinear Pass~1 keywords round out the type.

Each `mul!` performs

  1. `prepare_multiply_workspace!(workbuf, x, multiply_layout)`
  2. `apply_constraint_pre!(workbuf, x, dirichlet)`   (if dirichlet)
  3. `apply_K!(y, cache, asm, mesh, workbuf; configuration, …)`
  4. `apply_constraint_post!(y, x, dirichlet)`   (if dirichlet)
  5. `apply_constraint_post!(y, x, mpc)`         (if mpc)

so the constrained operator
`(K + λ·diag(eᵈ)) x`,  `K_ff x[free] ⊕ x[fixed]`, etc. is materialised
without ever forming `K`.

Optional fields `configuration`, `global_material_cache`, and `Δt` are
forwarded to [`apply_K!`](@ref) (Pass~1 material updates). Defaults match
the linear reference configuration.

# Examples

```julia
op = MatrixFreeOperator(cache, asm, mesh; dirichlet = bc)
mul!(y, op, x)                       # in-place
y2 = op * x                          # allocating
linop = LinearOperators.LinearOperator(Float64, size(op, 1), size(op, 2),
                                       true, true, op)   # Krylov plug-in
```
"""
struct MatrixFreeOperator{C<:DOFBasedCOOCache,
                          A<:DOFBasedCOOAssembler,
                          M<:AbstractMesh,
                          D, P,
                          L<:AbstractMultiplyGhostLayout} <: AbstractMatrixFreeOperator
    cache::C
    asm::A
    mesh::M
    dirichlet::D
    mpc::P
    workbuf::Vector{Float64}
    mulbuf::Vector{Float64}
    multiply_layout::L
    configuration::Union{Nothing,AbstractVector{Float64}}
    global_material_cache::Union{Nothing,GlobalMaterialCache}
    Δt::Float64
end

MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    dirichlet,
    mpc,
    workbuf::Vector{Float64},
) =
    MatrixFreeOperator(
        cache, asm, mesh, dirichlet, mpc, workbuf, similar(workbuf),
        LocalMultiplyLayout(), nothing, nothing, 0.0,
    )

MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    dirichlet,
    mpc,
    workbuf::Vector{Float64},
    mulbuf::Vector{Float64},
) =
    MatrixFreeOperator(
        cache, asm, mesh, dirichlet, mpc, workbuf, mulbuf,
        LocalMultiplyLayout(), nothing, nothing, 0.0,
    )

function MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    dirichlet = nothing,
    mpc = nothing,
    multiply_layout::L = LocalMultiplyLayout(),
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
) where {L <: AbstractMultiplyGhostLayout}
    workbuf = zeros(Float64, cache.ndofs)
    mulbuf = similar(workbuf)
    return MatrixFreeOperator(
        cache, asm, mesh, dirichlet, mpc, workbuf, mulbuf, multiply_layout,
        configuration, global_material_cache, Float64(Δt),
    )
end

@inline function MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    kwargs...,
)
    _depwarn_redundant_kernel_arg!(:MatrixFreeOperator)
    return MatrixFreeOperator(cache, asm, mesh; kwargs...)
end

MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    dirichlet,
    mpc,
    workbuf::Vector{Float64},
) =
    begin
        _depwarn_redundant_kernel_arg!(:MatrixFreeOperator)
        MatrixFreeOperator(cache, asm, mesh, dirichlet, mpc, workbuf)
    end

MatrixFreeOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    dirichlet,
    mpc,
    workbuf::Vector{Float64},
    mulbuf::Vector{Float64},
) =
    begin
        _depwarn_redundant_kernel_arg!(:MatrixFreeOperator)
        MatrixFreeOperator(cache, asm, mesh, dirichlet, mpc, workbuf, mulbuf)
    end

@inline Base.size(op::MatrixFreeOperator) = (op.cache.ndofs, op.cache.ndofs)
@inline Base.size(op::MatrixFreeOperator, d::Integer) =
    (d == 1 || d == 2) ? op.cache.ndofs : 1

LinearAlgebra.issymmetric(::MatrixFreeOperator) = true
LinearAlgebra.ishermitian(::MatrixFreeOperator) = true

# Whether `K` is SPD is a property of the kernel; mixed / saddle-point
# kernels override `operator_is_posdef` to `false` so Krylov stacks (CG, …)
# do not pick the SPD branch.
@inline LinearAlgebra.isposdef(op::MatrixFreeOperator) =
    operator_is_posdef(prototype_kernel(op.cache.kernel_column))

function LinearAlgebra.mul!(y::AbstractVector{Float64},
                            op::MatrixFreeOperator,
                            x::AbstractVector{Float64})
    workbuf = op.workbuf
    prepare_multiply_workspace!(workbuf, x, op.multiply_layout)
    if op.dirichlet !== nothing
        apply_constraint_pre!(workbuf, x, op.dirichlet)
    end
    # Avoid keyword `apply_K!` on the default linear path so `mul!` stays
    # allocation-free after warmup (see `test_matrix_free_operator.jl`).
    if op.configuration === nothing && op.global_material_cache === nothing && iszero(op.Δt)
        apply_K!(y, op.cache, op.asm, op.mesh, workbuf)
    else
        apply_K!(
            y, op.cache, op.asm, op.mesh, workbuf;
            configuration = op.configuration,
            global_material_cache = op.global_material_cache,
            Δt = op.Δt,
        )
    end
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
    matrix_free_op(cache, asm, mesh; dirichlet = nothing, mpc = nothing, kwargs...)
    matrix_free_op(cache, asm, kernel, mesh; …)

Convenience wrapper that builds a `MatrixFreeOperator` for `K` (with
the optional `dirichlet` and `mpc` constraints folded into every
mat-vec). Returns the typed operator directly; the operator is callable
(`op(y, x)` does an in-place mat-vec) so existing call sites that
treat `matrix_free_op(...)` as a closure keep working.

The three-argument form `matrix_free_op(cache, asm, mesh; …)` is primary.
The four-argument form with a trailing `kernel` argument ignores that
kernel (backward compatibility; emits `Base.depwarn` once per session); the
kernel always comes from `cache.kernel_column`.

Additional keywords `configuration`, `global_material_cache`, and `Δt`
are forwarded to [`MatrixFreeOperator`](@ref) and then into each
[`apply_K!`](@ref) during `mul!`.

The constraint type controls the constrained operator:

  * `PenaltyDirichlet`     →  `op(x) = K x + λ · diag(eᵈ) x`
  * `EliminatedDirichlet`  →  `op(x)[free]  = K_ff x[free]`,
                              `op(x)[fixed] = x[fixed]`   (identity)

# Example

```julia
using LinearOperators, IterativeSolvers, JuliaFEM

c       = EliminatedDirichlet(fixed_dofs, û)
op      = matrix_free_op(cache, asm, mesh; dirichlet = c)
linop   = LinearOperator(Float64, size(op, 1), size(op, 2), true, true, op)

T_mf = zeros(cache.ndofs)
cg!(T_mf, linop, b_lifted)
```

For new code prefer constructing `MatrixFreeOperator` directly; this
helper is kept for backward compatibility and as a single-call factory
that mirrors the kwargs the constraint and load APIs expect.
"""
@inline function matrix_free_op(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    dirichlet::Union{AbstractDirichletConstraint,Nothing} = nothing,
    mpc = nothing,
    multiply_layout::AbstractMultiplyGhostLayout = LocalMultiplyLayout(),
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    return MatrixFreeOperator(
        cache, asm, mesh;
        dirichlet = dirichlet,
        mpc = mpc,
        multiply_layout = multiply_layout,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

@inline function matrix_free_op(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    dirichlet::Union{AbstractDirichletConstraint,Nothing} = nothing,
    mpc = nothing,
    multiply_layout::AbstractMultiplyGhostLayout = LocalMultiplyLayout(),
    configuration::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    _depwarn_redundant_kernel_arg!(:matrix_free_op)
    return matrix_free_op(
        cache, asm, mesh;
        dirichlet = dirichlet,
        mpc = mpc,
        multiply_layout = multiply_layout,
        configuration = configuration,
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

# ---------------------------------------------------------------------------
# Mass operator: M (no constraint hooks; mass is already symmetric and
# has no rigid-body / fixed-DOF semantics in our matrix-free path).
# ---------------------------------------------------------------------------

"""
    MatrixFreeMassOperator{C, A, M}

Typed matrix-free mass operator wrapping `cache`, assembler tag, and
`mesh`; each `mul!(y, op, x)` evaluates `apply_M!(y, cache, asm, mesh, workbuf)`.
The volume kernel is read from `cache.kernel_column`.

Used by `solve_eigenproblem` so the lowest-eigenpair routine no longer
needs ad-hoc `(y, x) -> apply_M!(...)` closures. A second buffer keeps
the 5-argument `mul!` interface allocation-free.
"""
struct MatrixFreeMassOperator{C<:DOFBasedCOOCache,
                              A<:DOFBasedCOOAssembler,
                              M<:AbstractMesh} <: AbstractMatrixFreeOperator
    cache::C
    asm::A
    mesh::M
    workbuf::Vector{Float64}
    mulbuf::Vector{Float64}
end

MatrixFreeMassOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
    workbuf::Vector{Float64},
) =
    MatrixFreeMassOperator(cache, asm, mesh, workbuf, similar(workbuf))

function MatrixFreeMassOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh,
)
    workbuf = zeros(Float64, cache.ndofs)
    mulbuf = similar(workbuf)
    return MatrixFreeMassOperator(cache, asm, mesh, workbuf, mulbuf)
end

@inline MatrixFreeMassOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
) =
    begin
        _depwarn_redundant_kernel_arg!(:MatrixFreeMassOperator)
        MatrixFreeMassOperator(cache, asm, mesh)
    end

MatrixFreeMassOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    workbuf::Vector{Float64},
) =
    begin
        _depwarn_redundant_kernel_arg!(:MatrixFreeMassOperator)
        MatrixFreeMassOperator(cache, asm, mesh, workbuf)
    end

MatrixFreeMassOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh,
    workbuf::Vector{Float64},
    mulbuf::Vector{Float64},
) =
    begin
        _depwarn_redundant_kernel_arg!(:MatrixFreeMassOperator)
        MatrixFreeMassOperator(cache, asm, mesh, workbuf, mulbuf)
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
    apply_M!(y, op.cache, op.asm, op.mesh, workbuf)
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

# ---------------------------------------------------------------------------
# Mass operator on `DOFBasedCOOCacheKA` (KernelAbstractions path).
# ---------------------------------------------------------------------------

"""
    MatrixFreeMassOperatorKA{F, K, CKA}

Matrix-free mass matvec on a [`DOFBasedCOOCacheKA`](@ref): each
`mul!(y, op, x)` calls [`apply_M!`](@ref)`(y, op.cache_ka, op.kernel, x)`
after copying `x` into an internal scratch (so `y` and `x` may alias the
same storage as long as `y !== x` is not required by callers; the copy matches
[`MatrixFreeMassOperator`](@ref) behaviour).

Precision `F` matches `eltype(cache_ka.detJ_w_batch)` (typically `Float64`
or `Float32` after [`to_float32`](@ref)). Not a subtype of
[`AbstractMatrixFreeOperator`](@ref) because `eltype` is parametric and GPU
backends use `KernelAbstractions.get_backend(y)`.

Use after Pass~1 on the CPU cache and [`sync_from_cpu!`](@ref); optionally
[`Adapt.adapt`](@ref)`(MetalBackend(), cache_ka)`, `Adapt.adapt(CUDABackend(), cache_ka)`,
`Adapt.adapt(ROCBackend(), cache_ka)`, or `Adapt.adapt(oneAPIBackend(), cache_ka)` when the
corresponding GPU package is loaded.

# Example

```julia
op_m = MatrixFreeMassOperatorKA(cache_ka, prototype_kernel(cpu_cache.kernel_column))
mul!(y, op_m, x)   # y, x same precision as cache_ka
```
"""
struct MatrixFreeMassOperatorKA{F<:AbstractFloat,
                                 K<:AbstractKernel,
                                 CKA<:DOFBasedCOOCacheKA,
                                 B<:AbstractVector{F}}
    cache_ka::CKA
    kernel::K
    workbuf::B
    mulbuf::B
end

function MatrixFreeMassOperatorKA(cache_ka::DOFBasedCOOCacheKA, kernel::K) where {K<:AbstractKernel}
    F = eltype(cache_ka.detJ_w_batch)
    n = length(cache_ka.dof_counts)
    be = KernelAbstractions.get_backend(cache_ka.detJ_w_batch)
    z = Adapt.adapt(be, zeros(F, n))
    return MatrixFreeMassOperatorKA{F,K,typeof(cache_ka),typeof(z)}(
        cache_ka,
        kernel,
        z,
        similar(z),
    )
end

@inline function matrix_free_mass_op_ka(cache_ka::DOFBasedCOOCacheKA, kernel::AbstractKernel)
    return MatrixFreeMassOperatorKA(cache_ka, kernel)
end

@inline Base.size(op::MatrixFreeMassOperatorKA) =
    (length(op.cache_ka.dof_counts), length(op.cache_ka.dof_counts))
@inline Base.size(op::MatrixFreeMassOperatorKA, d::Integer) =
    (d == 1 || d == 2) ? length(op.cache_ka.dof_counts) : 1
Base.eltype(::MatrixFreeMassOperatorKA{F}) where {F} = F

LinearAlgebra.issymmetric(::MatrixFreeMassOperatorKA) = true
LinearAlgebra.ishermitian(::MatrixFreeMassOperatorKA) = true

@inline function (op::MatrixFreeMassOperatorKA)(y::AbstractVector{F},
                                               x::AbstractVector{F}) where {F}
    return LinearAlgebra.mul!(y, op, x)
end

function Base.:*(op::MatrixFreeMassOperatorKA{F}, x::AbstractVector{F}) where {F}
    y = similar(x, F, size(op, 1))
    return LinearAlgebra.mul!(y, op, x)
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::MatrixFreeMassOperatorKA{F},
                            x::AbstractVector{F}) where {F<:AbstractFloat}
    n = size(op, 1)
    length(y) == n || throw(DimensionMismatch("y length $(length(y)); expected $n"))
    length(x) == n || throw(DimensionMismatch("x length $(length(x)); expected $n"))
    workbuf = op.workbuf
    @inbounds @simd for i in eachindex(workbuf)
        workbuf[i] = x[i]
    end
    apply_M!(y, op.cache_ka, op.kernel, workbuf)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::MatrixFreeMassOperatorKA{F},
                            x::AbstractVector{F},
                            α::Number, β::Number) where {F<:AbstractFloat}
    n = size(op, 1)
    if iszero(β)
        fill!(y, zero(F))
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

# ---------------------------------------------------------------------------
# Stiffness operator on `DOFBasedCOOCacheKA` (KernelAbstractions path).
# ---------------------------------------------------------------------------

"""
    MatrixFreeOperatorKA{F, K, CKA, B, L}

Matrix-free stiffness matvec on a [`DOFBasedCOOCacheKA`](@ref): each
`mul!(y, op, x)` fills the scratch with [`prepare_multiply_workspace!`](@ref)
(default [`LocalMultiplyLayout`](@ref): copy `x`), then calls
[`apply_K!`](@ref)`(y, op.cache_ka, op.kernel, workbuf)`.

This is the KA analogue of unconstrained [`MatrixFreeOperator`](@ref) volume
matvecs: there are no Dirichlet / MPC hooks and no Pass~1 `configuration` /
`global_material_cache` / `Δt` forwarding, because the KA `apply_K!` entry
point is stiffness-only on the device batch. Use [`matrix_free_op`](@ref)
when you need constraints or nonlinear Pass~1 on the CPU cache.

Precision `F` matches `eltype(cache_ka.detJ_w_batch)`. Not a subtype of
[`AbstractMatrixFreeOperator`](@ref); see [`MatrixFreeMassOperatorKA`](@ref).

# Example

```julia
op_k = MatrixFreeOperatorKA(cache_ka, prototype_kernel(cpu_cache.kernel_column))
mul!(y, op_k, x)
```
"""
struct MatrixFreeOperatorKA{F<:AbstractFloat,
                             K<:AbstractKernel,
                             CKA<:DOFBasedCOOCacheKA,
                             B<:AbstractVector{F},
                             L<:AbstractMultiplyGhostLayout}
    cache_ka::CKA
    kernel::K
    workbuf::B
    mulbuf::B
    multiply_layout::L
end

function MatrixFreeOperatorKA(
    cache_ka::DOFBasedCOOCacheKA,
    kernel::K;
    multiply_layout::L = LocalMultiplyLayout(),
) where {K<:AbstractKernel, L <: AbstractMultiplyGhostLayout}
    F = eltype(cache_ka.detJ_w_batch)
    n = length(cache_ka.dof_counts)
    be = KernelAbstractions.get_backend(cache_ka.detJ_w_batch)
    z = Adapt.adapt(be, zeros(F, n))
    return MatrixFreeOperatorKA{F,K,typeof(cache_ka),typeof(z),L}(
        cache_ka,
        kernel,
        z,
        similar(z),
        multiply_layout,
    )
end

@inline function matrix_free_op_ka(
    cache_ka::DOFBasedCOOCacheKA,
    kernel::AbstractKernel;
    multiply_layout::L = LocalMultiplyLayout(),
) where {L <: AbstractMultiplyGhostLayout}
    return MatrixFreeOperatorKA(cache_ka, kernel; multiply_layout = multiply_layout)
end

@inline Base.size(op::MatrixFreeOperatorKA) =
    (length(op.cache_ka.dof_counts), length(op.cache_ka.dof_counts))
@inline Base.size(op::MatrixFreeOperatorKA, d::Integer) =
    (d == 1 || d == 2) ? length(op.cache_ka.dof_counts) : 1
Base.eltype(::MatrixFreeOperatorKA{F}) where {F} = F

LinearAlgebra.issymmetric(::MatrixFreeOperatorKA) = true
LinearAlgebra.ishermitian(::MatrixFreeOperatorKA) = true

@inline LinearAlgebra.isposdef(op::MatrixFreeOperatorKA) =
    operator_is_posdef(op.kernel)

@inline function (op::MatrixFreeOperatorKA)(y::AbstractVector{F},
                                            x::AbstractVector{F}) where {F}
    return LinearAlgebra.mul!(y, op, x)
end

function Base.:*(op::MatrixFreeOperatorKA{F}, x::AbstractVector{F}) where {F}
    y = similar(x, F, size(op, 1))
    return LinearAlgebra.mul!(y, op, x)
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::MatrixFreeOperatorKA{F},
                            x::AbstractVector{F}) where {F<:AbstractFloat}
    n = size(op, 1)
    length(y) == n || throw(DimensionMismatch("y length $(length(y)); expected $n"))
    length(x) == n || throw(DimensionMismatch("x length $(length(x)); expected $n"))
    workbuf = op.workbuf
    prepare_multiply_workspace!(workbuf, x, op.multiply_layout)
    apply_K!(y, op.cache_ka, op.kernel, workbuf)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{F},
                            op::MatrixFreeOperatorKA{F},
                            x::AbstractVector{F},
                            α::Number, β::Number) where {F<:AbstractFloat}
    n = size(op, 1)
    if iszero(β)
        fill!(y, zero(F))
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

# ---------------------------------------------------------------------------
# Nonlinear internal force and equilibrium residual (callable wrappers).
# ---------------------------------------------------------------------------

"""
    InternalForceOperator{C,A,M}

Callable wrapper around [`assemble_internal_force!`](@ref): `(op)(f, u)`
zeros and fills `f` with ``f_{\\mathrm{int}}(u)`` using `configuration = u`.

Holds optional Pass~1 `global_material_cache` and `Δt` (no separate
`configuration` field: the displacement state is always the second argument).

Not a subtype of [`AbstractMatrixFreeOperator`](@ref); there is no `mul!`
contract for a nonlinear force map.

`InternalForceOperator(cache, asm, kernel, mesh; …)` forwards to the
three-argument mesh form and ignores `kernel` (same deprecation hook as
[`matrix_free_op`](@ref)).
"""
struct InternalForceOperator{C<:DOFBasedCOOCache,
                             A<:DOFBasedCOOAssembler,
                             M<:AbstractMesh}
    cache::C
    asm::A
    mesh::M
    global_material_cache::Union{Nothing,GlobalMaterialCache}
    Δt::Float64
end

function InternalForceOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    return InternalForceOperator(
        cache, asm, mesh, global_material_cache, Float64(Δt),
    )
end

@inline function InternalForceOperator(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    kwargs...,
)
    _depwarn_redundant_kernel_arg!(:InternalForceOperator)
    return InternalForceOperator(cache, asm, mesh; kwargs...)
end

@inline function (op::InternalForceOperator)(
    f::AbstractVector{Float64},
    u::AbstractVector{Float64},
)
    return assemble_internal_force!(
        f, op.cache, op.asm, op.mesh;
        configuration = u,
        global_material_cache = op.global_material_cache,
        Δt = op.Δt,
    )
end

"""
    internal_force_op(cache, asm, mesh; global_material_cache = nothing, Δt = 0.0)
    internal_force_op(cache, asm, kernel, mesh; …)

Build an [`InternalForceOperator`](@ref). The four-argument form ignores `kernel`
(emits `Base.depwarn` once per session, same hook as [`matrix_free_op`](@ref)).
"""
@inline function internal_force_op(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    return InternalForceOperator(
        cache, asm, mesh;
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

@inline function internal_force_op(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    _depwarn_redundant_kernel_arg!(:internal_force_op)
    return internal_force_op(
        cache, asm, mesh;
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

"""
    NonlinearResidualOperator{C,A,M,F,W}

Callable wrapper around [`nonlinear_equilibrium_residual!`](@ref):
`(op)(r, u)` sets `r = f_{\\mathrm{ext}} - f_{\\mathrm{int}}(u)`.

References external load `f_ext` and a scratch vector `work` (same length as
`cache.ndofs`) owned by the operator or supplied at construction.
"""
struct NonlinearResidualOperator{C<:DOFBasedCOOCache,
                                 A<:DOFBasedCOOAssembler,
                                 M<:AbstractMesh,
                                 F<:AbstractVector{Float64},
                                 W<:AbstractVector{Float64}}
    cache::C
    asm::A
    mesh::M
    f_ext::F
    work::W
    global_material_cache::Union{Nothing,GlobalMaterialCache}
    Δt::Float64
end

function NonlinearResidualOperator(
    f_ext::AbstractVector{Float64},
    work::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    nd = cache.ndofs
    length(f_ext) == nd ||
        throw(DimensionMismatch("f_ext length $(length(f_ext)); expected $nd"))
    length(work) == nd ||
        throw(DimensionMismatch("work length $(length(work)); expected $nd"))
    return NonlinearResidualOperator(
        cache, asm, mesh, f_ext, work, global_material_cache, Float64(Δt),
    )
end

"""
    nonlinear_residual_op(f_ext, cache, asm, mesh; kwargs…)
    nonlinear_residual_op(f_ext, cache, asm, kernel, mesh; kwargs…)

Build a [`NonlinearResidualOperator`](@ref). The five-argument form ignores
`kernel` (emits `Base.depwarn` once per session, same hook as [`matrix_free_op`](@ref)).
"""
function nonlinear_residual_op(
    f_ext::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    mesh::AbstractMesh;
    work::Union{Nothing,AbstractVector{Float64}} = nothing,
    global_material_cache::Union{Nothing,GlobalMaterialCache} = nothing,
    Δt::Real = 0.0,
)
    nd = cache.ndofs
    w = work === nothing ? zeros(Float64, nd) : work
    return NonlinearResidualOperator(
        f_ext, w, cache, asm, mesh;
        global_material_cache = global_material_cache,
        Δt = Δt,
    )
end

@inline function nonlinear_residual_op(
    f_ext::AbstractVector{Float64},
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    ::AbstractKernel,
    mesh::AbstractMesh;
    kwargs...,
)
    _depwarn_redundant_kernel_arg!(:nonlinear_residual_op)
    return nonlinear_residual_op(f_ext, cache, asm, mesh; kwargs...)
end

@inline function (op::NonlinearResidualOperator)(
    r::AbstractVector{Float64},
    u::AbstractVector{Float64},
)
    return nonlinear_equilibrium_residual!(
        r, op.f_ext, op.work, op.cache, op.asm, op.mesh, u;
        global_material_cache = op.global_material_cache,
        Δt = op.Δt,
    )
end
