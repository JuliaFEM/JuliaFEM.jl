# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Lightweight, matrix-free preconditioners for Krylov solves built on top
of `apply_K!` + `matrix_free_op`. Concrete types include:

  * `JacobiPreconditioner`: stores `1 ./ diag(K_op)` and implements the
    `LinearAlgebra.ldiv!` protocol consumed by `IterativeSolvers.cg!`,
    `Krylov.jl`, `LinearSolve.jl`, etc.

The diagonal is extracted matrix-free via `compute_diagonal!` (same
DOF-row traversal as `apply_K!`), so building the preconditioner has
the same complexity as a single `apply_K!` call — no extra storage
beyond the `ndofs`-vector of inverse diagonal entries.

For symmetric-indefinite mixed `u`–`p` problems (`MixedUPKernel` with
`inv_bulk = 0`), scalar Jacobi is only a baseline: Krylov solvers should
use GMRES or MINRES, not CG, and a field-split or approximate Schur
preconditioner is usually required for robust convergence (see
`test/domains/continuum/test_mixed_up_incompressible_solve.jl`).
[`ApproxSchurDiagBlockPreconditioner`](@ref) applies diagonal scaling on the primal block
and an entry-wise diagonal approximation of the pressure Schur complement from an assembled
sparse `K` (useful baseline for symmetric saddle-point systems; not a general-purpose
solver for hard indefinite problems).
[`ApproxSchurICholDiagBlockPreconditioner`](@ref) replaces the primal diagonal by IC(0) on
the `A` block when that slice is SPD enough for an incomplete factorisation.

# Usage

```julia
using IterativeSolvers, LinearOperators, JuliaFEM

bc = PenaltyDirichlet(fixed_dofs, vals; penalty = 1e8)
op = matrix_free_op(cache, asm, kernel, mesh; dirichlet = bc)
P  = JacobiPreconditioner(cache, asm, kernel, mesh; dirichlet = bc)

linop = LinearOperator(Float64, n, n, true, true, op)

T = zeros(n)
cg!(T, linop, b; Pl = P, abstol = 1e-12, reltol = 1e-12, maxiter = 4n)
```

For `PenaltyDirichlet` the Jacobi diagonal includes the `λ`
contribution, so CG sees a roughly `O(1)`-conditioned system on the
fixed DOFs and the original conditioning on the free block — closing
the gap that motivated `EliminatedDirichlet` in the first place.
"""

import LinearAlgebra
import LinearAlgebra: ldiv!
using LinearAlgebra: diag, inv, LowerTriangular, UpperTriangular
using SparseArrays: SparseMatrixCSC, sparse, nnz, rowvals, nonzeros, getcolptr

"""
    JacobiPreconditioner(inv_diag::Vector{Float64})

Diagonal preconditioner with stored *inverse* diagonal so that
`ldiv!(P, x)` is a single multiplication per entry. Direct constructor;
prefer the `(cache, asm, kernel, mesh; dirichlet)` factory for the
common matrix-free path.
"""
struct JacobiPreconditioner
    inv_diag::Vector{Float64}
end

"""
    JacobiPreconditioner(cache, asm, kernel, mesh; dirichlet = nothing, mpc = nothing)

Build a Jacobi preconditioner from the matrix-free operator. Internally:

  1. extract `diag(K)` matrix-free via `compute_diagonal!`,
  2. apply the constraint's diagonal hook (`apply_constraint_diag!`) so
     the preconditioner matches the operator returned by
     `matrix_free_op(...; dirichlet, mpc)`. Both Dirichlet and MPC
     constraints are added if supplied (additive: the MPC adds
     `λ · diag(Cᵀ C)` on top of the Dirichlet diagonal),
  3. invert entry-wise (zeros are kept as `1.0` so the preconditioner
     stays well-defined; in practice every DOF row of `K` has a positive
     diagonal in any well-posed FEM problem).
"""
function JacobiPreconditioner(cache::DOFBasedCOOCache,
                              asm::DOFBasedCOOAssembler,
                              kernel::AbstractKernel,
                              mesh::AbstractMesh;
                              dirichlet::Union{AbstractDirichletConstraint, Nothing} = nothing,
                              mpc::Union{AbstractMultipointConstraint, Nothing} = nothing)
    n = cache.ndofs
    d = zeros(Float64, n)
    compute_diagonal!(d, cache, asm, kernel, mesh)
    if dirichlet !== nothing
        apply_constraint_diag!(d, dirichlet)
    end
    if mpc !== nothing
        apply_constraint_diag!(d, mpc)
    end

    inv_d = similar(d)
    @inbounds for i in eachindex(d)
        inv_d[i] = d[i] != 0.0 ? 1.0 / d[i] : 1.0
    end
    return JacobiPreconditioner(inv_d)
end

# `ldiv!(y, P, x)` is the contract IterativeSolvers.cg! / Krylov.jl call
# for left-preconditioning: compute `y = P^{-1} * x`. For a diagonal
# `P`, this is just an entry-wise multiplication by the stored inverse.
function ldiv!(y::AbstractVector{Float64}, P::JacobiPreconditioner,
               x::AbstractVector{Float64})
    @inbounds @simd for i in eachindex(y)
        y[i] = x[i] * P.inv_diag[i]
    end
    return y
end

# In-place variant — mutates `x` directly.
function ldiv!(P::JacobiPreconditioner, x::AbstractVector{Float64})
    @inbounds @simd for i in eachindex(x)
        x[i] *= P.inv_diag[i]
    end
    return x
end


# ============================================================================
# BlockJacobiPreconditioner{N} — N×N block-diagonal preconditioner
# ============================================================================
#
# Scalar Jacobi is weak for vector problems where the off-diagonal entries
# *within a node block* are O(diag) — typical for elasticity where the
# 3×3 nodal block has full off-diagonal couplings. Inverting the 3×3
# block as a unit recovers a much better approximation of `K^{-1}` at
# the same matrix-free cost (one extra `evaluate_entry` call per off-
# diagonal entry of each block).
#
# The block layout assumes the DOFHandler's standard interleaved
# numbering: `dof = N · (block_id − 1) + comp_in_block`, with
# `comp_in_block ∈ 1..N`. This matches how `create_elements!` numbers
# vector-field DOFs (and the same convention generalizes to N=4 for
# `Displacement + Temperature`).

"""
    compute_block_diagonal!(blocks::Array{Float64,3},
                            cache::DOFBasedCOOCache,
                            asm::DOFBasedCOOAssembler,
                            kernel::AbstractKernel,
                            mesh::AbstractMesh) -> blocks

Assemble the `N × N` block-diagonal of the stiffness matrix into
`blocks`, a `(N, N, n_blocks)` array. Block `b` covers global DOFs
`N · (b − 1) + 1 : N · b` (interleaved numbering, the standard layout
for vector fields).

Same DOF-by-element traversal as `assemble!` but only the entries with
both DOFs in the same block contribute. Allocation-free after warmup.
"""
function compute_block_diagonal!(
    blocks::Array{Float64,3},
    cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType},
    asm::DOFBasedCOOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh,
) where {T,B,IPS,E<:AbstractElement,GC,Buf,FieldType,StateType}
    N, N2, n_blocks = size(blocks)
    @assert N == N2 (
        "compute_block_diagonal!: blocks must be (N,N,n_blocks); got $(size(blocks))")
    @assert N * n_blocks == cache.ndofs (
        "compute_block_diagonal!: N · n_blocks = $(N * n_blocks) must equal " *
        "cache.ndofs = $(cache.ndofs)")

    fill!(blocks, 0.0)
    _prepare_caches!(cache, kernel, mesh)

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches
    qp_buffers       = cache.qp_buffers

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for elem_idx in 1:length(elements)
        ec = element_caches[elem_idx]
        gc = geometry_caches[elem_idx]
        qp = view(qp_buffers, :, elem_idx)
        dofs_elem = ec.dofs

        @inbounds for li in 1:ndofs_elem
            entry_i  = layout[li]
            dof_i    = Int(dofs_elem[li])
            blk_i    = div(dof_i - 1, N) + 1
            comp_i   = mod(dof_i - 1, N) + 1

            @inbounds for lj in 1:ndofs_elem
                dof_j = Int(dofs_elem[lj])
                blk_j = div(dof_j - 1, N) + 1
                if blk_i != blk_j
                    continue
                end
                comp_j  = mod(dof_j - 1, N) + 1
                entry_j = layout[lj]

                K_ij = evaluate_entry(kernel, gc, qp, entry_i, entry_j, elem_idx)
                blocks[comp_i, comp_j, blk_i] += K_ij
            end
        end
    end

    return blocks
end

"""
    apply_constraint_block_diag!(blocks::Array{Float64,3}, c) -> blocks

Apply a Dirichlet constraint to the block-diagonal storage so the
preconditioner stays consistent with the operator returned by
`matrix_free_op(...; dirichlet = c)`. Default: no-op. Override for new
constraint types parallel to `apply_constraint_diag!`.
"""
@inline apply_constraint_block_diag!(blocks::Array{Float64,3},
                                     ::AbstractDirichletConstraint) = blocks

@inline function apply_constraint_block_diag!(blocks::Array{Float64,3},
                                              c::PenaltyDirichlet)
    N, _, _ = size(blocks)
    λ = c.penalty
    @inbounds for fdof in c.fixed_dofs
        blk = div(fdof - 1, N) + 1
        cmp = mod(fdof - 1, N) + 1
        blocks[cmp, cmp, blk] += λ
    end
    return blocks
end

@inline function apply_constraint_block_diag!(blocks::Array{Float64,3},
                                              c::EliminatedDirichlet)
    N, _, _ = size(blocks)
    @inbounds for fdof in c.fixed_dofs
        blk = div(fdof - 1, N) + 1
        cmp = mod(fdof - 1, N) + 1
        # Zero the row/column of the eliminated DOF inside the block,
        # then identity on the diagonal — mirrors the eliminated
        # operator on the corresponding block row.
        @inbounds for k in 1:N
            blocks[cmp, k, blk] = 0.0
            blocks[k, cmp, blk] = 0.0
        end
        blocks[cmp, cmp, blk] = 1.0
    end
    return blocks
end

"""
    BlockJacobiPreconditioner{N}

`N × N` block-diagonal preconditioner. Stores the *inverse* of every
block (a single `(N, N, n_blocks)` array) so `ldiv!` is a single dense
`N`-by-`N` matrix-vector multiply per block — `O(N² · n_blocks)` work,
no allocations.

For elasticity (`Displacement{3}`) use `N = 3`; for thermo-elastic
(displacement + temperature) use `N = 4`. The block grouping assumes
DOFs are numbered `dof = N · (block − 1) + comp`, which matches the
`create_elements!` layout for any `@DOFSet` whose total per-vertex DOF
count is `N`.

Mixed displacement (`Vertex`) plus cell-centered pressure (`Cell`) places all
velocity DOFs first, then all pressures ([`global_field_ranges`](@ref)); that layout
does **not** match fixed-width `N`-DOF blocks over the full unknown vector, so
[`BlockJacobiPreconditioner`](@ref) is not appropriate for the whole saddle-point
system — use scalar [`JacobiPreconditioner`](@ref), a field-split / Schur-based
preconditioner on extracted blocks ([`saddle_point_blocks`](@ref)), or a direct solver.
"""
struct BlockJacobiPreconditioner{N}
    inv_blocks::Array{Float64,3}     # (N, N, n_blocks)
end

"""
    BlockJacobiPreconditioner{N}(cache, asm, kernel, mesh; dirichlet = nothing)

Build a block-Jacobi preconditioner from the matrix-free operator.
Internally:

  1. assemble the `N × N` diagonal blocks via `compute_block_diagonal!`,
  2. apply `apply_constraint_block_diag!` so the preconditioner matches
     the operator returned by `matrix_free_op(...; dirichlet)`,
  3. invert each block in place (`LinearAlgebra.inv` via `LU`); on
     singular blocks, fall back to identity to keep the preconditioner
     well-defined (a singular nodal block typically signals an ill-posed
     problem, so this is intentionally permissive).
"""
function BlockJacobiPreconditioner{N}(
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh;
    dirichlet::Union{AbstractDirichletConstraint, Nothing} = nothing,
) where {N}
    @assert N >= 1 "BlockJacobiPreconditioner: N must be >= 1 (got $N)"
    n_blocks = div(cache.ndofs, N)
    @assert N * n_blocks == cache.ndofs (
        "BlockJacobiPreconditioner{N=$N}: ndofs = $(cache.ndofs) is not divisible by N")

    blocks = zeros(Float64, N, N, n_blocks)
    compute_block_diagonal!(blocks, cache, asm, kernel, mesh)
    if dirichlet !== nothing
        apply_constraint_block_diag!(blocks, dirichlet)
    end

    inv_blocks = similar(blocks)
    block_view = zeros(Float64, N, N)
    @inbounds for b in 1:n_blocks
        @inbounds for j in 1:N, i in 1:N
            block_view[i, j] = blocks[i, j, b]
        end
        invblk = try
            inv(block_view)
        catch
            Matrix{Float64}(LinearAlgebra.I, N, N)
        end
        @inbounds for j in 1:N, i in 1:N
            inv_blocks[i, j, b] = invblk[i, j]
        end
    end

    return BlockJacobiPreconditioner{N}(inv_blocks)
end

# ldiv!(y, P, x): y = P^{-1} x, block by block.
function ldiv!(y::AbstractVector{Float64},
               P::BlockJacobiPreconditioner{N},
               x::AbstractVector{Float64}) where {N}
    n_blocks = size(P.inv_blocks, 3)
    @inbounds for b in 1:n_blocks
        base = N * (b - 1)
        @inbounds for i in 1:N
            yi = 0.0
            @inbounds for j in 1:N
                yi += P.inv_blocks[i, j, b] * x[base + j]
            end
            y[base + i] = yi
        end
    end
    return y
end

# In-place: x ← P^{-1} x.
function ldiv!(P::BlockJacobiPreconditioner{N},
               x::AbstractVector{Float64}) where {N}
    n_blocks = size(P.inv_blocks, 3)
    tmp = zeros(Float64, N)
    @inbounds for b in 1:n_blocks
        base = N * (b - 1)
        @inbounds for j in 1:N
            tmp[j] = x[base + j]
        end
        @inbounds for i in 1:N
            yi = 0.0
            @inbounds for j in 1:N
                yi += P.inv_blocks[i, j, b] * tmp[j]
            end
            x[base + i] = yi
        end
    end
    return x
end

# ============================================================================
# ApproxSchurDiagBlockPreconditioner — field-split diagonal + diag(Schur) estimate
# ============================================================================
#
# For K ≈ [ A  B ; Bt C ] with A (velocity block) square and SPD-ish, a classical
# cheap preconditioner uses inv(diag(A)) on the u block and inv(diag(S)) on p,
# where S ≈ C - B' diag(A)^{-1} B is the Schur complement. We approximate diag(S)
# entry-wise as  Cjj - sum_i B_ij^2 / A_ii  (using B and Bt consistently only when
# K is symmetric). For non-symmetric K the same diagonal formula is still used as a
# heuristic when building from `B = K[r_u,r_p]` and `diag(A)`.

function _inv_diag_A_and_schur(blks)
    A = blks.A
    B = blks.B
    C = blks.C
    nu = size(A, 1)
    np = size(C, 1)
    dA = diag(A)
    inv_dA = Vector{Float64}(undef, nu)
    @inbounds for i in 1:nu
        inv_dA[i] = dA[i] != 0.0 ? 1.0 / dA[i] : 1.0
    end
    dC = diag(C)
    schur_diag = Vector{Float64}(undef, np)
    @inbounds for j in 1:np
        s = dC[j]
        @inbounds for i in 1:nu
            bij = B[i, j]
            s -= bij * bij * inv_dA[i]
        end
        schur_diag[j] = s
    end
    inv_schur = Vector{Float64}(undef, np)
    @inbounds for j in 1:np
        sd = schur_diag[j]
        inv_schur[j] = abs(sd) > 1e-30 ? 1.0 / sd : 1.0
    end
    return inv_dA, inv_schur
end

"""
    ApproxSchurDiagBlockPreconditioner(inv_diag_A, inv_schur_diag, r_u, r_p)

Block-diagonal preconditioner in field-split ordering:

  * `y[r_u] ← inv_diag_A .* x[r_u]`  (diagonal scaling on the primal block)
  * `y[r_p] ← inv_schur_diag .* x[r_p]`  (diagonal Schur approximation on the second field)

Ranges must partition the relevant global indices (typical two-field mixed layout from
[`global_field_ranges`](@ref)). Construct with the `(K, r_u, r_p)` factory on an assembled
[`SparseMatrixCSC`](@ref).
"""
struct ApproxSchurDiagBlockPreconditioner
    inv_diag_A::Vector{Float64}
    inv_schur_diag::Vector{Float64}
    r_u::UnitRange{Int}
    r_p::UnitRange{Int}
end

"""
    ApproxSchurDiagBlockPreconditioner(K::SparseMatrixCSC, r_u, r_p)

Build from sparse `K` using [`saddle_point_matrix_blocks`](@ref): diagonal of `A`,
then `diag(S) ≈ diag(C) - sum_i B[i,j]^2 / A[i,i]` for each pressure column `j`.
Zeros on `diag(A)` fall back to unit inverse entries; tiny `diag(S)` uses `1.0`.
"""
function ApproxSchurDiagBlockPreconditioner(
    K::SparseMatrixCSC{Float64, Ti},
    r_u::AbstractRange{Int},
    r_p::AbstractRange{Int},
) where {Ti<:Integer}
    ru = UnitRange(first(r_u), last(r_u))
    rp = UnitRange(first(r_p), last(r_p))
    blks = saddle_point_matrix_blocks(K, ru, rp)
    nu = size(blks.A, 1)
    np = size(blks.C, 1)
    length(ru) == nu || error("ApproxSchurDiagBlockPreconditioner: |r_u| mismatch")
    length(rp) == np || error("ApproxSchurDiagBlockPreconditioner: |r_p| mismatch")

    inv_dA, inv_schur = _inv_diag_A_and_schur(blks)
    return ApproxSchurDiagBlockPreconditioner(inv_dA, inv_schur, ru, rp)
end

function ldiv!(
    y::AbstractVector{Float64},
    P::ApproxSchurDiagBlockPreconditioner,
    x::AbstractVector{Float64},
)
    length(y) == length(x) || throw(DimensionMismatch("ldiv!"))
    ru, rp = P.r_u, P.r_p
    @views @. y[ru] = P.inv_diag_A * x[ru]
    @views @. y[rp] = P.inv_schur_diag * x[rp]
    return y
end

function ldiv!(P::ApproxSchurDiagBlockPreconditioner, x::AbstractVector{Float64})
    ru, rp = P.r_u, P.r_p
    @views @. x[ru] = P.inv_diag_A * x[ru]
    @views @. x[rp] = P.inv_schur_diag * x[rp]
    return x
end

# ============================================================================
# ICholPreconditioner — Incomplete Cholesky with no fill-in (IC(0))
# ============================================================================
#
# `IC(0)` is the canonical "industrial-strength" preconditioner for SPD
# Krylov solves: it computes a sparse lower-triangular `L` such that
# `L L^T ≈ K`, with `L`'s sparsity pattern *equal* to the lower
# triangle of `K` (no fill). For most well-shaped FE meshes this gives
# CG convergence in `O(√cond(K))` iterations — compared to scalar
# Jacobi's `O(cond(K))` — at the cost of one sparse triangular solve
# per Krylov iteration plus a one-time factorisation.
#
# Implementation notes
# --------------------
# * IC(0) is built directly on the user's assembled `K` (via
#   `extract_system(cache)`) — the matrix-free operator stays
#   matrix-free for the *Krylov product* (`apply_K!`) but uses the
#   assembled K *only* for the preconditioner. This hybrid is the
#   standard pattern: PCG hits `K` once per `apply_K!` call but `L`
#   twice per iteration via `ldiv!`.
# * The factorisation is a Crout left-looking IC(0) over CSC
#   storage. Per-column workspace is a dense `Vector{Float64}` of
#   length `n` plus an integer `pos` array of the same length;
#   total `O(n)` extra memory, completely independent of the matrix.
# * After factorisation, `ldiv!(P, x)` runs SparseArrays' optimised
#   forward / backward substitution via `LowerTriangular(L)` and
#   `UpperTriangular(L')`. Both are zero-allocation hot paths.
# * Apply Dirichlet BCs to `K` *before* building `IC(0)` (typical
#   pattern: the penalty / elimination has already added `λ` to the
#   diagonal entries on the constrained DOFs). The preconditioner
#   then matches the operator the Krylov solver sees.

"""
    ICholPreconditioner

Lower-triangular Incomplete-Cholesky preconditioner. Build it with
either `ICholPreconditioner(K::SparseMatrixCSC)` (factor an existing
SPD matrix) or `ICholPreconditioner(cache, asm, kernel, mesh; dirichlet)`
(assemble + factor in one shot for the matrix-free workflow).

`ldiv!(P, x)` solves `L L^T y = x` via two sparse triangular solves
(forward then backward) — the canonical PCG preconditioner action.
"""
struct ICholPreconditioner
    L::SparseMatrixCSC{Float64,Int}
    # Inner constructor used by the IC(0) factorisation pipeline.
    # The `::Val{:ichol_internal}` tag prevents the outer
    # `ICholPreconditioner(K::SparseMatrixCSC...)` constructor from
    # recursing into itself when it tries to wrap the freshly-built
    # `L` factor (both have the same `SparseMatrixCSC{Float64,Int}`
    # signature).
    ICholPreconditioner(L::SparseMatrixCSC{Float64,Int}, ::Val{:ichol_internal}) = new(L)
end

"""
    ICholPreconditioner(K::SparseMatrixCSC{Float64,Int})

Build the IC(0) factor of the SPD matrix `K` and wrap it as a
preconditioner. Sparsity pattern of `L` = lower triangle of `K`
(no fill).

# Robustness

If a Cholesky update produces a non-positive diagonal (numerically not
SPD — e.g. the user passed an indefinite or near-singular matrix), the
factorisation falls back to a small diagonal-shift retry: `K + α · I`
with `α = max(0, 1e-12 · ‖K‖∞)`. This recovers in practice for
slightly-ill-conditioned PenaltyDirichlet systems while keeping a
well-defined preconditioner.
"""
function ICholPreconditioner(K::SparseMatrixCSC{Float64,Int})
    n = size(K, 1)
    @assert n == size(K, 2) "ICholPreconditioner: K must be square (got $(size(K)))"

    # Try the unshifted IC(0) first. On breakdown, fall back to a
    # progressively larger diagonal shift `K + α·I`. This is the
    # standard Manteuffel-style robustification for elasticity-grade
    # matrices, where IC(0) frequently produces a non-positive pivot
    # because of dropped fill — even though `K` itself is SPD. The
    # operator the Krylov solver sees is still the unshifted `K`
    # (we just need a usable preconditioner).
    L = try
        _ic0_factor(K)
    catch err
        if err isa ErrorException && occursin("IC(0)", err.msg)
            kmax  = maximum(abs.(nonzeros(K)))
            shift_levels = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0)
            local Lα = nothing
            for s in shift_levels
                α   = max(s * kmax, 1e-300)
                Kα  = K + α * sparse(LinearAlgebra.I, n, n)
                Lα = try
                    _ic0_factor(Kα)
                catch e
                    if e isa ErrorException && occursin("IC(0)", e.msg)
                        nothing
                    else
                        rethrow(e)
                    end
                end
                Lα === nothing || break
            end
            Lα === nothing ?
                error("ICholPreconditioner: IC(0) breakdown even after " *
                      "diagonal shift up to ‖K‖∞ = $(kmax). Matrix may not " *
                      "be SPD or is too far from diagonally dominant for " *
                      "no-fill incomplete Cholesky.") :
                Lα
        else
            rethrow(err)
        end
    end
    return ICholPreconditioner(L, Val(:ichol_internal))
end

"""
    ICholPreconditioner(cache::DOFBasedCOOCache, asm, kernel, mesh;
                        dirichlet = nothing) -> ICholPreconditioner

Convenience constructor for the matrix-free workflow:

    1. assemble K via `assemble!(cache, asm, kernel, mesh)` and
       extract it with `extract_system(cache)`,
    2. apply the Dirichlet constraint (if given) to `K`,
    3. build `IC(0)` of the modified `K`.

This *does* materialise the sparse `K` once — the price of IC(0). The
returned preconditioner then plugs into the matrix-free Krylov solve.
"""
function ICholPreconditioner(cache::DOFBasedCOOCache,
                             asm::DOFBasedCOOAssembler,
                             kernel::AbstractKernel,
                             mesh::AbstractMesh;
                             dirichlet::Union{AbstractDirichletConstraint, Nothing} = nothing)
    assemble!(cache, asm, kernel, mesh)
    K, _ = extract_system(cache)
    if dirichlet !== nothing
        apply_constraint!(K, dirichlet)
    end
    return ICholPreconditioner(K)
end

# Crout left-looking IC(0) on CSC storage. Returns the factor `L`
# such that `L * L^T ≈ K` and `nnz(L) == nnz(tril(K))`.
function _ic0_factor(K::SparseMatrixCSC{Float64,Int})
    n = size(K, 1)
    # Initialise `L` with the lower-triangle pattern of `K`. We must
    # deep-copy `nzval` explicitly: in some Julia / SparseArrays
    # versions `sparse(LowerTriangular(K))` can alias `K.nzval`, so
    # writing IC(0) factor entries into `L.nzval` would silently
    # corrupt the user's `K`. Found the hard way during the
    # diagonal-shift fallback path.
    Ltmp   = sparse(LowerTriangular(K))
    L      = SparseMatrixCSC(size(Ltmp, 1), size(Ltmp, 2),
                             copy(getcolptr(Ltmp)),
                             copy(rowvals(Ltmp)),
                             copy(nonzeros(Ltmp)))
    rowval = rowvals(L)
    nzval  = nonzeros(L)
    colptr = getcolptr(L)

    # Workspace: dense column buffer + position lookup. `pos[i] != 0`
    # iff row `i` is in column `k`'s pattern (i.e. there's an entry
    # `L[i,k]`).
    work = zeros(Float64, n)
    pos  = zeros(Int,     n)

    @inbounds for k in 1:n
        # 1. Splat column k of L into the dense workspace.
        kpos = 0
        for jp in colptr[k]:(colptr[k+1] - 1)
            i = rowval[jp]
            work[i] = nzval[jp]
            pos[i]  = jp
            if i == k
                kpos = jp
            end
        end
        if kpos == 0
            error("IC(0): missing diagonal entry K[$k,$k] — matrix is not " *
                  "structurally SPD.")
        end

        # 2. For each previous column j < k that has L[k, j] ≠ 0,
        #    subtract `L[k,j] * L[i,j]` from `work[i]` for every row
        #    `i ≥ k` already in the pattern of column k.
        for j in 1:(k-1)
            # Look up L[k, j] (row k in column j). Bail out fast if
            # row k isn't in column j's pattern.
            jp_kj = _find_row_in_col(rowval, colptr, j, k)
            if jp_kj == 0
                continue
            end
            Lkj = nzval[jp_kj]

            # Walk down column j from row k onwards; each row i with
            # `pos[i] ≠ 0` (i.e. in column k's pattern) contributes
            # `-Lkj * L[i,j]` to `work[i]`. Other rows fall on
            # would-be fill, dropped by IC(0).
            for jp in jp_kj:(colptr[j+1] - 1)
                i = rowval[jp]
                if pos[i] != 0
                    work[i] -= Lkj * nzval[jp]
                end
            end
        end

        # 3. Diagonal — must be positive for the matrix to be SPD.
        d = work[k]
        if d <= 0.0
            error("IC(0): non-positive diagonal at row $k (got $d). " *
                  "Matrix may not be SPD.")
        end
        sqrt_d = sqrt(d)
        nzval[kpos] = sqrt_d

        # 4. Sub-diagonals of column k are the updated work entries
        #    divided by the new diagonal.
        for jp in (kpos + 1):(colptr[k+1] - 1)
            i = rowval[jp]
            nzval[jp] = work[i] / sqrt_d
        end

        # 5. Clear the dense workspace for the next column. We only
        #    touched entries in column k's pattern, so this is cheap.
        for jp in colptr[k]:(colptr[k+1] - 1)
            i = rowval[jp]
            work[i] = 0.0
            pos[i]  = 0
        end
    end

    return L
end

# Binary search column j (sorted row indices) for row r. Returns the
# nzval index, or `0` if not found.
@inline function _find_row_in_col(rowval, colptr, j::Int, r::Int)
    lo = colptr[j]
    hi = colptr[j+1] - 1
    while lo <= hi
        mid = (lo + hi) >> 1
        rv  = rowval[mid]
        if rv == r
            return mid
        elseif rv < r
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

# `ldiv!(y, P, x)`: y = (L L^T)^{-1} x. Two sparse triangular solves.
function ldiv!(y::AbstractVector{Float64},
               P::ICholPreconditioner,
               x::AbstractVector{Float64})
    L = P.L
    # Forward solve: L * z = x  ⇒  z = L \ x. We use `y` as the
    # output buffer; SparseArrays' triangular solve is allocation-
    # free when both args are concrete `Vector{Float64}`.
    copyto!(y, x)
    LinearAlgebra.ldiv!(LowerTriangular(L), y)
    # Backward solve: L^T * y = z. Reuses the same storage.
    LinearAlgebra.ldiv!(UpperTriangular(L'), y)
    return y
end

# In-place: x ← (L L^T)^{-1} x.
function ldiv!(P::ICholPreconditioner, x::AbstractVector{Float64})
    L = P.L
    LinearAlgebra.ldiv!(LowerTriangular(L), x)
    LinearAlgebra.ldiv!(UpperTriangular(L'), x)
    return x
end

# ============================================================================
# ApproxSchurICholDiagBlockPreconditioner — IC(0) on A + diagonal Schur on p
# ============================================================================
#
# Stronger than [`ApproxSchurDiagBlockPreconditioner`](@ref) on the primal block:
# applies [`ICholPreconditioner`](@ref) to the sparse `A = K[r_u,r_u]` slice (requires that
# block to be SPD enough for IC(0)), and keeps the same diagonal Schur approximation on
# the pressure block.

"""
    ApproxSchurICholDiagBlockPreconditioner(Pa, inv_schur_diag, r_u, r_p, tmp_u)

Field-split preconditioner: [`ldiv!`](@ref) with [`ICholPreconditioner`](@ref) on `x[r_u]`
(stored in `tmp_u`) and diagonal Schur scaling on `x[r_p]`.
"""
struct ApproxSchurICholDiagBlockPreconditioner
    Pa::ICholPreconditioner
    inv_schur_diag::Vector{Float64}
    r_u::UnitRange{Int}
    r_p::UnitRange{Int}
    tmp_u::Vector{Float64}
end

"""
    ApproxSchurICholDiagBlockPreconditioner(K::SparseMatrixCSC, r_u, r_p)

Factor `A = K[r_u,r_u]` with IC(0) and build the same `inv_schur_diag` as
[`ApproxSchurDiagBlockPreconditioner`](@ref).
"""
function ApproxSchurICholDiagBlockPreconditioner(
    K::SparseMatrixCSC{Float64, Ti},
    r_u::AbstractRange{Int},
    r_p::AbstractRange{Int},
) where {Ti<:Integer}
    ru = UnitRange(first(r_u), last(r_u))
    rp = UnitRange(first(r_p), last(r_p))
    blks = saddle_point_matrix_blocks(K, ru, rp)
    nu = size(blks.A, 1)
    np = size(blks.C, 1)
    length(ru) == nu || error("ApproxSchurICholDiagBlockPreconditioner: |r_u| mismatch")
    length(rp) == np || error("ApproxSchurICholDiagBlockPreconditioner: |r_p| mismatch")

    Aint = convert(SparseMatrixCSC{Float64, Int}, blks.A)
    Pa = ICholPreconditioner(Aint)
    _, inv_schur = _inv_diag_A_and_schur(blks)
    tmp_u = zeros(Float64, nu)
    return ApproxSchurICholDiagBlockPreconditioner(Pa, inv_schur, ru, rp, tmp_u)
end

function ldiv!(
    y::AbstractVector{Float64},
    P::ApproxSchurICholDiagBlockPreconditioner,
    x::AbstractVector{Float64},
)
    length(y) == length(x) || throw(DimensionMismatch("ldiv!"))
    ru, rp = P.r_u, P.r_p
    @views copyto!(P.tmp_u, x[ru])
    ldiv!(P.Pa, P.tmp_u)
    @views y[ru] .= P.tmp_u
    @views @. y[rp] = P.inv_schur_diag * x[rp]
    return y
end

function ldiv!(P::ApproxSchurICholDiagBlockPreconditioner, x::AbstractVector{Float64})
    ru, rp = P.r_u, P.r_p
    @views copyto!(P.tmp_u, x[ru])
    ldiv!(P.Pa, P.tmp_u)
    @views x[ru] .= P.tmp_u
    @views @. x[rp] = P.inv_schur_diag * x[rp]
    return x
end
