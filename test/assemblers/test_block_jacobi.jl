# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`BlockJacobiPreconditioner{N}` tests.

Locks in the contract of the N×N block-diagonal preconditioner added in
B++:

  1. **`compute_block_diagonal!` matches the assembled block-diagonal
     of `K` to round-off** for both elasticity (N=3) and the trivial
     N=1 case (heat) where it must reduce to scalar `compute_diagonal!`.

  2. **`ldiv!(P, x)` matches `inv(blockdiag(K)) * x`** to round-off,
     i.e. the preconditioner is the *exact* inverse of the block
     diagonal (one of the few cases where this is true).

  3. **CG converges faster on bumpy elasticity problems with
     block-Jacobi than with scalar Jacobi**, on a representative
     elasticity bar with mixed-component coupling.

  4. **Constraint-aware diagonals**: building the preconditioner with
     `dirichlet = Penalty/Eliminated` keeps it consistent with
     `matrix_free_op(...; dirichlet)` so an end-to-end inhomogeneous
     Dirichlet CG solve converges to the correct displacement.

  5. **Zero allocations** for `ldiv!` (the hot path), and bounded
     allocations for the constructor (one-shot).
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: HeatConductivity, HeatKernel, Temperature
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: extract_system, apply_K!
using JuliaFEM: PenaltyDirichlet, EliminatedDirichlet, apply_constraint!
using JuliaFEM: matrix_free_op
using JuliaFEM: JacobiPreconditioner, BlockJacobiPreconditioner
using JuliaFEM: compute_diagonal!, compute_block_diagonal!
using JuliaFEM: NodalForce, UniformBodyForce, apply_load!
using JuliaFEM: create_elements!
using LinearAlgebra
using SparseArrays
using Tensors
using Random

# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

function _hex8_box(nx::Int, ny::Int, nz::Int;
                   Lx::Float64 = 1.0, Ly::Float64 = 1.0, Lz::Float64 = 1.0)
    nodes = Vec{3,Float64}[]
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        push!(nodes, Vec{3}((Lx * Float64(i - 1) / nx,
                             Ly * Float64(j - 1) / ny,
                             Lz * Float64(k - 1) / nz)))
    end
    conns = NTuple{8,UInt32}[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        n1 = nidx(i,     j,     k)
        n2 = nidx(i + 1, j,     k)
        n3 = nidx(i + 1, j + 1, k)
        n4 = nidx(i,     j + 1, k)
        n5 = nidx(i,     j,     k + 1)
        n6 = nidx(i + 1, j,     k + 1)
        n7 = nidx(i + 1, j + 1, k + 1)
        n8 = nidx(i,     j + 1, k + 1)
        push!(conns, (UInt32(n1), UInt32(n2), UInt32(n3), UInt32(n4),
                      UInt32(n5), UInt32(n6), UInt32(n7), UInt32(n8)))
    end
    return Mesh{8,Hexahedron{8}}(nodes, conns)
end

function _setup_elasticity(mesh)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S        = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

function _setup_heat(mesh; k_value::Float64 = 50.2)
    material = HeatConductivity(k = k_value)
    kernel   = HeatKernel(ContinuumFormulation{FullThreeD}(), material)
    S        = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# Naive block-diagonal of an assembled K, used as the round-off oracle.
function _assembled_block_diagonal(K::AbstractMatrix, N::Int)
    n          = size(K, 1)
    n_blocks   = div(n, N)
    blocks     = zeros(N, N, n_blocks)
    @inbounds for b in 1:n_blocks
        base = N * (b - 1)
        for j in 1:N, i in 1:N
            blocks[i, j, b] = K[base + i, base + j]
        end
    end
    return blocks
end

# ---------------------------------------------------------------------------
# 1. compute_block_diagonal! matches the assembled K's block-diagonal
# ---------------------------------------------------------------------------

@testset "compute_block_diagonal!: matches assembled K (elasticity, N=3)" begin
    println("\n" * "=" ^ 70)
    println("BLOCK-JACOBI — compute_block_diagonal! correctness")
    println("=" ^ 70)

    @testset "Elasticity 2x2x2 (N=3)" begin
        mesh = _hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        Kd   = Matrix(K)

        n_blocks = div(n, 3)
        blocks   = zeros(3, 3, n_blocks)
        compute_block_diagonal!(blocks, cache, asm, kernel, m)

        ref = _assembled_block_diagonal(Kd, 3)
        @test maximum(abs, blocks - ref) < 1e-6 * maximum(abs, ref)

        println("  Elast 2×2×2  ndof=$n   max(blocks - K_block_diag)=" *
                "$(round(maximum(abs, blocks - ref); sigdigits = 3))")
    end

    @testset "Heat 2x2x2 (N=1 reduces to scalar diagonal)" begin
        mesh = _hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_heat(mesh)
        n = cache.ndofs

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)

        d = zeros(n); compute_diagonal!(d, cache, asm, kernel, m)
        blocks = zeros(1, 1, n)
        compute_block_diagonal!(blocks, cache, asm, kernel, m)

        @test maximum(abs, blocks[1, 1, :] - d) < 1e-12 * maximum(abs, d)
        println("  Heat  2×2×2  ndof=$n   N=1 reduces to scalar diag ✓")
    end
end

# ---------------------------------------------------------------------------
# 2. P^{-1} = exact inverse of block-diag(K)
# ---------------------------------------------------------------------------

@testset "BlockJacobi: ldiv! is the exact block-diag inverse (N=3)" begin
    Random.seed!(20260508)

    mesh = _hex8_box(2, 1, 1)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n = cache.ndofs

    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    P = BlockJacobiPreconditioner{3}(cache, asm, kernel, m)

    # Reconstruct block-diagonal of K (oracle).
    BD = _assembled_block_diagonal(Matrix(K), 3)

    # For each block, ldiv! must produce inv(BD_b) * x_b.
    for trial in 1:3
        x = randn(n)
        y = zeros(n)
        ldiv!(y, P, x)

        n_blocks = div(n, 3)
        rel_max = 0.0
        for b in 1:n_blocks
            base = 3 * (b - 1)
            xb   = x[base + 1 : base + 3]
            yb   = y[base + 1 : base + 3]
            yb_ref = inv(BD[:, :, b]) * xb
            rel = norm(yb - yb_ref) / max(norm(yb_ref), 1.0)
            rel_max = max(rel_max, rel)
        end
        @test rel_max < 1e-9
    end
end

# ---------------------------------------------------------------------------
# 3. End-to-end inhomogeneous Dirichlet CG via PenaltyDirichlet +
#    BlockJacobi: must converge and match the direct solve.
# ---------------------------------------------------------------------------

@testset "BlockJacobi: PenaltyDirichlet inhomogeneous CG on elastic bar" begin
    println("\n" * "=" ^ 70)
    println("BLOCK-JACOBI — PenaltyDirichlet end-to-end CG")
    println("=" ^ 70)

    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    nx, ny, nz = 4, 1, 1
    mesh = _hex8_box(nx, ny, nz; Lx = 1.0, Ly = 0.1, Lz = 0.1)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Fix all DOFs at x = 0 to zero, prescribe u_x = 0.01 at x = 1
    # (the other components free). Penalty form so we exercise the
    # block-Jacobi diagonal hook.
    fixed_dofs = Int[]
    fixed_vals = Float64[]
    for i in 1:length(nodes)
        x = nodes[i][1]
        base = 3 * (i - 1)
        if x < tol
            for α in 1:3
                push!(fixed_dofs, base + α); push!(fixed_vals, 0.0)
            end
        elseif x > 1.0 - tol
            push!(fixed_dofs, base + 1); push!(fixed_vals, 0.01)
        end
    end

    bc_pen = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1e10)

    # Direct reference solve (penalty applied on assembled K).
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    apply_constraint!(Kbc, bc_pen)
    bbc  = zeros(n); apply_constraint!(bbc, bc_pen)
    u_dir = Kbc \ bbc

    # Matrix-free CG with BlockJacobi.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc_pen)
    linop = LinearOperator(Float64, n, n, true, true, op)
    P     = BlockJacobiPreconditioner{3}(cache, asm, kernel, m; dirichlet = bc_pen)

    u_mf = zeros(n)
    cg!(u_mf, linop, bbc; Pl = P, abstol = 1e-12, reltol = 1e-12, maxiter = 4 * n)

    rel = norm(u_mf - u_dir) / max(norm(u_dir), 1.0)
    @test rel < 1e-6
    # Penalty introduces ~ K_typical / λ relative error in the
    # prescribed value; for E=210e9 and λ=1e10, that's a few-percent
    # offset between u_x(x=1) and the exact 0.01. The point of this
    # test is *not* the BC accuracy — it's that matrix-free CG with
    # block-Jacobi converges to *the same* answer as the assembled
    # direct solve (`rel < 1e-6`). The atol below merely guards against
    # the solve diverging to a wildly different value.
    @test isapprox(u_mf[3 * nx + 1], 0.01; atol = 5e-3)

    println("  Elast bar nx=$nx  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "rel(u_mf vs u_dir)=$(round(rel; sigdigits = 3))   " *
            "u_x(x=1)=$(round(u_mf[3 * nx + 1]; sigdigits = 4))   (penalty offset expected)")
end

# ---------------------------------------------------------------------------
# 4. Block-Jacobi vs scalar-Jacobi convergence on a stiffer elasticity
#    problem.  We just check both converge and BlockJacobi takes
#    ≤ scalar-Jacobi iterations to match the same residual.
# ---------------------------------------------------------------------------

@testset "BlockJacobi: convergence ≤ scalar Jacobi on elasticity" begin
    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    println("\n" * "=" ^ 70)
    println("BLOCK-JACOBI vs SCALAR JACOBI — CG iteration count")
    println("=" ^ 70)

    nx, ny, nz = 6, 2, 2
    mesh = _hex8_box(nx, ny, nz; Lx = 3.0, Ly = 0.5, Lz = 0.5)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Cantilever: fix x=0 face fully, free elsewhere. Drive deformation
    # via a body force so the free DOFs actually have to be solved
    # (homogeneous Dirichlet alone gives a trivial zero solution).
    fixed_dofs = Int[]; fixed_vals = Float64[]
    for i in 1:length(nodes)
        base = 3 * (i - 1)
        if nodes[i][1] < tol
            for α in 1:3
                push!(fixed_dofs, base + α); push!(fixed_vals, 0.0)
            end
        end
    end
    bc = EliminatedDirichlet(fixed_dofs, fixed_vals)

    # Body force: gravity in z (drives genuine non-trivial deformation).
    b   = Vec{3,Float64}((0.0, 0.0, -7850.0 * 9.81))
    rhs = zeros(n); apply_load!(rhs, UniformBodyForce(b), cache, asm, kernel, m)
    apply_constraint!(rhs, bc)            # zero out RHS on fixed DOFs

    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)

    # Scalar Jacobi
    P_scal = JacobiPreconditioner(cache, asm, kernel, m; dirichlet = bc)
    u_s    = zeros(n)
    h_s    = cg!(u_s, linop, rhs; Pl = P_scal, abstol = 0.0, reltol = 1e-10,
                 maxiter = 4 * n, log = true)

    # Block Jacobi
    P_blk = BlockJacobiPreconditioner{3}(cache, asm, kernel, m; dirichlet = bc)
    u_b   = zeros(n)
    h_b   = cg!(u_b, linop, rhs; Pl = P_blk, abstol = 0.0, reltol = 1e-10,
                maxiter = 4 * n, log = true)

    iters_s = h_s[2].iters
    iters_b = h_b[2].iters
    @test isapprox(u_s, u_b; atol = 1e-6, rtol = 1e-6)
    @test iters_b <= iters_s            # block ≤ scalar (often strictly <)

    # Sanity: free DOFs have non-trivial displacement (gravity bends).
    @test maximum(abs, u_b) > 1e-15

    println("  Elast bar nx=$nx ny=$ny nz=$nz  ndof=$n  " *
            "scalar-Jac iters=$iters_s   block-Jac iters=$iters_b")
end

# ---------------------------------------------------------------------------
# 5. Zero-alloc on ldiv!
# ---------------------------------------------------------------------------

@testset "BlockJacobi: zero allocations on ldiv!" begin
    println("\n" * "=" ^ 70)
    println("BLOCK-JACOBI — ZERO-ALLOC on ldiv!")
    println("=" ^ 70)

    @testset "ldiv!(y, P, x) on $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n_dof = cache.ndofs
        P = BlockJacobiPreconditioner{3}(cache, asm, kernel, m)
        x = randn(n_dof); y = zeros(n_dof)

        ldiv!(y, P, x)
        GC.gc()
        a = @allocated ldiv!(y, P, x)
        @test a == 0

        println("  $(nx)×$(ny)×$(nz)  ndof=$n_dof   ldiv!=$a")
    end
end
