# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`ICholPreconditioner` (IC(0)) tests.

Locks in the contract of the no-fill incomplete-Cholesky preconditioner
added in B+++:

  1. **Algebraic correctness** on small dense / tridiagonal SPD matrices:
     `L * L'  ≈  K` exactly when the IC(0) sparsity pattern is full
     (dense lower triangle, tridiagonals, etc.); when fill is dropped,
     it merely approximates `K`.

  2. **Constructor robustness**: the diagonal-shift retry path produces
     a valid factor on a near-indefinite matrix without corrupting the
     input `K`.

  3. **`ldiv!(y, P, x)` is the exact inverse of `L L'`** — i.e. the
     two triangular solves are applied correctly.

  4. **PCG with IC(0) converges in ≤ scalar-Jacobi iterations** on
     stiffer elasticity problems (the canonical benchmark for IC(0)).

  5. **End-to-end matrix-free PenaltyDirichlet pipeline** through the
     `(cache, asm, kernel, mesh; dirichlet)` factory: IC(0) of the
     constrained `K` plus `matrix_free_op` reproduces the assembled
     direct solve.

  6. **Zero allocations on `ldiv!`** (the hot path).
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: extract_system, apply_K!
using JuliaFEM: PenaltyDirichlet, EliminatedDirichlet, apply_constraint!
using JuliaFEM: matrix_free_op
using JuliaFEM: JacobiPreconditioner, ICholPreconditioner
using JuliaFEM: UniformBodyForce, apply_load!
using JuliaFEM: create_elements!
using LinearAlgebra
using SparseArrays
using Tensors
using Random

# ---------------------------------------------------------------------------
# Mesh + setup helpers (same shape as test_block_jacobi.jl)
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

# ---------------------------------------------------------------------------
# 1. Algebraic correctness: L * L' ≈ K on representative SPD matrices
# ---------------------------------------------------------------------------

@testset "ICholPreconditioner: L * L' ≈ K (algebraic correctness)" begin
    println("\n" * "=" ^ 70)
    println("IC(0) — algebraic correctness")
    println("=" ^ 70)

    @testset "1D Laplacian (tridiagonal — IC(0) is exact)" begin
        for n in (5, 10, 50)
            K = spdiagm(-1 => -ones(n - 1),
                         0 =>  2.0 * ones(n),
                         1 => -ones(n - 1))
            P  = ICholPreconditioner(K)
            Kr = Matrix(P.L) * Matrix(P.L)'
            err = norm(Kr - Matrix(K)) / norm(Matrix(K))
            @test err < 1e-12
            println("  n=$n  Laplacian  ‖L Lᵀ - K‖/‖K‖ = $(round(err; sigdigits=3))")
        end
    end

    @testset "Random dense SPD (full lower-triangle pattern)" begin
        Random.seed!(20260508)
        for n in (10, 20, 50)
            M = randn(n, n)
            K = sparse(M * M' + 5.0 * LinearAlgebra.I)
            P = ICholPreconditioner(K)
            Kr = Matrix(P.L) * Matrix(P.L)'
            err = norm(Kr - Matrix(K)) / norm(Matrix(K))
            @test err < 1e-10
            println("  n=$n  dense SPD  ‖L Lᵀ - K‖/‖K‖ = $(round(err; sigdigits=3))")
        end
    end

    @testset "Sparse SPD with dropped fill (IC(0) is approximate)" begin
        Random.seed!(20260508)
        n = 60
        # Banded SPD with bandwidth ~5; off-band entries inside the
        # band trigger fill in exact Cholesky → IC(0) drops it.
        K = spdiagm(0 => 4.0 * ones(n))
        for d in 1:5
            v = -randn(n - d) ./ d
            K += spdiagm(d => v, -d => v)
        end
        # Make safely SPD.
        K += (2.0 * abs(minimum(eigvals(Matrix(K))))) * sparse(LinearAlgebra.I, n, n)
        @test issymmetric(K)
        @test minimum(eigvals(Matrix(K))) > 0

        P  = ICholPreconditioner(K)
        Kr = Matrix(P.L) * Matrix(P.L)'
        err = norm(Kr - Matrix(K)) / norm(Matrix(K))
        # Approximate, so just bound modestly.
        @test err < 5e-1
        # Sparsity preserved.
        @test nnz(P.L) == nnz(LowerTriangular(K))
        println("  banded SPD n=$n  err=$(round(err; sigdigits=3))  " *
                "nnz(L)=$(nnz(P.L)) (= nnz(tril(K)))")
    end
end

# ---------------------------------------------------------------------------
# 2. Constructor robustness: diagonal-shift retry, non-aliasing of input
# ---------------------------------------------------------------------------

@testset "ICholPreconditioner: diagonal-shift retry + input preservation" begin
    println("\n" * "=" ^ 70)
    println("IC(0) — diagonal-shift retry + non-aliasing")
    println("=" ^ 70)

    Random.seed!(20260508)
    n = 30
    # Construct a matrix that is SPD but with a near-zero eigenvalue
    # — IC(0) on its lower triangle may produce a non-positive
    # diagonal mid-factorisation. The shift retry should rescue it.
    M  = randn(n, n)
    K0 = M * M'
    # Slightly perturb diagonal downwards to provoke breakdown.
    K0 -= (0.95 * minimum(eigvals(K0))) * Matrix(LinearAlgebra.I, n, n)
    K  = sparse(K0)

    Korig = copy(K)                        # snapshot for non-aliasing test
    P     = ICholPreconditioner(K)         # must NOT throw

    # Input preserved (no aliasing leaked through):
    @test K.nzval == Korig.nzval
    @test K.rowval == Korig.rowval
    @test K.colptr == Korig.colptr

    # Factor is at least defined on every diagonal (no NaN / Inf).
    L = P.L
    @test all(isfinite, nonzeros(L))
    diagL = [L[i, i] for i in 1:n]
    @test all(>(0), diagL)

    println("  n=$n  retried IC(0) succeeded   min(diag(L))=" *
            "$(round(minimum(diagL); sigdigits=3))")
end

# ---------------------------------------------------------------------------
# 3. ldiv!(y, P, x) is the exact (L * L')^{-1} action
# ---------------------------------------------------------------------------

@testset "ICholPreconditioner: ldiv! agrees with (L * L')^{-1} * x" begin
    Random.seed!(20260508)
    n = 25
    M = randn(n, n)
    K = sparse(M * M' + 2.0 * LinearAlgebra.I)
    P = ICholPreconditioner(K)
    L = Matrix(P.L)

    for trial in 1:3
        x   = randn(n)
        y   = zeros(n)
        ldiv!(y, P, x)
        ref = (L * L') \ x
        rel = norm(y - ref) / max(norm(ref), 1.0)
        @test rel < 1e-9
    end
end

# ---------------------------------------------------------------------------
# 4. PCG with IC(0) ≤ scalar-Jacobi iterations on a stiff elasticity
#    cantilever (the canonical benchmark for IC(0)).
# ---------------------------------------------------------------------------

@testset "IC(0): ≤ scalar-Jacobi iterations on elasticity cantilever" begin
    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    println("\n" * "=" ^ 70)
    println("IC(0) vs SCALAR JACOBI — CG iteration count")
    println("=" ^ 70)

    nx, ny, nz = 12, 3, 3
    mesh = _hex8_box(nx, ny, nz; Lx = 3.0, Ly = 0.5, Lz = 0.5)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Cantilever fixed at x=0, gravity body force.
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

    b   = Vec{3,Float64}((0.0, 0.0, -7850.0 * 9.81))
    rhs = zeros(n); apply_load!(rhs, UniformBodyForce(b), cache, asm, kernel, m)
    apply_constraint!(rhs, bc)

    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)

    # Scalar Jacobi
    P_jac = JacobiPreconditioner(cache, asm, kernel, m; dirichlet = bc)
    u_j   = zeros(n)
    h_j   = cg!(u_j, linop, rhs; Pl = P_jac, abstol = 0.0, reltol = 1e-10,
                maxiter = 4 * n, log = true)

    # IC(0)
    P_ic = ICholPreconditioner(cache, asm, kernel, m; dirichlet = bc)
    u_i  = zeros(n)
    h_i  = cg!(u_i, linop, rhs; Pl = P_ic, abstol = 0.0, reltol = 1e-10,
               maxiter = 4 * n, log = true)

    iters_j = h_j[2].iters
    iters_i = h_i[2].iters
    @test isapprox(u_j, u_i; atol = 1e-6, rtol = 1e-6)
    @test iters_i <= iters_j     # IC(0) ≤ scalar-Jacobi (often much smaller)
    @test maximum(abs, u_i) > 1e-15

    println("  cantilever nx=$nx ny=$ny nz=$nz  ndof=$n  " *
            "scalar-Jac iters=$iters_j   IC(0) iters=$iters_i")
end

# ---------------------------------------------------------------------------
# 5. End-to-end PenaltyDirichlet matrix-free PCG with IC(0) factory
# ---------------------------------------------------------------------------

@testset "IC(0): PenaltyDirichlet inhomogeneous CG matches direct solve" begin
    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    println("\n" * "=" ^ 70)
    println("IC(0) — PenaltyDirichlet end-to-end CG")
    println("=" ^ 70)

    nx, ny, nz = 4, 1, 1
    mesh = _hex8_box(nx, ny, nz; Lx = 1.0, Ly = 0.1, Lz = 0.1)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Same prescribed displacement as test_block_jacobi.jl (penalty form):
    #   u(x=0) = 0, u_x(x=1) = 0.01.
    fixed_dofs = Int[]; fixed_vals = Float64[]
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

    # Direct reference
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    apply_constraint!(Kbc, bc_pen)
    bbc  = zeros(n); apply_constraint!(bbc, bc_pen)
    u_dir = Kbc \ bbc

    # Matrix-free CG with IC(0)
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc_pen)
    linop = LinearOperator(Float64, n, n, true, true, op)
    P     = ICholPreconditioner(cache, asm, kernel, m; dirichlet = bc_pen)

    u_mf = zeros(n)
    h    = cg!(u_mf, linop, bbc; Pl = P, abstol = 1e-12, reltol = 1e-12,
               maxiter = 4 * n, log = true)

    rel = norm(u_mf - u_dir) / max(norm(u_dir), 1.0)
    @test rel < 1e-6
    @test isapprox(u_mf[3 * nx + 1], 0.01; atol = 5e-3)

    println("  Elast bar nx=$nx  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "iters=$(h[2].iters)   rel(u_mf vs u_dir)=" *
            "$(round(rel; sigdigits = 3))   u_x(x=1)=" *
            "$(round(u_mf[3 * nx + 1]; sigdigits = 4))   (penalty offset expected)")
end

# ---------------------------------------------------------------------------
# 6. Zero allocations on the hot path (ldiv!)
# ---------------------------------------------------------------------------

@testset "IC(0): zero allocations on ldiv!" begin
    println("\n" * "=" ^ 70)
    println("IC(0) — ZERO-ALLOC on ldiv!")
    println("=" ^ 70)

    @testset "ldiv!(y, P, x) on $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n_dof = cache.ndofs
        P = ICholPreconditioner(cache, asm, kernel, m)
        x = randn(n_dof); y = zeros(n_dof)

        ldiv!(y, P, x)
        GC.gc()
        a = @allocated ldiv!(y, P, x)
        @test a == 0

        println("  $(nx)×$(ny)×$(nz)  ndof=$n_dof   ldiv!=$a")
    end
end
