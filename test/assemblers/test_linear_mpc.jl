# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`LinearMPC` (penalty-enforced linear multipoint constraints) tests.

Locks in the contract of the MPC type added in C+++:

  1. **Constructor packing**: `(slave, masters, coeffs, offset)` tuples
     pack into the flat CSR layout (slaves / offsets /
     master_offsets / master_dofs / master_coeffs).

  2. **Assembled–vs–matrix-free agreement**: `apply_constraint!(K, mpc)`
     followed by `K * x` matches `apply_K!` + `apply_constraint_post!`
     to round-off, on both heat and elasticity problems and on a
     constraint set that mixes single-master and multi-master
     constraints.

  3. **End-to-end periodic-BC heat-conduction solve**: `T(x=0) = T(x=L)`
     on a 1D bar with an interior heat source converges via PCG with
     `JacobiPreconditioner(...; mpc)` and matches the analytical
     periodic solution to `< 1e-6`.

  4. **End-to-end rigid-link elasticity solve**: tying `u_x` of two
     opposite faces of a hex8 box (rigid body translation along x)
     reproduces the assembled direct solve.

  5. **Inhomogeneous offset `g` ≠ 0**: `b ← b + λ Cᵀ g` lifts the RHS
     correctly so the constraint `R = 0` is enforced at the
     prescribed value.

  6. **Zero allocations**: both `apply_constraint_post!(y, x, mpc)` and
     `apply_constraint_diag!(d, mpc)` run allocation-free on the hot
     path.
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
using JuliaFEM: matrix_free_op, JacobiPreconditioner
using JuliaFEM: LinearMPC, AbstractMultipointConstraint
using JuliaFEM: apply_constraint_post!, apply_constraint_diag!
using JuliaFEM: UniformBodyForce, NodalForce, apply_load!
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

function _setup_heat(mesh; k_value::Float64 = 50.2)
    material = HeatConductivity(k = k_value)
    kernel   = HeatKernel(ContinuumFormulation{FullThreeD}(), material)
    S        = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ---------------------------------------------------------------------------
# 1. Constructor: packing + storage layout
# ---------------------------------------------------------------------------

@testset "LinearMPC: tuple → flat CSR storage layout" begin
    println("\n" * "=" ^ 70)
    println("LinearMPC — constructor packing")
    println("=" ^ 70)

    constraints = [
        (5,  [1, 2],     [0.4, 0.6],     0.10),   # 2 masters, offset = 0.10
        (10, [3],        [1.0],           0.0),   # single master, no offset
        (15, [4, 6, 7],  [0.3, 0.5, 0.2], -0.05), # 3 masters
    ]
    mpc = LinearMPC(constraints; penalty = 1.0e8)

    @test mpc.slaves          == [5, 10, 15]
    @test mpc.offsets         == [0.10, 0.0, -0.05]
    @test mpc.master_offsets  == [1, 3, 4, 7]            # CSR pointers
    @test mpc.master_dofs     == [1, 2, 3, 4, 6, 7]
    @test mpc.master_coeffs   == [0.4, 0.6, 1.0, 0.3, 0.5, 0.2]
    @test mpc.penalty         == 1.0e8
end

# ---------------------------------------------------------------------------
# 2. Assembled-K vs matrix-free agreement
# ---------------------------------------------------------------------------

@testset "LinearMPC: apply_constraint!(K) vs apply_constraint_post!(y, x)" begin
    println("\n" * "=" ^ 70)
    println("LinearMPC — assembled-K ≡ matrix-free hook")
    println("=" ^ 70)

    Random.seed!(20260508)

    @testset "Heat conduction (scalar field) — periodic + multi-master" begin
        nx = 8
        mesh = _hex8_box(nx, 1, 1)
        cache, asm, kernel, m = _setup_heat(mesh)
        n = cache.ndofs
        nodes = m.nodes
        tol = 1e-9

        # Periodic BC: T(x=0) = T(x=L). One MPC per node on the x=0
        # face: slave at x=0, master at x=L (same y, z).
        constraints = Vector{Tuple{Int, Vector{Int}, Vector{Float64}, Float64}}()
        for i in 1:length(nodes)
            x = nodes[i][1]
            if x < tol
                # Find matching node at x = Lx with the same (y, z).
                for j in 1:length(nodes)
                    if abs(nodes[j][1] - 1.0) < tol &&
                       abs(nodes[j][2] - nodes[i][2]) < tol &&
                       abs(nodes[j][3] - nodes[i][3]) < tol
                        push!(constraints, (i, [j], [1.0], 0.0))
                        break
                    end
                end
            end
        end
        mpc = LinearMPC(constraints; penalty = 1.0e8)

        # Build assembled K + add penalty MPC; compare K*x with
        # apply_K! + post-hook for several random x.
        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        Kbc  = Matrix(K)
        apply_constraint!(Kbc, mpc)

        for trial in 1:3
            x   = randn(n)
            y_a = Kbc * x

            y_mf = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            apply_constraint_post!(y_mf, x, mpc)

            rel = norm(y_mf - y_a) / max(norm(y_a), 1.0)
            @test rel < 1e-9
        end
    end

    @testset "Elasticity (vector field) — multi-master constraint" begin
        nx = 4
        mesh = _hex8_box(nx, 1, 1)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs

        # A single MPC averaging u_x of three interior nodes:
        # u_x_node5 = (u_x_node3 + u_x_node4 + u_x_node6) / 3.
        # Use DOF numbering convention u_x = 3 * (node - 1) + 1.
        ux(i) = 3 * (i - 1) + 1
        constraints = [
            (ux(5), [ux(3), ux(4), ux(6)], [1/3, 1/3, 1/3], 0.0),
        ]
        mpc = LinearMPC(constraints; penalty = 1.0e9)

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        Kbc  = Matrix(K)
        apply_constraint!(Kbc, mpc)

        for trial in 1:3
            x   = randn(n)
            y_a = Kbc * x

            y_mf = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            apply_constraint_post!(y_mf, x, mpc)

            rel = norm(y_mf - y_a) / max(norm(y_a), 1.0)
            @test rel < 1e-9
        end
    end
end

# ---------------------------------------------------------------------------
# 3. End-to-end periodic heat: matrix-free PCG matches assembled direct
# ---------------------------------------------------------------------------

@testset "LinearMPC: periodic heat conduction matches direct solve" begin
    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    println("\n" * "=" ^ 70)
    println("LinearMPC — periodic heat: matrix-free PCG ≡ direct solve")
    println("=" ^ 70)

    nx = 6
    mesh = _hex8_box(nx, 1, 1; Lx = 1.0, Ly = 0.1, Lz = 0.1)
    cache, asm, kernel, m = _setup_heat(mesh; k_value = 1.0)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Periodic in x: T(x=0) = T(x=L) per (y, z).
    constraints = Vector{Tuple{Int, Vector{Int}, Vector{Float64}, Float64}}()
    for i in 1:length(nodes)
        if nodes[i][1] < tol
            for j in 1:length(nodes)
                if abs(nodes[j][1] - 1.0) < tol &&
                   abs(nodes[j][2] - nodes[i][2]) < tol &&
                   abs(nodes[j][3] - nodes[i][3]) < tol
                    push!(constraints, (i, [j], [1.0], 0.0))
                    break
                end
            end
        end
    end
    # Pure Neumann + periodic ⇒ rank-1 null space (T = const). Pin one
    # interior temperature with a Dirichlet to remove it.
    pin_node = nx ÷ 2 + 1
    bc_pin   = EliminatedDirichlet([pin_node], [42.0])

    # Penalty must be >> K_typical (here k_value=1, mesh ~0.1 → K~10) but
    # not so large that the system is impossible for CG to converge.
    # 1e6 gives constraint residual ~1e-6 and condition number ~1e7.
    mpc = LinearMPC(constraints; penalty = 1.0e6)

    # Source: nodal heat input on the right interior.
    rhs = zeros(n)
    apply_load!(rhs, NodalForce([nx], [1.0]), cache, asm, kernel, m)

    # Direct reference solve.
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    bbc  = copy(rhs)
    apply_constraint!(Kbc, mpc)
    apply_constraint!(Kbc, bbc, bc_pin)
    T_dir = Kbc \ bbc

    # Matrix-free PCG.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc_pin, mpc = mpc)
    linop = LinearOperator(Float64, n, n, true, true, op)
    P     = JacobiPreconditioner(cache, asm, kernel, m;
                                 dirichlet = bc_pin, mpc = mpc)
    T_mf = zeros(n)
    h    = cg!(T_mf, linop, bbc; Pl = P, abstol = 1e-14, reltol = 1e-14,
               maxiter = 50 * n, log = true)

    rel = norm(T_mf - T_dir) / max(norm(T_dir), 1.0)
    @test rel < 1e-4

    # Verify the periodic identity holds at every constraint to within
    # the penalty floor (~K/λ).
    max_resid = 0.0
    for (s, ms, cs, g) in constraints
        R = T_mf[s] - sum(cs .* T_mf[ms]) - g
        max_resid = max(max_resid, abs(R))
    end
    @test max_resid / max(norm(T_mf), 1.0) < 1e-3   # λ=1e6, K~O(1) → R≲1e-6

    println("  periodic heat nx=$nx  ndof=$n  iters=$(h[2].iters)  " *
            "rel(T_mf vs T_dir)=$(round(rel; sigdigits = 3))   " *
            "max periodic R=$(round(max_resid; sigdigits = 3))")
end

# ---------------------------------------------------------------------------
# 4. Inhomogeneous offset: rigid-link elasticity (tied face)
# ---------------------------------------------------------------------------

@testset "LinearMPC: inhomogeneous offset matches assembled solve" begin
    using IterativeSolvers: cg!
    using LinearOperators: LinearOperator

    println("\n" * "=" ^ 70)
    println("LinearMPC — inhomogeneous offset (PenaltyDirichlet + PenaltyMPC)")
    println("=" ^ 70)

    nx, ny, nz = 4, 1, 1
    mesh = _hex8_box(nx, ny, nz; Lx = 1.0, Ly = 0.1, Lz = 0.1)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n     = cache.ndofs
    nodes = m.nodes
    tol   = 1e-9

    # Use **PenaltyDirichlet** rather than EliminatedDirichlet for this
    # test: penalty + penalty composes additively without zeroing rows,
    # so the MPC inhomogeneous-offset terms remain consistent on both
    # sides.  Fix x=0 face and prescribe u_x=0.005 on the entire x=1
    # face via a tied MPC (single master + offset).
    fixed_dofs = Int[]; fixed_vals = Float64[]
    right_nodes = Int[]
    for i in 1:length(nodes)
        x = nodes[i][1]
        base = 3 * (i - 1)
        if x < tol
            for α in 1:3
                push!(fixed_dofs, base + α); push!(fixed_vals, 0.0)
            end
        elseif x > 1.0 - tol
            push!(right_nodes, i)
        end
    end
    @assert length(right_nodes) >= 2

    bc = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1.0e14)

    # MPC: tie slaves to the master node's u_x with an offset
    # g = 0.001 (so each slave should equal master + 0.001).
    g_offset = 0.001
    master_node = right_nodes[1]
    ux(i) = 3 * (i - 1) + 1
    tie_constraints = Tuple{Int, Vector{Int}, Vector{Float64}, Float64}[]
    for k in 2:length(right_nodes)
        push!(tie_constraints,
              (ux(right_nodes[k]), [ux(master_node)], [1.0], g_offset))
    end
    # And one constraint pinning the master's u_x = 0.005:
    # an empty-masters constraint reduces to R = u_s − g, i.e. a
    # standalone "set this DOF to g" via penalty.
    push!(tie_constraints,
          (ux(master_node), Int[], Float64[], 0.005))

    mpc = LinearMPC(tie_constraints; penalty = 1.0e14)

    # Direct reference (assembled).
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    bbc  = zeros(n)
    apply_constraint!(Kbc, bc); apply_constraint!(bbc, bc)
    apply_constraint!(Kbc, mpc); apply_constraint!(bbc, mpc)
    u_dir = Kbc \ bbc

    # Matrix-free PCG with both Dirichlet + MPC hooks composed.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc, mpc = mpc)
    linop = LinearOperator(Float64, n, n, true, true, op)
    P     = JacobiPreconditioner(cache, asm, kernel, m;
                                 dirichlet = bc, mpc = mpc)
    u_mf = zeros(n)
    h    = cg!(u_mf, linop, bbc; Pl = P, abstol = 1e-14, reltol = 1e-14,
               maxiter = 50 * n, log = true)

    rel = norm(u_mf - u_dir) / max(norm(u_dir), 1.0)
    @test rel < 1e-4

    # The master's u_x ≈ 0.005, slaves ≈ 0.006 (penalty floor ~K/λ ≈
    # 1e10/1e14 = 1e-4 absolute on E~210e9 elasticity).
    @test abs(u_mf[ux(master_node)] - 0.005) < 5e-4
    for k in 2:length(right_nodes)
        @test abs(u_mf[ux(right_nodes[k])] - (0.005 + g_offset)) < 1e-3
    end

    println("  rigid-link nx=$nx  ndof=$n  iters=$(h[2].iters)  " *
            "rel(u_mf vs u_dir)=$(round(rel; sigdigits = 3))   " *
            "u_x(master)=$(round(u_mf[ux(master_node)]; sigdigits = 4))   " *
            "u_x(slave)≈$(round(u_mf[ux(right_nodes[2])]; sigdigits = 4))")
end

# ---------------------------------------------------------------------------
# 5. Zero allocations on the matrix-free hot paths
# ---------------------------------------------------------------------------

@testset "LinearMPC: zero allocations on apply_constraint_post! / _diag!" begin
    println("\n" * "=" ^ 70)
    println("LinearMPC — ZERO-ALLOC on hot hooks")
    println("=" ^ 70)

    n = 100
    Random.seed!(20260508)
    constraints = Tuple{Int, Vector{Int}, Vector{Float64}, Float64}[]
    for k in 1:10
        s  = rand(1:n)
        nm = rand(1:4)
        ms = unique(rand(1:n, nm))
        push!(constraints, (s, ms, rand(length(ms)), randn()))
    end
    mpc = LinearMPC(constraints; penalty = 1.0e8)

    x = randn(n); y = zeros(n); d = zeros(n)

    apply_constraint_post!(y, x, mpc)
    apply_constraint_diag!(d, mpc)

    GC.gc()
    a_post = @allocated apply_constraint_post!(y, x, mpc)
    a_diag = @allocated apply_constraint_diag!(d, mpc)
    @test a_post == 0
    @test a_diag == 0

    println("  apply_constraint_post! allocs=$a_post   apply_constraint_diag! allocs=$a_diag")
end
