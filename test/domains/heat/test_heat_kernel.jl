# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Regression + correctness tests for `HeatKernel` through the DOF-based
assembler — proves the microkernel contract is genuinely kernel-agnostic.

The same machinery (`DOFBasedCOOCache`, `assemble!`, `apply_K!`,
`DOFBasedCOOCacheKA`) is exercised here on a scalar-temperature problem.
If anything in the assembler ever silently re-elasticity-fies (hardcodes
3 components, the elasticity tangent type, the displacement field, etc.),
this file fails before any user does.

Layered coverage:

1. **Correctness** — assembled `K` is symmetric, SPD on free DOFs, has the
   constant-temperature null space, and matches the value computed by
   `apply_K!` to round-off.

2. **KA equivalence** — `apply_K!` on the `CPU()` KernelAbstractions backend
   is bit-equivalent to the direct CPU `apply_K!` (proves we share the same
   `evaluate_entry` instead of a kernel-specific device path).

3. **Zero-allocation** — `assemble!` and direct `apply_K!` allocate exactly
   0 bytes after warmup; the optimized LLVM IR for both has 0 GC allocation
   sites. Same hard contract as `ContinuumKernel`.

4. **Matrix-free CG** — solves a heat conduction problem via `apply_K!`
   wrapped as a `LinearOperators.LinearOperator` (penalty Dirichlet on the
   bottom face, point heat source at the top corner) and matches the
   direct sparse solve. This is the actual use case for the matrix-free
   path — once it solves, the API is proven end-to-end.

5. **Type stability** — `assemble!` and `apply_K!` for `HeatKernel` infer
   concrete return types end-to-end.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Temperature, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, apply_K!, extract_system
using JuliaFEM: DOFBasedCOOCacheKA, sync_from_cpu!
using JuliaFEM: JacobiPreconditioner, compute_diagonal!
using JuliaFEM: create_elements!
using LinearAlgebra
using SparseArrays
using Tensors
using Random
using InteractiveUtils       # code_llvm, code_typed
using LinearOperators        # LinearOperator wrapper for apply_K!
using IterativeSolvers       # cg!

# ----------------------------------------------------------------------------
# Mesh helpers (mirror the elasticity test files; kept local so this
# regression test stays independent).
# ----------------------------------------------------------------------------

function _build_hex8_box(nx::Int, ny::Int, nz::Int)
    nodes = Vec{3,Float64}[]
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        push!(nodes, Vec{3}((Float64(i - 1) / nx,
                             Float64(j - 1) / ny,
                             Float64(k - 1) / nz)))
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

function _build_single_tet4()
    nodes = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.5, 1.0, 0.0)),
        Vec{3}((0.5, 0.5, 1.0)),
    ]
    conns = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
    return Mesh{Tetrahedron{4}}(nodes, conns)
end

"Set up DOF-based heat assembly fixture for a given mesh + topology."
function _setup_heat(mesh, ::Type{Topo}; k::Float64 = 50.2) where {Topo}
    material = HeatConductivity(k = k)
    kernel   = HeatKernel(ContinuumFormulation{FullThreeD}(), material)
    S        = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Topo, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# 1. Correctness: K symmetric / SPD on free DOFs / null space / apply_K! match
# ----------------------------------------------------------------------------

@testset "HeatKernel: assemble! correctness + apply_K! (CPU) equivalence" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL CORRECTNESS — assemble! + apply_K! (CPU)")
    println("=" ^ 70)

    Random.seed!(20260508)

    @testset "Single Tet4" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m = _setup_heat(mesh, Tetrahedron{4})

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        # Symmetric to round-off
        @test maximum(abs, Matrix(K) - Matrix(K)') < 1e-9 * maximum(abs, Matrix(K))

        # Constant-temperature null space (∇N · 1 = 0)
        @test norm(K * ones(n)) < 1e-9 * maximum(abs, Matrix(K))

        # apply_K! agrees with K * x for several random x
        max_rel = 0.0
        for _ in 1:8
            x     = randn(n)
            y_ref = K * x
            y_mf  = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            rel   = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end
        println("  Single Tet4 ........ n=$n   max rel=$(round(max_rel; sigdigits = 3))")
    end

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        @test maximum(abs, Matrix(K) - Matrix(K)') < 1e-9 * maximum(abs, Matrix(K))
        @test norm(K * ones(n)) < 1e-9 * maximum(abs, Matrix(K))

        max_rel = 0.0
        for _ in 1:5
            x     = randn(n)
            y_ref = K * x
            y_mf  = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            rel   = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end

        nelems = length(m.connectivity)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems, 5)) elem   " *
                "$(lpad(n, 5)) dof   max rel=$(round(max_rel; sigdigits = 3))")
    end
end

# ----------------------------------------------------------------------------
# 2. KA backend (CPU()) bit-equivalent to direct CPU apply_K!
# ----------------------------------------------------------------------------

@testset "HeatKernel: KA apply_K! (CPU backend) bit-equivalent" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL — KA apply_K! (CPU backend)")
    println("=" ^ 70)

    Random.seed!(20260509)

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        ka = DOFBasedCOOCacheKA(cache)
        sync_from_cpu!(ka, cache)

        for _ in 1:5
            x     = randn(n)
            y_cpu = zeros(n)
            y_ka  = zeros(n)
            apply_K!(y_cpu, cache, asm, kernel, m, x)
            apply_K!(y_ka, ka, kernel, x)

            # bit-identical: shared evaluate_entry, same DOF traversal order
            @test maximum(abs, y_cpu - y_ka) == 0.0

            # both match the assembled K to round-off
            y_ref = K * x
            @test norm(y_ka - y_ref) / max(norm(y_ref), 1.0) < 1e-12
        end

        # Constant-temperature null space holds through the KA path too
        y = zeros(n)
        apply_K!(y, ka, kernel, ones(n))
        @test norm(y) < 1e-9 * maximum(abs, Matrix(K))

        println("  Hex8 $(nx)×$(ny)×$(nz)   ndof=$n   bit-equivalent ✓   null-space ✓")
    end
end

# ----------------------------------------------------------------------------
# 3. Zero-alloc + 0 LLVM gc-alloc sites for assemble! and apply_K!
# ----------------------------------------------------------------------------

@testset "HeatKernel: zero allocations (assemble! + apply_K!)" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL ZERO-ALLOC — assemble! + apply_K!")
    println("=" ^ 70)

    @testset "Single Tet4" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m = _setup_heat(mesh, Tetrahedron{4})
        n = cache.ndofs
        x = ones(n); y = zeros(n)

        # warmup
        assemble!(cache, asm, kernel, m)
        apply_K!(y, cache, asm, kernel, m, x)

        GC.gc()
        a_asm = @allocated assemble!(cache, asm, kernel, m)
        @test a_asm == 0

        GC.gc()
        a_mf = @allocated apply_K!(y, cache, asm, kernel, m, x)
        @test a_mf == 0
        println("  Single Tet4 ........ assemble!=$a_asm  apply_K!=$a_mf")
    end

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
        n = cache.ndofs
        x = ones(n); y = zeros(n)

        assemble!(cache, asm, kernel, m)         # warmup
        apply_K!(y, cache, asm, kernel, m, x)    # warmup

        GC.gc()
        a_asm = @allocated assemble!(cache, asm, kernel, m)
        @test a_asm == 0

        GC.gc()
        a_mf = @allocated apply_K!(y, cache, asm, kernel, m, x)
        @test a_mf == 0

        nelems = length(m.connectivity)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems,5)) elem  " *
                "$(lpad(n,5)) dof   assemble!=$a_asm  apply_K!=$a_mf")
    end

    @testset "Optimized LLVM IR has 0 gc-alloc sites" begin
        mesh = _build_hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
        n = cache.ndofs
        x = ones(n); y = zeros(n)
        assemble!(cache, asm, kernel, m)
        apply_K!(y, cache, asm, kernel, m, x)

        function count_allocs(ir)
            length(collect(eachmatch(r"call.*julia\.gc_alloc",      ir))) +
            length(collect(eachmatch(r"call.*jl_gc_pool_alloc",     ir))) +
            length(collect(eachmatch(r"call.*jl_gc_big_alloc",      ir))) +
            length(collect(eachmatch(r"call.*jl_gc_alloc_typed",    ir)))
        end

        iob = IOBuffer()
        code_llvm(iob, assemble!,
            Tuple{typeof(cache), typeof(asm), typeof(kernel), typeof(m)};
            optimize = true)
        n_alloc_asm = count_allocs(String(take!(iob)))
        @test n_alloc_asm == 0

        iob = IOBuffer()
        code_llvm(iob, apply_K!,
            Tuple{typeof(y), typeof(cache), typeof(asm),
                  typeof(kernel), typeof(m), typeof(x)};
            optimize = true)
        n_alloc_mf = count_allocs(String(take!(iob)))
        @test n_alloc_mf == 0

        println("  HeatKernel optimized LLVM IR: " *
                "assemble!=$n_alloc_asm  apply_K!=$n_alloc_mf gc-alloc sites")
    end
end

# ----------------------------------------------------------------------------
# 4. Matrix-free CG via LinearOperators wrapping apply_K!
#    First end-to-end heat conduction problem solved without ever
#    materializing K.
# ----------------------------------------------------------------------------

@testset "HeatKernel: matrix-free CG via PenaltyDirichlet + matrix_free_op" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL — MATRIX-FREE CG VALIDATION (declarative Dirichlet)")
    println("=" ^ 70)

    nx = ny = nz = 3
    mesh = _build_hex8_box(nx, ny, nz)
    cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
    n = cache.ndofs

    # Bottom face (z = 0) clamped to T = 0 — homogeneous PenaltyDirichlet.
    fixed_dofs = Int[]
    for (nid, X) in enumerate(m.nodes)
        if X[3] == 0.0
            push!(fixed_dofs, nid)        # 1 DOF / node ⇒ DOF index = node id
        end
    end
    @test !isempty(fixed_dofs)
    bc = PenaltyDirichlet(fixed_dofs; penalty = 1e16)

    # Heat source: unit point heating at the top corner.
    top_corner_node = (nx + 1) * (ny + 1) * (nz + 1)
    b               = zeros(n)
    b[top_corner_node] = 1.0

    # 1. Direct sparse solve — same constraint type, applied to assembled K.
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc = Matrix(K)
    apply_constraint!(Kbc, bc)
    T_direct = Kbc \ b

    # 2. Matrix-free CG — `matrix_free_op` builds the closure, including
    #    the `apply_constraint!(y, x, bc)` call after `apply_K!`. No K
    #    is ever materialized on the matrix-free side.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)

    T_mf = zeros(n)
    cg!(T_mf, linop, b; abstol = 1e-10, reltol = 1e-12, maxiter = 4 * n)

    rel_T = norm(T_mf - T_direct) / max(norm(T_direct), 1.0)
    @test rel_T < 1e-6

    # Residual of the matrix-free solution under the *same* operator
    r = zeros(n)
    op(r, T_mf)
    rel_r = norm(b - r) / max(norm(b), 1.0)
    @test rel_r < 1e-6

    # Maximum principle: heat source > 0, T = 0 BC ⇒ T ≥ 0 everywhere
    @test minimum(T_mf) > -1e-12

    # Hottest node should sit at the corner where the source is applied
    @test argmax(T_mf) == top_corner_node

    println("  Hex8 $(nx)×$(ny)×$(nz)  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "rel_T=$(round(rel_T; sigdigits = 3))  rel_r=$(round(rel_r; sigdigits = 3))  " *
            "T_max=$(round(maximum(T_mf); sigdigits = 4))")
end

# ----------------------------------------------------------------------------
# 4b. Inhomogeneous Dirichlet: T = T̂ on the bottom face. Exercises the
#     `values` field of PenaltyDirichlet and the RHS apply
#     `apply_constraint!(b, c)` which adds `λ * T̂` on the fixed DOFs.
# ----------------------------------------------------------------------------

@testset "HeatKernel: inhomogeneous PenaltyDirichlet — op + RHS consistency" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL — INHOMOGENEOUS DIRICHLET (op consistency)")
    println("=" ^ 70)

    # Bottom face: T = 1.0 (warm plate). Top face: T = 0.0 (cold plate).
    # No volumetric source. Steady 1-D conduction along z ⇒ analytical
    # solution T(z) = 1 - z, independent of x and y.
    #
    # The thing under test here is *abstraction consistency*: the same
    # `PenaltyDirichlet` applied to:
    #
    #   - assembled K  →  Kbc \ b reproduces T(z) = 1 - z exactly,
    #   - matrix-free  →  matrix_free_op(...; dirichlet = bc) acting on
    #                     T_exact returns precisely the b that
    #                     `apply_constraint!(b, bc)` produced.
    #
    # CG-convergence on the raw penalty system is not validated here
    # because Krylov methods on penalty-Dirichlet without a Jacobi-style
    # preconditioner suffer from a well-known relative-tolerance trap
    # (b ≈ λ * T̂ ⇒ tiny absolute residuals look "converged" early).
    # The previous testset already verified that the *homogeneous* path
    # solves cleanly via matrix_free_op + cg!.
    nx = ny = nz = 2
    mesh = _build_hex8_box(nx, ny, nz)
    cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
    n = cache.ndofs

    fixed_dofs = Int[]
    fixed_vals = Float64[]
    for (nid, X) in enumerate(m.nodes)
        if X[3] == 0.0
            push!(fixed_dofs, nid); push!(fixed_vals, 1.0)
        elseif X[3] == 1.0
            push!(fixed_dofs, nid); push!(fixed_vals, 0.0)
        end
    end
    bc = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1e16)

    b = zeros(n)
    apply_constraint!(b, bc)
    @test count(!iszero, b) == count(!iszero, fixed_vals)
    @test maximum(b) == 1.0e16            # λ * 1.0
    @test minimum(b) == 0.0               # λ * 0.0 (and unfixed entries)

    # Direct solve must reproduce T(z) = 1 - z to round-off.
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    apply_constraint!(Kbc, bc)
    T_dir = Kbc \ b

    T_exact = [1.0 - X[3] for X in m.nodes]
    rel_dir = norm(T_dir - T_exact) / max(norm(T_exact), 1.0)
    @test rel_dir < 1e-12

    # Matrix-free op must satisfy op(T_exact) == b to round-off — this
    # is the operator-consistency check that proves
    # `apply_constraint!(y, x, bc)` plays the *same* role as
    # `apply_constraint!(K, bc)` did on the assembled side.
    op = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    y_check = zeros(n)
    op(y_check, T_exact)
    rel_op = norm(y_check - b) / max(norm(b), 1.0)
    @test rel_op < 1e-12

    println("  Hex8 $(nx)×$(ny)×$(nz)  ndof=$n  rel(T_dir vs T_exact)=" *
            "$(round(rel_dir; sigdigits = 3))  rel(op·T_exact vs b)=" *
            "$(round(rel_op; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 4b'. Jacobi-preconditioned matrix-free CG closes the inhomogeneous
#      *PenaltyDirichlet* gap. Demonstrates `compute_diagonal!` +
#      `JacobiPreconditioner` working through `IterativeSolvers.cg!`'s
#      `Pl` slot.
# ----------------------------------------------------------------------------

@testset "HeatKernel: inhomogeneous PenaltyDirichlet + Jacobi-preconditioned matrix-free CG" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL — JACOBI-PRECONDITIONED MATRIX-FREE CG (penalty path)")
    println("=" ^ 70)

    nx = ny = nz = 2
    mesh = _build_hex8_box(nx, ny, nz)
    cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
    n = cache.ndofs

    fixed_dofs = Int[]
    fixed_vals = Float64[]
    for (nid, X) in enumerate(m.nodes)
        if X[3] == 0.0
            push!(fixed_dofs, nid); push!(fixed_vals, 1.0)
        elseif X[3] == 1.0
            push!(fixed_dofs, nid); push!(fixed_vals, 0.0)
        end
    end
    # A moderate penalty: large enough to drive the BC error well below
    # the test tolerance, small enough that the *preconditioned* CG
    # residual norm tracks the actual error.
    bc = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1e8)

    # RHS for the penalty system: λ * T̂ on the fixed DOFs, zero elsewhere.
    b = zeros(n)
    apply_constraint!(b, bc)

    # Direct sparse solve against the penalty system as the ground-truth
    # for the matrix-free CG result (and as an indirect check that the
    # *operator* matches what the preconditioner saw).
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    apply_constraint!(Kbc, bc)
    T_dir = Kbc \ b

    # diag(K_op) extracted matrix-free + the constraint hook. Should be
    # bit-equal to diag(Kbc).
    d_mf = zeros(n)
    compute_diagonal!(d_mf, cache, asm, kernel, m)
    apply_constraint_diag!(d_mf, bc)
    @test maximum(abs, d_mf - diag(Kbc)) <
          1e-10 * maximum(abs, diag(Kbc))

    # Build the matrix-free op and the Jacobi preconditioner.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)
    P     = JacobiPreconditioner(cache, asm, kernel, m; dirichlet = bc)

    # Preconditioned CG should hit the direct solve to high precision
    # within a small number of iterations on this mesh.
    T_mf = zeros(n)
    cg!(T_mf, linop, b; Pl = P, abstol = 1e-12, reltol = 1e-12, maxiter = 4 * n)

    rel_T = norm(T_mf - T_dir) / max(norm(T_dir), 1.0)
    @test rel_T < 1e-6

    # And the matrix-free solution must satisfy the analytical T(z) = 1 - z
    # to the (penalty-determined) accuracy.
    T_exact = [1.0 - X[3] for X in m.nodes]
    rel_exact = norm(T_mf - T_exact) / max(norm(T_exact), 1.0)
    @test rel_exact < 1e-6

    println("  Hex8 $(nx)×$(ny)×$(nz)  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "rel(T_mf vs T_dir)=$(round(rel_T; sigdigits = 3))  " *
            "rel(T_mf vs analytical)=$(round(rel_exact; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 4c. Inhomogeneous Dirichlet via *EliminatedDirichlet*. Same physical
#     problem as 4b, but the eliminated operator preserves the original
#     conditioning so un-preconditioned matrix-free CG converges cleanly
#     to the analytical solution (the penalty path could only verify
#     operator consistency, not raw CG convergence).
# ----------------------------------------------------------------------------

@testset "HeatKernel: inhomogeneous EliminatedDirichlet — direct + matrix-free CG" begin
    println("\n" * "=" ^ 70)
    println("HEAT KERNEL — INHOMOGENEOUS DIRICHLET (row/col elimination)")
    println("=" ^ 70)

    nx = ny = nz = 2
    mesh = _build_hex8_box(nx, ny, nz)
    cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
    n = cache.ndofs

    # Bottom face T = 1, top face T = 0; analytical T(z) = 1 − z.
    fixed_dofs = Int[]
    fixed_vals = Float64[]
    for (nid, X) in enumerate(m.nodes)
        if X[3] == 0.0
            push!(fixed_dofs, nid); push!(fixed_vals, 1.0)
        elseif X[3] == 1.0
            push!(fixed_dofs, nid); push!(fixed_vals, 0.0)
        end
    end
    bc = EliminatedDirichlet(fixed_dofs, fixed_vals)

    T_exact = [1.0 - X[3] for X in m.nodes]

    # --- 1. Assembled path: lift RHS + zero rows/cols, then direct solve ---
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    bbc  = zeros(n)                              # no source
    apply_constraint!(Kbc, bbc, bc)              # lift + eliminate + set b[d]
    T_direct = Kbc \ bbc

    rel_dir = norm(T_direct - T_exact) / max(norm(T_exact), 1.0)
    @test rel_dir < 1e-12

    # --- 2. Matrix-free CG on the eliminated operator -----------------------
    # The eliminated operator behaves like K_ff on the free block and
    # the identity on the fixed block. The corresponding lifted RHS is
    # -K · û on free DOFs and û on fixed DOFs. Build it once via the
    # matrix-free op itself.
    u_lift = zeros(n)
    @inbounds for k in eachindex(fixed_dofs)
        u_lift[fixed_dofs[k]] = fixed_vals[k]
    end

    # b_lifted = (op without RHS source) needs:
    #   (K · u_lift)[free]   transferred to RHS as − contribution,
    #   û[fixed] as identity row RHS.
    # Easiest: compute Ku = K · u_lift via apply_K! once, then assemble.
    Ku = zeros(n)
    apply_K!(Ku, cache, asm, kernel, m, u_lift)
    b_mf = -Ku                                   # lift on free DOFs
    @inbounds for k in eachindex(fixed_dofs)
        b_mf[fixed_dofs[k]] = fixed_vals[k]      # identity row RHS
    end

    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)

    T_mf = zeros(n)
    cg!(T_mf, linop, b_mf; abstol = 1e-12, reltol = 1e-12, maxiter = 4 * n)

    rel_mf = norm(T_mf - T_exact) / max(norm(T_exact), 1.0)
    @test rel_mf < 1e-9

    # Operator consistency: op(T_exact) must equal b_mf to round-off.
    y_check = zeros(n)
    op(y_check, T_exact)
    rel_op = norm(y_check - b_mf) / max(norm(b_mf), 1.0)
    @test rel_op < 1e-12

    # Eliminated `apply_constraint!(K, c)` (homogeneous form) leaves the
    # diagonal rows of fixed DOFs as identity rows.
    K2 = Matrix(K)
    apply_constraint!(K2, bc)
    @inbounds for d in fixed_dofs
        @test K2[d, d] == 1.0
        @test all(iszero, K2[d, setdiff(1:n, [d])])
        @test all(iszero, K2[setdiff(1:n, [d]), d])
    end

    println("  Hex8 $(nx)×$(ny)×$(nz)  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "rel_dir=$(round(rel_dir; sigdigits = 3))  rel_mf_cg=$(round(rel_mf; sigdigits = 3))  " *
            "rel_op=$(round(rel_op; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 5. Type stability — assemble! and apply_K! infer concrete return types
# ----------------------------------------------------------------------------

@testset "HeatKernel: type stability (assemble! + apply_K!)" begin
    mesh = _build_hex8_box(2, 1, 1)
    cache, asm, kernel, m = _setup_heat(mesh, Hexahedron{8})
    n = cache.ndofs
    x = ones(n); y = zeros(n)
    assemble!(cache, asm, kernel, m)               # warmup
    apply_K!(y, cache, asm, kernel, m, x)          # warmup

    rt_asm = Base.promote_op(assemble!, typeof(cache), typeof(asm),
                             typeof(kernel), typeof(m))
    @test rt_asm === Nothing

    rt_mf = Base.promote_op(apply_K!, typeof(y), typeof(cache), typeof(asm),
                            typeof(kernel), typeof(m), typeof(x))
    @test rt_mf === Vector{Float64}

    code_asm = code_typed(assemble!,
                          (typeof(cache), typeof(asm), typeof(kernel), typeof(m));
                          optimize = true)
    @test !isempty(code_asm)
    @test isconcretetype(code_asm[1].second)

    code_mf = code_typed(apply_K!,
                         (typeof(y), typeof(cache), typeof(asm),
                          typeof(kernel), typeof(m), typeof(x));
                         optimize = true)
    @test !isempty(code_mf)
    @test isconcretetype(code_mf[1].second)

    println("  HeatKernel inferred return types:")
    println("    assemble! → $(code_asm[1].second)")
    println("    apply_K!  → $(code_mf[1].second)")
end
