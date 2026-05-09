# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Regression + correctness tests for `apply_K!` — the matrix-free `y = K x`
sibling of `assemble!` on the DOF-based assembler.

For every problem size:

1. Build assembled `K` via `assemble!` + `extract_system`.
2. Build the same `y_ref = K * x_random` for several random `x`.
3. Compare against `y_mf` from `apply_K!(y_mf, …, x)`. Must agree to
   round-off (no accumulation difference: same DOF traversal, same
   evaluate_entry inside).
4. Assert `apply_K!` is zero-allocation after warmup.
5. Assert the optimized LLVM IR for `apply_K!` has 0 GC allocation
   sites — guarantees the inner accumulate loop never heap-allocates,
   which is the whole point of the matrix-free path for Krylov.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, apply_K!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using LinearAlgebra
using SparseArrays
using Tensors
using Random
using InteractiveUtils       # code_llvm
using LinearOperators        # LinearOperator wrapper for apply_K!
using IterativeSolvers       # cg

# ----------------------------------------------------------------------------
# Mesh helpers (shared shape with test_dof_based_zero_alloc.jl, kept local
# so the two regression tests stay independent).
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

"Set up DOF-based cache + kernel for elasticity."
function _setup(mesh, ::Type{Topo}) where {Topo}
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Topo, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@testset "apply_K!: correctness vs assembled K" begin
    println("\n" * "=" ^ 70)
    println("DOF-BASED APPLY_K! CORRECTNESS")
    println("=" ^ 70)

    Random.seed!(20260508)

    # ------------------------------------------------------------------
    # 1. Single Tet4
    # ------------------------------------------------------------------
    @testset "Single Tet4" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m = _setup(mesh, Tetrahedron{4})

        # Assemble K once; compare K*x against apply_K!(y, …, x) for
        # several random x.
        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        max_rel = 0.0
        for trial in 1:8
            x  = randn(n)
            y_ref = K * x
            y_mf  = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            rel = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end
        println("  Single Tet4 ...... n=$n   max rel=$(round(max_rel; sigdigits=3))")
    end

    # ------------------------------------------------------------------
    # 2. Growing Hex8 cube meshes
    # ------------------------------------------------------------------
    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3), (8, 4, 4)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup(mesh, Hexahedron{8})

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        max_rel = 0.0
        for trial in 1:5
            x  = randn(n)
            y_ref = K * x
            y_mf  = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            rel = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end

        nelems = length(m.connectivity)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems, 5)) elem  " *
                "$(lpad(n, 5)) dof   max rel=$(round(max_rel; sigdigits=3))")
    end
end

# ----------------------------------------------------------------------------
# Zero-allocation + LLVM IR check for apply_K!
# ----------------------------------------------------------------------------
@testset "apply_K!: zero allocation + 0 LLVM gc-alloc sites" begin
    println("\n" * "=" ^ 70)
    println("DOF-BASED APPLY_K! ZERO-ALLOC")
    println("=" ^ 70)

    @testset "Single Tet4" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m = _setup(mesh, Tetrahedron{4})
        n = cache.ndofs

        x = ones(n)
        y = zeros(n)

        # Warmup: compile + populate caches
        apply_K!(y, cache, asm, kernel, m, x)

        GC.gc()
        a = @allocated apply_K!(y, cache, asm, kernel, m, x)
        @test a == 0
        println("  Single Tet4 ........................ allocs=$a")
    end

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3), (8, 4, 4)]
        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup(mesh, Hexahedron{8})
        n = cache.ndofs

        x = ones(n)
        y = zeros(n)
        apply_K!(y, cache, asm, kernel, m, x)  # warmup

        GC.gc()
        a = @allocated apply_K!(y, cache, asm, kernel, m, x)
        @test a == 0

        nelems = length(m.connectivity)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems,5)) elem  " *
                "$(lpad(n,5)) dof   allocs=$a")
    end

    @testset "Optimized LLVM IR has 0 gc-alloc sites" begin
        mesh = _build_hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup(mesh, Hexahedron{8})
        n = cache.ndofs
        x = ones(n)
        y = zeros(n)
        apply_K!(y, cache, asm, kernel, m, x)  # warmup

        iob = IOBuffer()
        code_llvm(iob, apply_K!,
            Tuple{typeof(y), typeof(cache), typeof(asm),
                  typeof(kernel), typeof(m), typeof(x)};
            optimize = true)
        ir = String(take!(iob))
        n_alloc =
            length(collect(eachmatch(r"call.*julia\.gc_alloc", ir))) +
            length(collect(eachmatch(r"call.*jl_gc_pool_alloc", ir))) +
            length(collect(eachmatch(r"call.*jl_gc_big_alloc", ir))) +
            length(collect(eachmatch(r"call.*jl_gc_alloc_typed", ir)))
        @test n_alloc == 0
        println("  apply_K! optimized LLVM IR: $n_alloc gc-alloc sites " *
                "($(length(ir)) IR chars)")
    end
end

# ----------------------------------------------------------------------------
# Krylov demo: matrix-free CG using `apply_K!` as a `LinearOperator`.
# This is the actual use case for the matrix-free path — once it solves,
# the API is proven end-to-end against an off-the-shelf solver.
# ----------------------------------------------------------------------------
@testset "apply_K!: matrix-free CG via LinearOperators + IterativeSolvers" begin
    println("\n" * "=" ^ 70)
    println("DOF-BASED APPLY_K! — KRYLOV VALIDATION")
    println("=" ^ 70)

    # Small unit cube, 3×3×3 = 27 elements, 192 DOFs.
    nx = ny = nz = 3
    mesh = _build_hex8_box(nx, ny, nz)
    cache, asm, kernel, m = _setup(mesh, Hexahedron{8})
    n = cache.ndofs

    # Bottom-face homogeneous Dirichlet via the shared `PenaltyDirichlet`
    # abstraction — same struct that the heat domain uses.
    fixed_dofs = Int[]
    for (nid, X) in enumerate(m.nodes)
        if X[3] == 0.0
            push!(fixed_dofs, 3*(nid - 1) + 1)
            push!(fixed_dofs, 3*(nid - 1) + 2)
            push!(fixed_dofs, 3*(nid - 1) + 3)
        end
    end
    @test !isempty(fixed_dofs)
    bc = PenaltyDirichlet(fixed_dofs; penalty = 1e16)

    # Loading: unit downward force on the corner (nx+1, ny+1, nz+1).
    top_corner_node = (nx + 1) * (ny + 1) * (nz + 1)
    fz_dof          = 3 * (top_corner_node - 1) + 3
    b               = zeros(n)
    b[fz_dof]       = -1e6     # 1 MN (problem is in SI; just a number)

    # 1. Direct solution via assembled K with the same penalty BC.
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)
    Kbc  = Matrix(K)
    apply_constraint!(Kbc, bc)
    u_direct = Kbc \ b

    # 2. Matrix-free CG: `matrix_free_op` builds the closure with the
    #    Dirichlet contribution baked in. No K is materialized.
    op    = matrix_free_op(cache, asm, kernel, m; dirichlet = bc)
    linop = LinearOperator(Float64, n, n, true, true, op)

    u_mf = zeros(n)
    cg!(u_mf, linop, b; abstol = 1e-8, reltol = 1e-10, maxiter = 4 * n)

    # 3. Compare matrix-free to direct.
    rel_u  = norm(u_mf - u_direct) / max(norm(u_direct), 1.0)
    @test rel_u < 1e-6

    # Also verify residual of matrix-free solution under the same operator.
    r = zeros(n)
    op(r, u_mf)
    rel_r = norm(b - r) / max(norm(b), 1.0)
    @test rel_r < 1e-6

    println("  Hex8 $(nx)×$(ny)×$(nz)  ndof=$n  fixed=$(length(fixed_dofs))  " *
            "rel_u=$(round(rel_u; sigdigits=3))  rel_r=$(round(rel_r; sigdigits=3))")
end

# ----------------------------------------------------------------------------
# Type stability: apply_K! must infer `Vector{Float64}` (==typeof(y)) end-to-end
# ----------------------------------------------------------------------------
@testset "apply_K!: type stability" begin
    mesh = _build_hex8_box(2, 1, 1)
    cache, asm, kernel, m = _setup(mesh, Hexahedron{8})
    n = cache.ndofs
    x = ones(n)
    y = zeros(n)
    apply_K!(y, cache, asm, kernel, m, x)  # warmup

    rt = Base.promote_op(apply_K!, typeof(y), typeof(cache), typeof(asm),
                         typeof(kernel), typeof(m), typeof(x))
    @test rt === Vector{Float64}

    code = code_typed(apply_K!,
                      (typeof(y), typeof(cache), typeof(asm),
                       typeof(kernel), typeof(m), typeof(x));
                      optimize = true)
    @test !isempty(code)
    info = code[1]
    @test isconcretetype(info.second)
    println("  apply_K! inferred return type: $(info.second)")
end
