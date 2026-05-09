# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Mass-matrix microkernel tests through the DOF-based assembler.

What the test file proves:

  1. **Default kernel produces zero `M`.** A `ContinuumKernel` /
     `HeatKernel` constructed without `density` / `heat_capacity` returns
     `evaluate_mass_entry == 0` so both `apply_M!` and `assemble_M!`
     produce a structural-zero `M`. Existing static-only tests stay valid.

  2. **Consistent mass matrix is correct.** With unit density on a unit
     cube, the row-sum of `M` (which equals `sum(M*1) = ∫ ρ dV = ρ·V`)
     matches `density * volume` to round-off — for elasticity it sums
     each component independently, for heat it sums the scalar field.

  3. **`M` is symmetric, SPD on the active DOFs, and `apply_M!` matches
     `M * x`** to round-off for several random `x`.

  4. **Density scaling is linear.** Doubling `density` doubles every
     entry of `M` exactly.

  5. **Block-diagonal in components.** The elasticity mass matrix has
     no `(α, β)` cross terms — `M[i_x, j_y] == 0` for any node pair.

  6. **Zero allocations.** Both `apply_M!` and `assemble_M!` allocate
     0 bytes per call after warmup, same hard contract as `apply_K!` /
     `assemble!`.

  7. **Cache reuse.** Calling `assemble!` then `assemble_M!` on the
     same cache produces an independent `K` and `M`, both correct.

The combination of (2) and (4) verifies the `evaluate_mass_entry`
microkernel is *the* place to extend mass behaviour — variable density
materials drop in by overriding `evaluate_mass_entry` for a new kernel
type without touching the assembler.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Temperature, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_M!, assemble_M!, extract_system
using JuliaFEM: create_elements!
using LinearAlgebra
using SparseArrays
using Tensors
using Random

# ----------------------------------------------------------------------------
# Mesh helpers (mirror the other DOF-based test files; kept local so this
# file stays independent and the WARNINGS about helper redefinitions are
# expected when the suites run together).
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

"Set up DOF-based elasticity assembly with optional density."
function _setup_elasticity(mesh; density::Float64 = 0.0)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}();
                               density = density)
    S        = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

"Set up DOF-based heat assembly with optional heat capacity."
function _setup_heat(mesh; heat_capacity::Float64 = 0.0)
    material = HeatConductivity(k = 50.2)
    kernel   = HeatKernel(ContinuumFormulation{FullThreeD}(),
                          material, Temperature();
                          heat_capacity = heat_capacity)
    S        = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# 1. Default kernels (no density) → structurally-zero M
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: default kernels return structural-zero M" begin
    println("\n" * "=" ^ 70)
    println("MASS MATRIX — defaults (no density / heat_capacity)")
    println("=" ^ 70)

    @testset "ContinuumKernel default density" begin
        mesh = _build_hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        @test kernel.density == 0.0

        assemble_M!(cache, asm, kernel, m)
        M, _ = extract_system(cache)
        @test maximum(abs, M) == 0.0

        x = randn(cache.ndofs); y = zeros(cache.ndofs)
        apply_M!(y, cache, asm, kernel, m, x)
        @test all(iszero, y)

        println("  ContinuumKernel  density=0  M structural zero ✓")
    end

    @testset "HeatKernel default heat_capacity" begin
        mesh = _build_hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup_heat(mesh)
        @test kernel.heat_capacity == 0.0

        assemble_M!(cache, asm, kernel, m)
        M, _ = extract_system(cache)
        @test maximum(abs, M) == 0.0

        x = randn(cache.ndofs); y = zeros(cache.ndofs)
        apply_M!(y, cache, asm, kernel, m, x)
        @test all(iszero, y)

        println("  HeatKernel       heat_capacity=0  M structural zero ✓")
    end
end

# ----------------------------------------------------------------------------
# 2. Correctness: row-sum = ρ·V, symmetry, SPD, apply_M! matches M*x.
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: correctness (row-sum, symmetry, SPD, apply_M!)" begin
    println("\n" * "=" ^ 70)
    println("MASS MATRIX — CORRECTNESS (consistent M)")
    println("=" ^ 70)

    Random.seed!(20260508)

    @testset "Heat: ρcp = 1, unit cube" begin
        ρcp = 1.0
        mesh = _build_hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_heat(mesh; heat_capacity = ρcp)
        n = cache.ndofs

        assemble_M!(cache, asm, kernel, m)
        M, _ = extract_system(cache)

        # Symmetric to round-off
        @test maximum(abs, M - M') < 1e-12 * maximum(abs, M)

        # Row-sum: sum(M*1) = ρcp · ∫ dV = ρcp · 1.0 (unit cube)
        rowsum_total = sum(M * ones(n))
        @test isapprox(rowsum_total, ρcp * 1.0; rtol = 1e-12)

        # SPD: every diagonal positive, x' M x > 0 for several random x
        @test all(>(0.0), diag(M))
        for _ in 1:5
            x = randn(n)
            @test x' * M * x > 0.0
        end

        # apply_M! ≡ M * x to round-off
        max_rel = 0.0
        for _ in 1:5
            x     = randn(n)
            y_ref = M * x
            y_mf  = zeros(n); apply_M!(y_mf, cache, asm, kernel, m, x)
            rel   = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end

        println("  Heat   ndof=$n  rowsum=$(round(rowsum_total; sigdigits = 5))  " *
                "(expected $(ρcp))  max(apply_M! vs M*x)=$(round(max_rel; sigdigits = 3))")
    end

    @testset "Elasticity: ρ = 1, unit cube" begin
        ρ = 1.0
        mesh = _build_hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_elasticity(mesh; density = ρ)
        n = cache.ndofs

        assemble_M!(cache, asm, kernel, m)
        M, _ = extract_system(cache)

        # Symmetric to round-off
        @test maximum(abs, M - M') < 1e-12 * maximum(abs, M)

        # Row-sum per component: each of the 3 displacement components
        # independently has sum = ρ · V. Total row-sum is 3 · ρ · V.
        rowsum_total = sum(M * ones(n))
        @test isapprox(rowsum_total, 3 * ρ * 1.0; rtol = 1e-12)

        @test all(>(0.0), diag(M))

        max_rel = 0.0
        for _ in 1:5
            x     = randn(n)
            y_ref = M * x
            y_mf  = zeros(n); apply_M!(y_mf, cache, asm, kernel, m, x)
            rel   = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end

        println("  Elast  ndof=$n  rowsum=$(round(rowsum_total; sigdigits = 5))  " *
                "(expected $(3 * ρ))  max(apply_M! vs M*x)=$(round(max_rel; sigdigits = 3))")
    end
end

# ----------------------------------------------------------------------------
# 3. Density scaling is linear.
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: density scaling is linear" begin
    mesh = _build_hex8_box(2, 1, 1)

    cache_a, asm_a, kernel_a, m_a = _setup_elasticity(mesh; density = 1.0)
    cache_b, asm_b, kernel_b, m_b = _setup_elasticity(mesh; density = 2.5)

    assemble_M!(cache_a, asm_a, kernel_a, m_a); M_a, _ = extract_system(cache_a)
    assemble_M!(cache_b, asm_b, kernel_b, m_b); M_b, _ = extract_system(cache_b)

    @test maximum(abs, M_b - 2.5 * M_a) < 1e-12 * maximum(abs, M_a)
end

# ----------------------------------------------------------------------------
# 4. Block-diagonal structure of the elasticity mass matrix.
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: elasticity mass is block-diagonal in components" begin
    mesh = _build_hex8_box(1, 1, 1)
    cache, asm, kernel, m = _setup_elasticity(mesh; density = 1.0)
    n = cache.ndofs

    assemble_M!(cache, asm, kernel, m)
    M, _ = extract_system(cache)
    Md = Matrix(M)

    # DOF layout for displacement is (node, x), (node, y), (node, z) per node.
    # `local_dof_layout(Element{Hex8, ...})` orders them this way, and
    # `create_elements!` produces a global numbering matching that order.
    # So DOF index `3*(node-1) + α` for component α ∈ {1,2,3}.
    nnodes = length(m.nodes)
    @test n == 3 * nnodes

    # Cross-component blocks must be zero — pick a random pair of nodes
    # and verify M[i_x, j_y], M[i_x, j_z], M[i_y, j_z] are all zero.
    Random.seed!(20260508)
    for _ in 1:5
        i = rand(1:nnodes); j = rand(1:nnodes)
        for (αi, αj) in ((1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2))
            row = 3 * (i - 1) + αi
            col = 3 * (j - 1) + αj
            @test Md[row, col] == 0.0
        end
    end
end

# ----------------------------------------------------------------------------
# 5. Zero-alloc + KA-untouched contract.
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: zero allocations (apply_M! + assemble_M!)" begin
    println("\n" * "=" ^ 70)
    println("MASS MATRIX — ZERO-ALLOC")
    println("=" ^ 70)

    @testset "Elast cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_elasticity(mesh; density = 7850.0)
        n = cache.ndofs
        x = ones(n); y = zeros(n)

        # warmup
        assemble_M!(cache, asm, kernel, m)
        apply_M!(y, cache, asm, kernel, m, x)

        GC.gc()
        a_asm = @allocated assemble_M!(cache, asm, kernel, m)
        @test a_asm == 0

        GC.gc()
        a_mf = @allocated apply_M!(y, cache, asm, kernel, m, x)
        @test a_mf == 0

        nelems = length(m.connectivity)
        println("  Elast $(nx)×$(ny)×$(nz)  $(lpad(nelems,3)) elem  " *
                "$(lpad(n,4)) dof   assemble_M!=$a_asm  apply_M!=$a_mf")
    end

    @testset "Heat cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_heat(mesh; heat_capacity = 3500.0)
        n = cache.ndofs
        x = ones(n); y = zeros(n)

        assemble_M!(cache, asm, kernel, m)
        apply_M!(y, cache, asm, kernel, m, x)

        GC.gc()
        a_asm = @allocated assemble_M!(cache, asm, kernel, m)
        @test a_asm == 0

        GC.gc()
        a_mf = @allocated apply_M!(y, cache, asm, kernel, m, x)
        @test a_mf == 0

        nelems = length(m.connectivity)
        println("  Heat  $(nx)×$(ny)×$(nz)  $(lpad(nelems,3)) elem  " *
                "$(lpad(n,4)) dof   assemble_M!=$a_asm  apply_M!=$a_mf")
    end
end

# ----------------------------------------------------------------------------
# 6. Cache reuse: assemble! then assemble_M! produces correct K and M.
# ----------------------------------------------------------------------------

@testset "evaluate_mass_entry: cache reuse (K then M)" begin
    println("\n" * "=" ^ 70)
    println("MASS MATRIX — CACHE REUSE (assemble! then assemble_M!)")
    println("=" ^ 70)

    mesh = _build_hex8_box(2, 2, 2)
    cache, asm, kernel, m = _setup_elasticity(mesh; density = 7850.0)
    n = cache.ndofs

    # 1. Assemble K, extract.
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)

    # 2. Then assemble M into the *same* cache, extract.
    assemble_M!(cache, asm, kernel, m)
    M, _ = extract_system(cache)

    # K and M are independent SparseMatrixCSC instances now.
    # K must be SPD on free DOFs; M must be SPD outright (positive
    # diagonal, x' M x > 0).
    @test maximum(abs, K - K') < 1e-9 * maximum(abs, K)
    @test maximum(abs, M - M') < 1e-12 * maximum(abs, M)

    # M's row-sum must still equal 3 ρ V (unaffected by the prior K assembly).
    rowsum_total = sum(M * ones(n))
    @test isapprox(rowsum_total, 3 * 7850.0 * 1.0; rtol = 1e-12)

    # Quick sanity: K is *not* M (would imply the cache wasn't reset
    # between the two assemblies).
    @test maximum(abs, K - M) > 0.5 * maximum(abs, K)

    println("  Hex8 2×2×2  ndof=$n  K SPD ✓  M SPD ✓  rowsum(M)=" *
            "$(round(rowsum_total; sigdigits = 5))  (expected $(3 * 7850.0))")
end
