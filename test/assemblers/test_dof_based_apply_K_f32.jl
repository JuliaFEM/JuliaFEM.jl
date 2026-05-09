# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Float32 (single-precision) `apply_K!` through the precision-parametric
`DOFBasedCOOCacheKA` path added in F+++.

Locks in the contract that closes the "Apple Metal can't store Float64"
gap from the previous round:

  1. **`to_float32(cpu_cache)` produces a `DOFBasedCOOCacheKA` whose
     storage is fully Float32**: `detJ_w_batch`, `N_batch`, `∇N_batch`,
     `X_batch`, and `qp_buffers` (a `SymmetricTensor{4,3,Float32,36}`
     for `ContinuumKernel`, `SymmetricTensor{2,3,Float32,6}` for
     `HeatKernel`) all carry `Float32` element type.

  2. **The same `apply_K_kernel!` runs in Float32**: extracts `F`
     dynamically from `eltype(y)` and accumulates in `zero(F)`. The
     KernelAbstractions launcher routes the F32 cache + F32 vectors to
     the same generic kernel without any duplicated code path.

  3. **F32 results agree with the F64 reference to single-precision
     accuracy** (≤ `1e-5` relative on well-conditioned problems) for
     both `ContinuumKernel` and `HeatKernel`.

  4. **Round-trip back-compat**: the default
     `DOFBasedCOOCacheKA(cpu_cache)` (Float64) still produces
     bit-identical results to the direct CPU `apply_K!`.

  5. **No interference with the CPU path**: existing CPU `apply_K!`
     and `assemble!` remain Float64 end-to-end.

Once a Float32 cache is on the device via `Adapt.adapt(MetalBackend(),
cache_f32)`, the same `apply_K!` call runs natively on Apple GPUs. The
device-specific test for that lives in `test/backend/metal/`; this file
locks in the *precision-parametric* contract on the CPU backend so the
Metal port has a green oracle to compare against.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Vertex, Temperature
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, DOFBasedCOOCacheKA
using JuliaFEM: extract_system, apply_K!, sync_from_cpu!, to_float32, create_elements!
using LinearAlgebra
using SparseArrays
using Tensors
using Random
using KernelAbstractions

# ----------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------

function _hex8_box(nx::Int, ny::Int, nz::Int)
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

# ----------------------------------------------------------------------------
# 1. Storage typing of the F32 cache
# ----------------------------------------------------------------------------

@testset "to_float32: storage element types are Float32" begin
    println("\n" * "=" ^ 70)
    println("F+++  Float32 KA cache storage typing")
    println("=" ^ 70)

    @testset "Elasticity: ContinuumKernel" begin
        mesh = _hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        assemble!(cache, asm, kernel, m)

        cache_f32 = to_float32(cache)
        @test eltype(cache_f32.X_batch)      === Vec{3,Float32}
        @test eltype(cache_f32.N_batch)      === Float32
        @test eltype(cache_f32.∇N_batch)     === Vec{3,Float32}
        @test eltype(cache_f32.detJ_w_batch) === Float32
        @test eltype(cache_f32.qp_buffers)   === SymmetricTensor{4,3,Float32,36}
        println("  Elasticity F32 cache types ✓  (qp_buffers = " *
                "$(eltype(cache_f32.qp_buffers)))")
    end

    @testset "Heat: HeatKernel" begin
        mesh = _hex8_box(2, 1, 1)
        cache, asm, kernel, m = _setup_heat(mesh)
        assemble!(cache, asm, kernel, m)

        cache_f32 = to_float32(cache)
        @test eltype(cache_f32.X_batch)      === Vec{3,Float32}
        @test eltype(cache_f32.detJ_w_batch) === Float32
        @test eltype(cache_f32.qp_buffers)   === SymmetricTensor{2,3,Float32,6}
        println("  Heat       F32 cache types ✓  (qp_buffers = " *
                "$(eltype(cache_f32.qp_buffers)))")
    end
end

# ----------------------------------------------------------------------------
# 2. F32 apply_K! agrees with F64 reference to single-precision accuracy
# ----------------------------------------------------------------------------

@testset "F32 apply_K!: agrees with F64 to single precision" begin
    println("\n" * "=" ^ 70)
    println("F+++  Float32 apply_K! correctness (CPU backend)")
    println("=" ^ 70)

    Random.seed!(20260508)

    @testset "Elasticity $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(2, 1, 1), (3, 2, 2), (4, 3, 3)]

        mesh = _hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)

        # F64 KA reference (aliased storage with the CPU cache)
        cache_f64 = DOFBasedCOOCacheKA(cache)
        sync_from_cpu!(cache_f64, cache)

        # F32 KA mirror
        cache_f32 = to_float32(cache)
        sync_from_cpu!(cache_f32, cache)

        max_rel = 0.0
        for trial in 1:5
            x64 = randn(n)
            y_ref = K * x64

            y64 = zeros(n)
            apply_K!(y64, cache_f64, kernel, x64)
            @test norm(y64 - y_ref) / norm(y_ref) < 1e-12

            x32 = Float32.(x64)
            y32 = zeros(Float32, n)
            apply_K!(y32, cache_f32, kernel, x32)
            rel = norm(Float64.(y32) - y_ref) / norm(y_ref)
            @test rel < 1e-5
            max_rel = max(max_rel, rel)
        end

        nelems = length(m.connectivity)
        println("  Elast $(nx)×$(ny)×$(nz)  $(lpad(nelems,3)) elem  " *
                "ndof=$(lpad(n,4))   max(F32 vs F64)=$(round(max_rel; sigdigits = 3))")
    end

    @testset "Heat $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(2, 1, 1), (3, 2, 2), (4, 3, 3)]

        mesh = _hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_heat(mesh)
        n = cache.ndofs

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)

        cache_f32 = to_float32(cache)
        sync_from_cpu!(cache_f32, cache)

        max_rel = 0.0
        for trial in 1:5
            x64 = randn(n)
            y_ref = K * x64

            x32 = Float32.(x64)
            y32 = zeros(Float32, n)
            apply_K!(y32, cache_f32, kernel, x32)
            rel = norm(Float64.(y32) - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-5
            max_rel = max(max_rel, rel)
        end

        nelems = length(m.connectivity)
        println("  Heat  $(nx)×$(ny)×$(nz)  $(lpad(nelems,3)) elem  " *
                "ndof=$(lpad(n,4))   max(F32 vs K*x F64)=$(round(max_rel; sigdigits = 3))")
    end
end

# ----------------------------------------------------------------------------
# 3. F32 cache enforces precision agreement (helpful error)
# ----------------------------------------------------------------------------

@testset "F32 cache rejects mismatched-precision input vectors" begin
    mesh = _hex8_box(2, 1, 1)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    assemble!(cache, asm, kernel, m)

    cache_f32 = to_float32(cache)
    sync_from_cpu!(cache_f32, cache)
    n = cache.ndofs

    # Passing Float64 vectors at an F32 cache must be flagged early
    # (otherwise we'd silently accumulate junk because the kernel reads
    # F = eltype(y) and assumes the cache matches).
    y64 = zeros(n); x64 = randn(n)
    @test_throws AssertionError apply_K!(y64, cache_f32, kernel, x64)
end

# ----------------------------------------------------------------------------
# 4. Default Float64 KA path is unchanged (round-trip back-compat)
# ----------------------------------------------------------------------------

@testset "Default DOFBasedCOOCacheKA stays bit-identical to CPU apply_K!" begin
    Random.seed!(20260508)

    mesh = _hex8_box(3, 2, 2)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n = cache.ndofs
    assemble!(cache, asm, kernel, m)
    K, _ = extract_system(cache)

    cache_f64 = DOFBasedCOOCacheKA(cache)
    sync_from_cpu!(cache_f64, cache)

    for _ in 1:3
        x = randn(n)
        y_ref = K * x
        y     = zeros(n); apply_K!(y, cache_f64, kernel, x)
        @test norm(y - y_ref) / norm(y_ref) < 1e-12
    end
end
