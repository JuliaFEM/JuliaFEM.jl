# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Backend-agnostic `apply_K!` (KernelAbstractions.jl) regression.

What we *can* test locally on darwin / aarch64:

1. The KA cache mirror builds without errors from a CPU `DOFBasedCOOCache`.
2. `apply_K!(y, cache_ka, kernel, x)` on `KernelAbstractions.CPU()` matches
   the direct CPU `apply_K!(y, cache, asm, kernel, mesh, x)` to round-off,
   for both Tet4 and Hex8 meshes of various sizes (max rel error == 0.0
   in practice).
3. The result also matches `K * x` from the assembled matrix (transitivity
   check: same path the original `apply_K!` test guarantees).
4. Re-running Pass 1 + re-syncing the KA cache produces identical output
   on subsequent calls (Newton-Krylov rehearsal).

What we cannot test here:

* Actual CUDA / Metal / AMDGPU execution. Those require the
  corresponding hardware. The KA `@kernel` is defined once and
  dispatches on backend; once a GPU host runs this same test with
  `Adapt.adapt(CUDABackend(), cache_ka)` it should pass identically.

# Allocation note

The KA path on `CPU()` backend allocates ≈700 bytes per `apply_K!` launch
for KA's host-side scheduler metadata (workgroup, ndrange, etc.). This
overhead is *not* present on real GPU backends and is not asserted here;
the canonical zero-allocation CPU path is the direct
`apply_K!(y, cache, asm, kernel, mesh, x)` (see `test_dof_based_apply_K.jl`).
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, apply_K!
using JuliaFEM: DOFBasedCOOCacheKA, sync_from_cpu!, _prepare_caches!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using LinearAlgebra
using SparseArrays
using Tensors
using Random
using KernelAbstractions

# ----------------------------------------------------------------------------
# Mesh helpers (kept local so this file is independent).
# ----------------------------------------------------------------------------

function _ka_build_hex8_box(nx::Int, ny::Int, nz::Int)
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

function _ka_build_single_tet4()
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
function _ka_setup(mesh, ::Type{Topo}) where {Topo}
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    ElemType = Element{Topo, Lagrange{1}, S}
    elements, dof_mgr = create_elements!(mesh, ElemType)
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@testset "apply_K! (KernelAbstractions, CPU backend) correctness" begin
    println("\n" * "=" ^ 70)
    println("DOF-BASED APPLY_K! VIA KERNELABSTRACTIONS — CPU BACKEND")
    println("=" ^ 70)

    Random.seed!(20260508)

    # ------------------------------------------------------------------
    # Single Tet4
    # ------------------------------------------------------------------
    @testset "Single Tet4" begin
        mesh = _ka_build_single_tet4()
        cache, asm, kernel, m = _ka_setup(mesh, Tetrahedron{4})

        # Pass 1 + assemble K (CPU); also build the KA mirror.
        assemble!(cache, asm, kernel, m)
        K, _    = extract_system(cache)
        n       = size(K, 1)
        cache_ka = DOFBasedCOOCacheKA(cache)
        sync_from_cpu!(cache_ka, cache)

        max_rel_vs_ref  = 0.0
        max_rel_vs_cpu  = 0.0
        for trial in 1:8
            x      = randn(n)
            y_ref  = K * x

            y_cpu  = zeros(n)
            apply_K!(y_cpu, cache, asm, kernel, m, x)

            y_ka   = zeros(n)
            apply_K!(y_ka, cache_ka, kernel, x)

            err_ref = norm(y_ka .- y_ref) / max(norm(y_ref), 1.0)
            err_cpu = norm(y_ka .- y_cpu) / max(norm(y_cpu), 1.0)
            max_rel_vs_ref = max(max_rel_vs_ref, err_ref)
            max_rel_vs_cpu = max(max_rel_vs_cpu, err_cpu)
        end

        @test max_rel_vs_ref < 1e-12
        @test max_rel_vs_cpu < 1e-14
        println("  Single Tet4  n=$n  max rel vs K*x=$(round(max_rel_vs_ref, sigdigits=3))  max rel vs CPU apply_K!=$(round(max_rel_vs_cpu, sigdigits=3))")
    end

    # ------------------------------------------------------------------
    # Hex8 cubes — sweep size to exercise multi-element row coverage
    # ------------------------------------------------------------------
    @testset "Hex8 cubes" begin
        for (nx, ny, nz) in ((1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3))
            @testset "Hex8 $(nx)×$(ny)×$(nz)" begin
                mesh = _ka_build_hex8_box(nx, ny, nz)
                cache, asm, kernel, m = _ka_setup(mesh, Hexahedron{8})

                assemble!(cache, asm, kernel, m)
                K, _    = extract_system(cache)
                n       = size(K, 1)
                cache_ka = DOFBasedCOOCacheKA(cache)
                sync_from_cpu!(cache_ka, cache)

                max_rel_vs_ref = 0.0
                max_rel_vs_cpu = 0.0
                for trial in 1:5
                    x = randn(n)
                    y_ref = K * x

                    y_cpu = zeros(n)
                    apply_K!(y_cpu, cache, asm, kernel, m, x)

                    y_ka = zeros(n)
                    apply_K!(y_ka, cache_ka, kernel, x)

                    err_ref = norm(y_ka .- y_ref) / max(norm(y_ref), 1.0)
                    err_cpu = norm(y_ka .- y_cpu) / max(norm(y_cpu), 1.0)
                    max_rel_vs_ref = max(max_rel_vs_ref, err_ref)
                    max_rel_vs_cpu = max(max_rel_vs_cpu, err_cpu)
                end

                @test max_rel_vs_ref < 1e-12
                @test max_rel_vs_cpu < 1e-14
                println("  Hex8 $(nx)×$(ny)×$(nz)  $(nx*ny*nz) elem  $(n) dof  max rel vs K*x=$(round(max_rel_vs_ref, sigdigits=3))  max rel vs CPU apply_K!=$(round(max_rel_vs_cpu, sigdigits=3))")
            end
        end
    end
end


@testset "apply_K! (KernelAbstractions, CPU backend) — re-prepare flow" begin
    # Verify that re-running Pass 1 on the CPU cache and re-syncing to KA
    # mirror produces the same result on a second call. This exercises
    # the path a Newton-Krylov solver would take across iterations.
    mesh = _ka_build_hex8_box(2, 1, 1)
    cache, asm, kernel, m = _ka_setup(mesh, Hexahedron{8})

    assemble!(cache, asm, kernel, m)
    K, _    = extract_system(cache)
    n       = size(K, 1)
    cache_ka = DOFBasedCOOCacheKA(cache)

    Random.seed!(11)
    x = randn(n)

    # First call
    sync_from_cpu!(cache_ka, cache)
    y1 = zeros(n)
    apply_K!(y1, cache_ka, kernel, x)

    # Re-run Pass 1 (no parameter change), re-sync, check identical y
    JuliaFEM.reset!(cache)
    _prepare_caches!(cache, kernel, m)
    sync_from_cpu!(cache_ka, cache)
    y2 = zeros(n)
    apply_K!(y2, cache_ka, kernel, x)

    @test y1 ≈ y2
    @test maximum(abs, y1 .- y2) ≤ eps(Float64) * max(norm(y1), 1.0)
end
