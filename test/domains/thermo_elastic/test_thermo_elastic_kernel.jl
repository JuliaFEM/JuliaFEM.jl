# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
First multi-field test through the DOF-based assembler.

`ThermoElasticKernel` couples a vector displacement field `u` (3 DOFs/
node) with a scalar temperature field `T` (1 DOF/node), giving 4 DOFs
per node. This file's purpose is to prove that:

1. `local_dof_layout(E)` for the multi-field DOFSet returns 4 entries
   per node with `field_idx ∈ {1, 2}`,
2. `_prepare_caches!` correctly fills `element_cache.dofs` from
   `elem.dof_indices` (multi-field), so `assemble!` writes triplets
   into the right global rows / columns,
3. With `β = 0` (block-diagonal) the assembled `K` decomposes exactly
   into the standalone `ContinuumKernel` and `HeatKernel` blocks,
4. With `β ≠ 0` the off-diagonal `K_uT` and `K_Tu` blocks are non-zero
   and the global `K` is symmetric,
5. `apply_K!` (CPU) matches `K * x` to round-off across the multi-field
   DOF traversal,
6. `apply_K!` (KA, CPU backend) is bit-equivalent to the direct CPU
   `apply_K!`,
7. Both `assemble!` and `apply_K!` are zero-allocation in the hot loop
   for the multi-field kernel.

If any of these regress, the multi-field path through the DOF-based
assembler is broken.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD
using JuliaFEM: Displacement, Temperature, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, HeatConductivity
using JuliaFEM: ContinuumKernel, HeatKernel, ThermoElasticKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, apply_K!, extract_system
using JuliaFEM: DOFBasedCOOCacheKA, sync_from_cpu!
using JuliaFEM: create_elements!
using JuliaFEM: local_dof_layout, DOFLayoutEntry, field_idx, entity_local, component
using LinearAlgebra
using SparseArrays
using Tensors
using Random

# ----------------------------------------------------------------------------
# Mesh helper — single Hex8 cube, enough to expose multi-field plumbing
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
        n1 = nidx(i,     j,     k);     n2 = nidx(i + 1, j,     k)
        n3 = nidx(i + 1, j + 1, k);     n4 = nidx(i,     j + 1, k)
        n5 = nidx(i,     j,     k + 1); n6 = nidx(i + 1, j,     k + 1)
        n7 = nidx(i + 1, j + 1, k + 1); n8 = nidx(i,     j + 1, k + 1)
        push!(conns, (UInt32(n1), UInt32(n2), UInt32(n3), UInt32(n4),
                      UInt32(n5), UInt32(n6), UInt32(n7), UInt32(n8)))
    end
    return Mesh{8,Hexahedron{8}}(nodes, conns)
end

# Block-extraction helpers — work directly off the multi-field
# `local_dof_layout` so we never hard-code field-block strides.
function _u_dofs(handler, n_nodes)
    starts = handler.field_starts[1]    # field 1 = u in our DOFSet
    return [starts[k] + (c - 1) for k in 1:n_nodes, c in 1:3] |> vec |> sort
end
function _T_dofs(handler, n_nodes)
    starts = handler.field_starts[2]    # field 2 = T
    return collect(starts)
end

# ----------------------------------------------------------------------------
# Setups for the three kernels (shared mesh + matching materials)
# ----------------------------------------------------------------------------

function _setup_te_block_diagonal(mesh; β = 0.0)
    mech = LinearElastic(E = 210e9, ν = 0.3)
    therm = HeatConductivity(k = 50.2)
    kernel = ThermoElasticKernel(ContinuumFormulation{FullThreeD}(),
                                 mech, therm, β)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex},
                T::DOF{Temperature, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    return cache, asm, kernel, mesh, handler
end

function _setup_pure_elasticity(mesh)
    mat = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat,
                             Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    return cache, asm, kernel, mesh
end

function _setup_pure_heat(mesh)
    mat = HeatConductivity(k = 50.2)
    kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), mat)
    S = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    return cache, asm, kernel, mesh
end


# ----------------------------------------------------------------------------
# 1. Multi-field local_dof_layout — verify the element template is right
# ----------------------------------------------------------------------------

@testset "ThermoElasticKernel: multi-field local_dof_layout" begin
    println("\n" * "=" ^ 70)
    println("THERMO-ELASTIC — MULTI-FIELD LOCAL_DOF_LAYOUT")
    println("=" ^ 70)

    mesh = _build_hex8_box(1, 1, 1)
    cache, asm, kernel, m, handler = _setup_te_block_diagonal(mesh; β = 0.0)

    ET = eltype(cache.elements)
    layout = local_dof_layout(ET)
    @test layout isa NTuple{32, DOFLayoutEntry}     # 8*3 + 8*1

    # Field-1 entries (u): 24 of them, vertices 1..8 × components 1..3
    f1 = filter(e -> field_idx(e) == 1, collect(layout))
    @test length(f1) == 24
    @test [Int(entity_local(e)) for e in f1] == repeat(1:8; inner = 3)
    @test [Int(component(e))    for e in f1] == repeat(1:3, 8)

    # Field-2 entries (T): 8 of them, vertices 1..8 × component 1
    f2 = filter(e -> field_idx(e) == 2, collect(layout))
    @test length(f2) == 8
    @test [Int(entity_local(e)) for e in f2] == collect(1:8)
    @test all(e -> component(e) == 1, f2)

    # Handler block ordering: u DOFs come first (24), T DOFs follow (8)
    n_nodes = length(m.nodes)
    @test handler.total_dofs == 4 * n_nodes
    @test length(handler.field_starts) == 2

    println("  Hex8 1×1×1   ndofs(elem)=32   ndofs(global)=$(handler.total_dofs)" *
            "   u/T split: 24/8 ✓")
end


# ----------------------------------------------------------------------------
# 2. Block-diagonal (β = 0): K_uu == pure elasticity K, K_TT == pure heat K
# ----------------------------------------------------------------------------

@testset "ThermoElasticKernel: β = 0 reproduces single-field blocks exactly" begin
    println("\n" * "=" ^ 70)
    println("THERMO-ELASTIC — BLOCK DIAGONAL CONSISTENCY (β = 0)")
    println("=" ^ 70)

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _build_hex8_box(nx, ny, nz)

        # Coupled system, β = 0
        cache_te, asm_te, ker_te, m_te, h_te = _setup_te_block_diagonal(mesh; β = 0.0)
        assemble!(cache_te, asm_te, ker_te, m_te)
        K_te, _ = extract_system(cache_te)

        # Pure elasticity reference
        cache_e, asm_e, ker_e, _ = _setup_pure_elasticity(mesh)
        assemble!(cache_e, asm_e, ker_e, mesh)
        K_e, _ = extract_system(cache_e)

        # Pure heat reference
        cache_h, asm_h, ker_h, _ = _setup_pure_heat(mesh)
        assemble!(cache_h, asm_h, ker_h, mesh)
        K_h, _ = extract_system(cache_h)

        n_nodes = length(mesh.nodes)
        u_idx = _u_dofs(h_te, n_nodes)
        T_idx = _T_dofs(h_te, n_nodes)
        @test length(u_idx) == 3 * n_nodes
        @test length(T_idx) == n_nodes
        @test size(K_te, 1) == 4 * n_nodes
        @test isempty(intersect(u_idx, T_idx))

        K_uu = Matrix(K_te[u_idx, u_idx])
        K_TT = Matrix(K_te[T_idx, T_idx])
        K_uT = Matrix(K_te[u_idx, T_idx])
        K_Tu = Matrix(K_te[T_idx, u_idx])

        # Block-diagonal: off-diagonal blocks must be exactly zero
        @test maximum(abs, K_uT) == 0.0
        @test maximum(abs, K_Tu) == 0.0

        # Diagonal blocks must agree with the single-field assemblies
        rel_uu = maximum(abs, K_uu - Matrix(K_e)) / max(maximum(abs, Matrix(K_e)), 1.0)
        rel_TT = maximum(abs, K_TT - Matrix(K_h)) / max(maximum(abs, Matrix(K_h)), 1.0)
        @test rel_uu < 1e-12
        @test rel_TT < 1e-12

        println("  Hex8 $(nx)×$(ny)×$(nz)   ndof=$(size(K_te,1))   " *
                "uu_rel=$(round(rel_uu; sigdigits=3))   TT_rel=$(round(rel_TT; sigdigits=3))   " *
                "uT_max=$(maximum(abs, K_uT))   Tu_max=$(maximum(abs, K_Tu))")
    end
end


# ----------------------------------------------------------------------------
# 3. Coupled (β ≠ 0): off-diagonal blocks non-zero + global K symmetric
# ----------------------------------------------------------------------------

@testset "ThermoElasticKernel: β ≠ 0 produces symmetric coupled K" begin
    println("\n" * "=" ^ 70)
    println("THERMO-ELASTIC — COUPLED K (β ≠ 0)")
    println("=" ^ 70)

    Random.seed!(20260509)

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m, h = _setup_te_block_diagonal(mesh; β = 1.0e7)

        assemble!(cache, asm, kernel, m)
        K, _ = extract_system(cache)
        n = size(K, 1)

        # Symmetry: the coupling form was chosen specifically to keep K SPD-shaped
        @test maximum(abs, Matrix(K) - Matrix(K)') < 1e-9 * maximum(abs, Matrix(K))

        # Off-diagonal blocks must be non-trivially populated
        n_nodes = length(m.nodes)
        u_idx = _u_dofs(h, n_nodes)
        T_idx = _T_dofs(h, n_nodes)
        K_uT = Matrix(K[u_idx, T_idx])
        K_Tu = Matrix(K[T_idx, u_idx])
        @test maximum(abs, K_uT) > 0.0
        @test maximum(abs, K_Tu) > 0.0

        # apply_K! (CPU) must agree with K * x for several random x
        max_rel = 0.0
        for _ in 1:5
            x  = randn(n)
            y_ref = K * x
            y_mf  = zeros(n)
            apply_K!(y_mf, cache, asm, kernel, m, x)
            rel = norm(y_mf - y_ref) / max(norm(y_ref), 1.0)
            @test rel < 1e-12
            max_rel = max(max_rel, rel)
        end

        # apply_K! through KA (CPU backend) must be bit-identical to CPU
        ka = DOFBasedCOOCacheKA(cache); sync_from_cpu!(ka, cache)
        x  = randn(n)
        y_cpu = zeros(n); y_ka = zeros(n)
        apply_K!(y_cpu, cache, asm, kernel, m, x)
        apply_K!(y_ka, ka, kernel, x)
        @test maximum(abs, y_cpu - y_ka) == 0.0

        println("  Hex8 $(nx)×$(ny)×$(nz)   ndof=$n   uT_max=" *
                "$(round(maximum(abs, K_uT); sigdigits=3))   max(K-K')/max(K)=" *
                "$(round(maximum(abs, Matrix(K) - Matrix(K)') / maximum(abs, Matrix(K));
                        sigdigits=3))   apply_K!_max_rel=" *
                "$(round(max_rel; sigdigits=3))")
    end
end


# ----------------------------------------------------------------------------
# 4. Zero-allocation in the hot loop (multi-field assemble! + apply_K!)
# ----------------------------------------------------------------------------

@testset "ThermoElasticKernel: zero-alloc assemble! + apply_K!" begin
    println("\n" * "=" ^ 70)
    println("THERMO-ELASTIC — ZERO-ALLOC")
    println("=" ^ 70)

    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]
        mesh = _build_hex8_box(nx, ny, nz)
        cache, asm, kernel, m, _ = _setup_te_block_diagonal(mesh; β = 1.5)
        n = cache.ndofs
        x = ones(n); y = zeros(n)

        assemble!(cache, asm, kernel, m)              # warmup
        apply_K!(y, cache, asm, kernel, m, x)         # warmup

        GC.gc()
        a_asm = @allocated assemble!(cache, asm, kernel, m)
        @test a_asm == 0

        GC.gc()
        a_mf  = @allocated apply_K!(y, cache, asm, kernel, m, x)
        @test a_mf == 0

        nelems = length(m.connectivity)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems,3)) elem  " *
                "$(lpad(n,4)) dof   assemble!=$a_asm  apply_K!=$a_mf")
    end
end
