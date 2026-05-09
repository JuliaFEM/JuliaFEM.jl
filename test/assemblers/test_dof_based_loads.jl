# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Neumann load tests for the DOF-based assembler.

Locks in the contract for `NodalForce`, `UniformBodyForce`, and the
shared `apply_load!` entry point added in C++:

  1. **NodalForce** is a pure indexed accumulation; result is `f` plus
     the prescribed values at the prescribed DOFs.

  2. **UniformBodyForce** integrates `∫ N_i b dV` correctly via the
     SoA `N_data` / `detJ_w` batches: the *sum of f over a component
     `α`* equals `b_α · V` for elasticity, and `sum(f) == Q · V` for
     heat sources.

  3. **Composition is additive**: chaining `apply_load!` calls produces
     the sum of the two loads.

  4. **End-to-end Poisson with body source** — solve
     `−∇·(k ∇T) = Q  in Ω,  T = 0  on ∂Ω` on a 1D bar discretized as a
     long thin Hex8 strip. Compare against the analytical solution
     `T(x) = Q x (L − x) / (2 k)`. Validates `UniformBodyForce` is
     dimensionally consistent with `apply_K!` and the assembled `K`.

  5. **Zero allocations** for both `NodalForce` and `UniformBodyForce`
     after warmup.
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Temperature, Vertex
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: extract_system
using JuliaFEM: NodalForce, UniformBodyForce, apply_load!
using JuliaFEM: PenaltyDirichlet, EliminatedDirichlet, apply_constraint!
using JuliaFEM: create_elements!
using LinearAlgebra
using SparseArrays
using Tensors

# ----------------------------------------------------------------------------
# Mesh helpers (independent of the other DOF-based test files)
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# 1. NodalForce: indexed accumulation, additive composition
# ----------------------------------------------------------------------------

@testset "NodalForce: indexed accumulation" begin
    println("\n" * "=" ^ 70)
    println("LOADS — NodalForce")
    println("=" ^ 70)

    mesh = _hex8_box(2, 2, 2)
    cache, asm, kernel, m = _setup_elasticity(mesh)
    n = cache.ndofs

    dofs   = [3, 7, 9, 21]
    values = [10.0, -5.0, 2.5, 100.0]
    load   = NodalForce(dofs, values)

    f = zeros(n)
    apply_load!(f, load, cache, asm, kernel, m)

    expected = zeros(n)
    for k in eachindex(dofs)
        expected[dofs[k]] = values[k]
    end
    @test f == expected

    apply_load!(f, load, cache, asm, kernel, m)
    @test f == 2 * expected     # additive

    println("  ndof=$n  $(length(dofs)) point loads  exact + additive ✓")
end

# ----------------------------------------------------------------------------
# 2. UniformBodyForce row-sum identities
# ----------------------------------------------------------------------------

@testset "UniformBodyForce: row-sum equals b * V (heat & elasticity)" begin
    println("\n" * "=" ^ 70)
    println("LOADS — UniformBodyForce row-sum")
    println("=" ^ 70)

    @testset "Heat: scalar source Q on a 2x2x2 unit cube" begin
        Q = 12.5
        mesh = _hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_heat(mesh)
        n = cache.ndofs

        f = zeros(n)
        apply_load!(f, UniformBodyForce(Q), cache, asm, kernel, m)

        # ∫_Ω Q dV = Q · V on a unit cube
        @test isapprox(sum(f), Q * 1.0; rtol = 1e-12)
        # All entries non-negative and inhomogeneous (corners < edges < interior)
        @test all(>=(0.0), f)
        @test maximum(f) > minimum(f)
        println("  Heat   ndof=$n   Q=$Q   sum(f)=$(round(sum(f); sigdigits = 5))  " *
                "(expected $(Q))")
    end

    @testset "Elasticity: gravity body force on a 2x2x2 unit cube" begin
        ρ  = 7850.0
        g  = 9.81
        bz = -ρ * g
        b  = Vec{3,Float64}((0.0, 0.0, bz))
        mesh = _hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs

        f = zeros(n)
        apply_load!(f, UniformBodyForce(b), cache, asm, kernel, m)

        # Sum each component independently. DOF layout is
        # (node, x), (node, y), (node, z) per node, so
        # x-DOFs: 1, 4, 7, …;  y-DOFs: 2, 5, 8, …;  z-DOFs: 3, 6, 9, …
        sx = sum(f[1:3:end])
        sy = sum(f[2:3:end])
        sz = sum(f[3:3:end])
        @test isapprox(sx, 0.0; atol = 1e-9 * abs(bz))
        @test isapprox(sy, 0.0; atol = 1e-9 * abs(bz))
        @test isapprox(sz, bz * 1.0; rtol = 1e-12)

        println("  Elast  ndof=$n   bz=$(round(bz; sigdigits = 5))   " *
                "sum(f_x)=$(round(sx; sigdigits = 3))  sum(f_y)=$(round(sy; sigdigits = 3))  " *
                "sum(f_z)=$(round(sz; sigdigits = 5))  (expected $(round(bz; sigdigits = 5)))")
    end
end

# ----------------------------------------------------------------------------
# 3. Composition: chaining loads is additive
# ----------------------------------------------------------------------------

@testset "Loads: composition (NodalForce + UniformBodyForce)" begin
    mesh = _hex8_box(1, 1, 1)
    cache, asm, kernel, m = _setup_heat(mesh)
    n = cache.ndofs

    Q    = 3.0
    body = UniformBodyForce(Q)
    pts  = NodalForce([1, 4], [10.0, 20.0])

    f_body = zeros(n); apply_load!(f_body, body, cache, asm, kernel, m)
    f_pts  = zeros(n); apply_load!(f_pts,  pts,  cache, asm, kernel, m)

    f_combined = zeros(n)
    apply_load!(f_combined, body, cache, asm, kernel, m)
    apply_load!(f_combined, pts,  cache, asm, kernel, m)

    @test f_combined ≈ (f_body + f_pts) atol = 1e-12
end

# ----------------------------------------------------------------------------
# 4. End-to-end Poisson with body source (heat)
#
#    Solve  −k T''(x) = Q,  T(0) = T(L) = 0
#    Analytical: T(x) = Q x (L − x) / (2 k)
#
#    On a thin Hex8 strip in y, z (1 element across each of those axes,
#    free Neumann on the side walls). The midplane temperature must
#    match the 1D analytical to high accuracy at the nodal positions.
# ----------------------------------------------------------------------------

@testset "UniformBodyForce: Poisson 1D with prescribed source matches analytical" begin
    println("\n" * "=" ^ 70)
    println("LOADS — Poisson 1D body-source convergence")
    println("=" ^ 70)

    L      = 2.0           # bar length
    nx     = 16            # axial elements
    k_val  = 4.0           # conductivity
    Q      = 6.0           # uniform heat source

    mesh = _hex8_box(nx, 1, 1; Lx = L, Ly = 0.1, Lz = 0.1)
    cache, asm, kernel, m = _setup_heat(mesh; k_value = k_val)
    n = cache.ndofs

    # Pin the temperature on x == 0 and x == L (both faces, all 4 corners
    # of each face).
    nodes = m.nodes
    tol   = 1e-9
    fixed_dofs = Int[]
    for i in 1:length(nodes)
        x = nodes[i][1]
        if x < tol || x > L - tol
            push!(fixed_dofs, i)   # 1 DOF/node, so node id == DOF id
        end
    end

    K, _ = (assemble!(cache, asm, kernel, m); extract_system(cache))
    f    = zeros(n)
    apply_load!(f, UniformBodyForce(Q), cache, asm, kernel, m)

    bc = EliminatedDirichlet(fixed_dofs, zeros(length(fixed_dofs)))
    Kc = Matrix(K)
    bc_b = copy(f)
    apply_constraint!(Kc, bc_b, bc)

    T = Kc \ bc_b

    # Analytical T(x) = Q x (L − x) / (2 k) at every node
    T_ana = [Q * nodes[i][1] * (L - nodes[i][1]) / (2 * k_val)
             for i in 1:length(nodes)]

    rel = norm(T - T_ana) / max(norm(T_ana), 1.0)
    @test rel < 1e-10
    @test isapprox(maximum(T), Q * L^2 / (8 * k_val); rtol = 1e-10)

    println("  L=$L  nx=$nx  k=$k_val  Q=$Q   max(T)=$(round(maximum(T); sigdigits = 5))  " *
            "(expected $(round(Q * L^2 / (8 * k_val); sigdigits = 5)))   rel=$(round(rel; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 5. Zero allocations
# ----------------------------------------------------------------------------

@testset "Loads: zero allocations" begin
    println("\n" * "=" ^ 70)
    println("LOADS — ZERO-ALLOC")
    println("=" ^ 70)

    @testset "NodalForce alloc count" begin
        mesh = _hex8_box(2, 2, 2)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs
        load = NodalForce([3, 7, 9, 21], [10.0, -5.0, 2.5, 100.0])
        f = zeros(n)

        apply_load!(f, load, cache, asm, kernel, m)
        GC.gc()
        a = @allocated apply_load!(f, load, cache, asm, kernel, m)
        @test a == 0
        println("  NodalForce       ndof=$n   apply_load!=$a")
    end

    @testset "UniformBodyForce alloc count" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (3, 2, 2)]

        mesh = _hex8_box(nx, ny, nz)
        cache, asm, kernel, m = _setup_elasticity(mesh)
        n = cache.ndofs
        load = UniformBodyForce(Vec{3,Float64}((0.0, 0.0, -7850.0 * 9.81)))
        f = zeros(n)

        # warmup
        apply_load!(f, load, cache, asm, kernel, m)

        GC.gc()
        a = @allocated apply_load!(f, load, cache, asm, kernel, m)
        @test a == 0

        nelems = length(m.connectivity)
        println("  UniformBodyForce $(nx)×$(ny)×$(nz)  $(lpad(nelems,3)) elem  " *
                "$(lpad(n,4)) dof   apply_load!=$a")
    end
end
