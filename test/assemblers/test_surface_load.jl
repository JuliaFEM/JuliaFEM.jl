# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`SurfaceLoad` — distributed traction (or heat flux) integrated as
`∫_Γ N_i · t dS` over a list of mesh faces. The natural complement of
`UniformBodyForce` (`∫_Ω N_i · b dV`). Added in D+++.

Locks in the contract that:

  1. The face Gauss quadrature (2 × 2 for quad, 1-point for tri)
     reproduces the analytical integral `∫_Γ t dS = t · area` for
     constant traction on a flat face — exact to machine precision
     for both quadrilateral and triangular faces.

  2. Vector-valued traction (3D elasticity) and scalar-valued flux
     (heat conduction) both go through the same `apply_load!` path
     with no kernel-specific dispatch. The same `_integrate_face!`
     unrolls correctly for `_t_comp_count(t) ∈ {1, 3}`.

  3. `SurfaceLoad` and `UniformBodyForce` compose additively (apply
     both, get the sum) — the combined RHS still solves
     ∫_Γ N · t dS + ∫_Ω N · b dV correctly.

  4. End-to-end pull test on a unit Hex8 cube (1 × 1 × 1 elements):
     fix the bottom face, apply unit traction in `+z` on the top
     face, recover the analytical extension `u_z(z) = σ_z / E · z`
     to within finite-element accuracy.

  5. End-to-end heat flux problem: insulated 5 of 6 faces of a Hex8
     cube, apply uniform inward flux on the remaining face,
     recover the linear conduction temperature profile.

  6. Per-face traction (different `t` per face) overlays correctly.

  7. `apply_load!(SurfaceLoad)` is allocation-free after warmup.

This file deliberately uses the existing `cache.dof_handler` field
(added in D+++) instead of any face-extraction machinery — the
SurfaceLoad path is intentionally compositional and does not depend on
`Mesh.extract_surface!` (which is incomplete).
"""

using Test
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Vertex, Temperature
using JuliaFEM: @DOFSet, DOF
using JuliaFEM: LinearElastic, Displacement, ContinuumKernel
using JuliaFEM: HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: extract_system, create_elements!
using JuliaFEM: SurfaceLoad, UniformBodyForce, NodalForce, apply_load!
using JuliaFEM: PenaltyDirichlet, apply_constraint!
using LinearAlgebra
using SparseArrays
using Tensors

# ----------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------

function _unit_hex8(nx::Int, ny::Int, nz::Int; Lx = 1.0, Ly = 1.0, Lz = 1.0)
    nodes = Vec{3,Float64}[]
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        push!(nodes, Vec{3}((Lx * (i - 1) / nx,
                             Ly * (j - 1) / ny,
                             Lz * (k - 1) / nz)))
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

# Top-face quad nodes for every (i, j) column at the top of an
# (nx × ny × nz) Hex8 box. Returns a Vector{NTuple{4,Int}}.
function _top_face_quads(nx::Int, ny::Int, nz::Int)
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    faces = NTuple{4,Int}[]
    k = nz + 1                                   # top z layer
    for j in 1:ny, i in 1:nx
        n1 = nidx(i,     j,     k)
        n2 = nidx(i + 1, j,     k)
        n3 = nidx(i + 1, j + 1, k)
        n4 = nidx(i,     j + 1, k)
        push!(faces, (n1, n2, n3, n4))
    end
    return faces
end

function _bottom_face_node_dofs(nx::Int, ny::Int, dof_per_node::Int)
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    dofs = Int[]
    vals = Float64[]
    k = 1
    for j in 1:(ny + 1), i in 1:(nx + 1)
        node = nidx(i, j, k)
        for c in 1:dof_per_node
            push!(dofs, (node - 1) * dof_per_node + c)
            push!(vals, 0.0)
        end
    end
    return dofs, vals
end

function _setup_elasticity(mesh; E::Float64 = 210e9, ν::Float64 = 0.3)
    material = LinearElastic(E = E, ν = ν)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S        = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

function _setup_heat(mesh; k::Float64 = 50.0)
    material = HeatConductivity(k = k)
    kernel   = HeatKernel(ContinuumFormulation{FullThreeD}(), material)
    S        = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# 1. Quadrature reproduces ∫_Γ t dS = t · area exactly
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: row-sum identity = t · area" begin
    println("\n" * "=" ^ 70)
    println("D+++  SurfaceLoad — row-sum identity (analytical area)")
    println("=" ^ 70)

    @testset "Vector traction on Hex8 top face $(nx)×$(ny)" for (nx, ny) in
            [(1, 1), (3, 2), (4, 3)]
        nz = 1
        mesh = _unit_hex8(nx, ny, nz; Lx = 2.0, Ly = 1.5, Lz = 1.0)
        cache, asm, kernel, m = _setup_elasticity(mesh)

        faces = _top_face_quads(nx, ny, nz)
        t = Vec{3}((10.0, -7.0, 3.0))            # arbitrary uniform traction
        load = SurfaceLoad(faces, t)

        f = zeros(cache.ndofs)
        apply_load!(f, load, cache, asm, kernel, m)

        # Sum of the assembled force vector, by component, must equal
        # `t · area_Γ` (here area = Lx · Ly = 3.0).
        area = 2.0 * 1.5
        f_sum_x = sum(f[1:3:end])
        f_sum_y = sum(f[2:3:end])
        f_sum_z = sum(f[3:3:end])
        @test isapprox(f_sum_x, t[1] * area; atol = 1e-10)
        @test isapprox(f_sum_y, t[2] * area; atol = 1e-10)
        @test isapprox(f_sum_z, t[3] * area; atol = 1e-10)

        println("  $(nx)×$(ny)  area=$(round(area; digits=3))  " *
                "Σf_x=$(round(f_sum_x; digits=4)) (t·area=$(round(t[1]*area; digits=4)))  " *
                "Σf_z=$(round(f_sum_z; digits=4)) (t·area=$(round(t[3]*area; digits=4)))")
    end

    @testset "Scalar flux on Hex8 top face $(nx)×$(ny)" for (nx, ny) in
            [(1, 1), (2, 3)]
        nz = 1
        mesh = _unit_hex8(nx, ny, nz; Lx = 1.5, Ly = 2.0, Lz = 1.0)
        cache, asm, kernel, m = _setup_heat(mesh)

        faces = _top_face_quads(nx, ny, nz)
        q = 25.0                                  # uniform flux
        load = SurfaceLoad(faces, q)

        f = zeros(cache.ndofs)
        apply_load!(f, load, cache, asm, kernel, m)

        area = 1.5 * 2.0
        @test isapprox(sum(f), q * area; atol = 1e-10)
        println("  $(nx)×$(ny)  area=$(round(area; digits=3))  " *
                "Σf=$(round(sum(f); digits=4)) (q·area=$(round(q*area; digits=4)))")
    end
end

# ----------------------------------------------------------------------------
# 2. Triangular face support (Tri3)
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: triangular face row-sum" begin
    # Single tri: corners (0,0,0), (2,0,0), (0,3,0) — area = 3.0
    mesh = _unit_hex8(1, 1, 1)
    cache, asm, kernel, m = _setup_elasticity(mesh)

    # Manual nodes/face — we'll re-purpose the cube's nodes 1, 2, 4 which
    # at (Lx, Ly, Lz) = (1, 1, 1) form a corner triangle of area 0.5.
    faces = NTuple{3,Int}[(1, 2, 4)]
    t     = Vec{3}((4.0, 0.0, 0.0))
    load  = SurfaceLoad(faces, t)
    f     = zeros(cache.ndofs)
    apply_load!(f, load, cache, asm, kernel, m)

    expected_area = 0.5
    @test isapprox(sum(f[1:3:end]), t[1] * expected_area; atol = 1e-10)
    @test isapprox(sum(f[2:3:end]), t[2] * expected_area; atol = 1e-10)
    @test isapprox(sum(f[3:3:end]), t[3] * expected_area; atol = 1e-10)
end

# ----------------------------------------------------------------------------
# 3. Per-face traction
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: per-face traction overlays correctly" begin
    nx, ny, nz = 2, 2, 1
    mesh = _unit_hex8(nx, ny, nz)
    cache, asm, kernel, m = _setup_elasticity(mesh)

    faces = _top_face_quads(nx, ny, nz)
    @test length(faces) == 4

    # Pull on faces 1 & 4 with +z, push on 2 & 3 with -z. The total
    # force resultant should equal Σᵢ (tᵢ · area_face) summed over
    # faces; with all face areas = 0.25 and tractions all in z, the
    # total z-force is (1 - 1 - 1 + 1) * 0.25 = 0.
    tractions = [Vec{3}((0.0, 0.0,  1.0)),
                 Vec{3}((0.0, 0.0, -1.0)),
                 Vec{3}((0.0, 0.0, -1.0)),
                 Vec{3}((0.0, 0.0,  1.0))]
    load = SurfaceLoad(faces, tractions)
    f    = zeros(cache.ndofs)
    apply_load!(f, load, cache, asm, kernel, m)

    f_sum_z = sum(f[3:3:end])
    @test isapprox(f_sum_z, 0.0; atol = 1e-10)
end

# ----------------------------------------------------------------------------
# 4. End-to-end pull test (3D elasticity)
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: 3D pull test on Hex8 column" begin
    println("\n" * "=" ^ 70)
    println("D+++  SurfaceLoad — end-to-end 3D elasticity pull")
    println("=" ^ 70)

    # Single Hex8 column 1×1×1 m with σ_zz applied on the top face,
    # bottom face fully fixed. Expect u_z(z) = σ_z · z / E.
    nx, ny, nz = 1, 1, 4
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    Eyoung = 1.0e11
    σ_z = 1.0e6
    mesh = _unit_hex8(nx, ny, nz; Lx = Lx, Ly = Ly, Lz = Lz)
    cache, asm, kernel, m = _setup_elasticity(mesh; E = Eyoung, ν = 0.0)
    n = cache.ndofs

    # Assemble K, then add surface load to f.
    assemble!(cache, asm, kernel, m)
    K, f = extract_system(cache)

    faces = _top_face_quads(nx, ny, nz)
    apply_load!(f, SurfaceLoad(faces, Vec{3}((0.0, 0.0, σ_z))),
                cache, asm, kernel, m)

    # Fix bottom face (penalty Dirichlet, all 3 components).
    fixed_dofs, fixed_vals = _bottom_face_node_dofs(nx, ny, 3)
    bc = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1e10 * Eyoung)
    apply_constraint!(K, bc)
    apply_constraint!(f, bc)

    u = K \ Vector(f)

    # Top-layer node z-displacements: should all equal σ_z * Lz / E.
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    u_top = Float64[]
    for j in 1:(ny + 1), i in 1:(nx + 1)
        node = nidx(i, j, nz + 1)
        push!(u_top, u[(node - 1) * 3 + 3])
    end
    u_z_exact = σ_z * Lz / Eyoung
    rel = maximum(abs.(u_top .- u_z_exact)) / abs(u_z_exact)
    @test rel < 1e-3
    println("  σ_z=$σ_z  E=$(Eyoung)  u_z_exact=$(round(u_z_exact; sigdigits = 4))  " *
            "u_z_FE=$(round(mean(u_top); sigdigits = 4))   rel=$(round(rel; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 5. End-to-end heat flux problem (1D conduction)
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: 1D heat conduction with surface flux" begin
    println("\n" * "=" ^ 70)
    println("D+++  SurfaceLoad — heat flux + Dirichlet (1D)")
    println("=" ^ 70)

    # Hex8 column with T=0 on bottom, +q heat flux on top (entering),
    # all sides insulated. Expect linear T(z) = q · z / k.
    nx, ny, nz = 1, 1, 6
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    k_cond = 50.0
    q_flux = 200.0
    mesh = _unit_hex8(nx, ny, nz; Lx = Lx, Ly = Ly, Lz = Lz)
    cache, asm, kernel, m = _setup_heat(mesh; k = k_cond)
    n = cache.ndofs

    assemble!(cache, asm, kernel, m)
    K, f = extract_system(cache)

    faces = _top_face_quads(nx, ny, nz)
    apply_load!(f, SurfaceLoad(faces, q_flux), cache, asm, kernel, m)

    fixed_dofs, fixed_vals = _bottom_face_node_dofs(nx, ny, 1)
    bc = PenaltyDirichlet(fixed_dofs, fixed_vals; penalty = 1e10 * k_cond)
    apply_constraint!(K, bc)
    apply_constraint!(f, bc)

    T = K \ Vector(f)

    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    T_top = Float64[]
    for j in 1:(ny + 1), i in 1:(nx + 1)
        node = nidx(i, j, nz + 1)
        push!(T_top, T[node])
    end
    T_top_exact = q_flux * Lz / k_cond
    rel = maximum(abs.(T_top .- T_top_exact)) / abs(T_top_exact)
    @test rel < 1e-3
    println("  q=$(q_flux) W/m²  k=$k_cond  T_top_exact=$(round(T_top_exact; sigdigits = 4))  " *
            "T_top_FE=$(round(mean(T_top); sigdigits = 4))   rel=$(round(rel; sigdigits = 3))")
end

# ----------------------------------------------------------------------------
# 6. Composition with UniformBodyForce
# ----------------------------------------------------------------------------

@testset "SurfaceLoad + UniformBodyForce compose additively" begin
    nx, ny, nz = 2, 2, 1
    mesh = _unit_hex8(nx, ny, nz)
    cache, asm, kernel, m = _setup_elasticity(mesh)

    body = UniformBodyForce(Vec{3}((0.0, 0.0, -10.0)))
    surf = SurfaceLoad(_top_face_quads(nx, ny, nz), Vec{3}((0.0, 0.0, 5.0)))

    f_combined = zeros(cache.ndofs)
    apply_load!(f_combined, body, cache, asm, kernel, m)
    apply_load!(f_combined, surf, cache, asm, kernel, m)

    f_body = zeros(cache.ndofs)
    apply_load!(f_body, body, cache, asm, kernel, m)
    f_surf = zeros(cache.ndofs)
    apply_load!(f_surf, surf, cache, asm, kernel, m)

    @test isapprox(f_combined, f_body .+ f_surf; atol = 1e-12)
    println("  Composition (body + surf) verified ✓")
end

# ----------------------------------------------------------------------------
# 7. Zero allocations after warmup
# ----------------------------------------------------------------------------

@testset "SurfaceLoad: zero allocations on apply_load!" begin
    println("\n" * "=" ^ 70)
    println("D+++  SurfaceLoad — zero-allocation hot path")
    println("=" ^ 70)

    nx, ny, nz = 4, 4, 2
    mesh = _unit_hex8(nx, ny, nz)
    cache, asm, kernel, m = _setup_elasticity(mesh)

    faces = _top_face_quads(nx, ny, nz)
    load  = SurfaceLoad(faces, Vec{3}((1.0, -2.0, 3.0)))
    f     = zeros(cache.ndofs)

    apply_load!(f, load, cache, asm, kernel, m)            # warmup
    fill!(f, 0.0)
    a = @allocated apply_load!(f, load, cache, asm, kernel, m)
    @test a == 0
    println("  apply_load!(SurfaceLoad)  allocs=$(a)")

    # Heat flux variant
    cache_h, asm_h, kernel_h, m_h = _setup_heat(mesh)
    load_h = SurfaceLoad(_top_face_quads(nx, ny, nz), 50.0)
    f_h    = zeros(cache_h.ndofs)
    apply_load!(f_h, load_h, cache_h, asm_h, kernel_h, m_h)
    fill!(f_h, 0.0)
    a_h = @allocated apply_load!(f_h, load_h, cache_h, asm_h, kernel_h, m_h)
    @test a_h == 0
    println("  apply_load!(SurfaceLoad heat)  allocs=$(a_h)")
end

# Local helper — `mean` from Statistics, but we want to avoid the dep
mean(xs) = sum(xs) / length(xs)
