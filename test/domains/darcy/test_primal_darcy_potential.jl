# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "HydraulicConductivity validation" begin
    @test_throws ArgumentError HydraulicConductivity(K = 0.0)
    @test_throws ArgumentError HydraulicConductivity(K = -1.0)
    m = HydraulicConductivity(K = 1e-3)
    @test m.K == 1e-3
end

@testset "HeatKernel invalid material–field pairing" begin
    form = ContinuumFormulation{FullThreeD}()
    @test_throws ArgumentError HeatKernel(form, HeatConductivity(k = 1.0), PressurePotential())
    @test_throws ArgumentError HeatKernel(form, HydraulicConductivity(K = 1.0), Temperature())
end

@testset "Primal Darcy stiffness equals thermal diffusion (same K)" begin
    mesh = create_unit_cube_mesh(Hex8)
    κ = 12.7

    S_T = @DOFSet{T::DOF{Temperature, Vertex}}
    S_p = @DOFSet{p::DOF{PressurePotential, Vertex}}

    k_heat = HeatKernel(ContinuumFormulation{FullThreeD}(), HeatConductivity(k = κ))
    k_darcy = DarcyPotentialKernel(ContinuumFormulation{FullThreeD}(), HydraulicConductivity(K = κ))

    el_T, h_T = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_T})
    el_p, h_p = create_elements!(mesh, Element{Hex8, Lagrange{1}, S_p})

    asm = DOFBasedCOOAssembler()
    cache_T = create_cache(asm, el_T, h_T, mesh, k_heat)
    cache_p = create_cache(asm, el_p, h_p, mesh, k_darcy)

    assemble!(cache_T, asm, k_heat, mesh)
    assemble!(cache_p, asm, k_darcy, mesh)

    K_T, _ = extract_system(cache_T)
    K_p, _ = extract_system(cache_p)

    @test Matrix(K_T) ≈ Matrix(K_p)

    x = rand(Float64, h_T.total_dofs)
    y_T = similar(x)
    y_p = similar(x)
    apply_K!(y_T, cache_T, asm, k_heat, mesh, x)
    apply_K!(y_p, cache_p, asm, k_darcy, mesh, x)
    @test y_T ≈ y_p
end

@testset "scalar_diffusion_tensor matches tensors" begin
    κ = 50.0
    hc = HeatConductivity(k = κ)
    hyd = HydraulicConductivity(K = κ)
    @test scalar_diffusion_tensor(hc) ≈ conductivity_tensor(hc)
    @test scalar_diffusion_tensor(hyd) ≈ hydraulic_conductivity_tensor(hyd)
end

function _box_nidx(nx::Int, ny::Int, nz::Int)
    return (i::Int, j::Int, k::Int) -> (k - 1) * (nx + 1) * (ny + 1) + (j - 1) * (nx + 1) + i
end

function _top_face_quads_hex_box(nx::Int, ny::Int, nz::Int)
    nidx = _box_nidx(nx, ny, nz)
    faces = NTuple{4, Int}[]
    k = nz + 1
    for j in 1:ny, i in 1:nx
        push!(
            faces,
            (nidx(i, j, k), nidx(i + 1, j, k), nidx(i + 1, j + 1, k), nidx(i, j + 1, k)),
        )
    end
    return faces
end

function _scalar_dofs_for_nodes(handler, node_ids)
    return [handler.field_starts[1][Int(n)] for n in node_ids]
end

@testset "SurfaceLoad Neumann flux + Dirichlet (primal Darcy)" begin
    nx, ny, nz = 1, 1, 6
    Kcond = 50.0
    q_flux = 200.0
    Lz = 1.0
    mesh = create_unit_cube_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{p::DOF{PressurePotential, Vertex}}
    kernel = DarcyPotentialKernel(ContinuumFormulation{FullThreeD}(), HydraulicConductivity(K = Kcond))
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)

    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)

    faces = _top_face_quads_hex_box(nx, ny, nz)
    apply_load!(f, SurfaceLoad(faces, q_flux), cache, asm, kernel, mesh)

    # Bottom face z = 0 (same convention as test/assemblers/test_surface_load.jl heat flux case).
    zmin_nodes = collect(get_node_set(mesh, :zmin))
    fixed_dofs = _scalar_dofs_for_nodes(handler, zmin_nodes)
    bc = PenaltyDirichlet(fixed_dofs, zeros(Float64, length(fixed_dofs)); penalty = 1e10 * Kcond)
    apply_constraint!(K, bc)
    apply_constraint!(f, bc)

    u = K \ Vector(f)
    zmax_nodes = collect(get_node_set(mesh, :zmax))
    p_top = [u[d] for d in _scalar_dofs_for_nodes(handler, zmax_nodes)]
    p_top_exact = q_flux * Lz / Kcond
    rel = maximum(abs.(p_top .- p_top_exact)) / abs(p_top_exact)
    @test rel < 1e-3
end

@testset "ElementWiseScalarDiffusion (two bricks in x)" begin
    nx, ny, nz = 2, 1, 1
    K1, K2 = 1.0, 4.0
    mesh = create_unit_cube_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    @test nelements(mesh) == 2

    mat = ElementWiseScalarDiffusion([K1, K2])
    kernel = DarcyPotentialKernel(ContinuumFormulation{FullThreeD}(), mat)
    S = @DOFSet{p::DOF{PressurePotential, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)

    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)

    nidx = _box_nidx(nx, ny, nz)
    dofs_lo = _scalar_dofs_for_nodes(handler, collect(get_node_set(mesh, :xmin)))
    dofs_hi = _scalar_dofs_for_nodes(handler, collect(get_node_set(mesh, :xmax)))
    dof_all = [dofs_lo; dofs_hi]
    vals = [zeros(Float64, length(dofs_lo)); ones(Float64, length(dofs_hi))]
    bc = PenaltyDirichlet(dof_all, vals; penalty = 1e10 * max(K1, K2))
    apply_constraint!(K, bc)
    apply_constraint!(f, bc)

    u = K \ Vector(f)
    q = 1.0 / (0.5 / K1 + 0.5 / K2)
    p_mid_exact = q * 0.5 / K1
    mid_node = nidx(2, 1, 1)
    p_mid = u[handler.field_starts[1][mid_node]]
    @test abs(p_mid - p_mid_exact) / p_mid_exact < 5e-3
end

@testset "ElementWiseScalarDiffusion length must match nelements" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 2, ny = 1, nz = 1)
    bad = ElementWiseScalarDiffusion([1.0, 2.0, 99.0])
    kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), bad, PressurePotential())
    S = @DOFSet{p::DOF{PressurePotential, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    @test_throws ArgumentError create_cache(asm, elements, handler, mesh, kernel)
end
