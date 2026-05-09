# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using LinearAlgebra: Symmetric, cross, dot, eigvals, norm
using Tensors: SymmetricTensor
using SparseArrays
using Test

function _single_unit_tet_mesh()
    nodes = Vec{3, Float64}[
        Vec((0.0, 0.0, 0.0)),
        Vec((1.0, 0.0, 0.0)),
        Vec((0.0, 1.0, 0.0)),
        Vec((0.0, 0.0, 1.0)),
    ]
    conn = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    return Mesh{4, Tet4}(nodes, [conn])
end

"""Two Tet4 elements sharing the triangular face (nodes 2, 3, 4)."""
function _two_tets_shared_face_mesh()
    nodes = Vec{3, Float64}[
        Vec((0.0, 0.0, 0.0)),
        Vec((1.0, 0.0, 0.0)),
        Vec((0.0, 1.0, 0.0)),
        Vec((0.0, 0.0, 1.0)),
        Vec((1.0, 1.0, 0.0)),
    ]
    conns = [
        (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),
        (UInt32(2), UInt32(5), UInt32(3), UInt32(4)),
    ]
    return Mesh{4, Tet4}(nodes, conns)
end

function _tet_volume_analytical(X::AbstractVector{Vec{3, Float64}})
    g1 = X[2] - X[1]
    g2 = X[3] - X[1]
    g3 = X[4] - X[1]
    return abs(dot(cross(g1, g2), g3)) / 6.0
end

@testset "DarcyMixedRT0P0Kernel block structure (one Tet4)" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    Ktarget = 2.0
    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = Ktarget))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    Ksp, _ = extract_system(cache)
    K = Matrix(Ksp)

    rσ, rp = global_field_ranges(handler)
    @test length(rσ) == 4
    @test length(rp) == 1

    Kσσ = K[rσ, rσ]
    Kσp = K[rσ, rp]
    Kpσ = K[rp, rσ]
    Kpp = K[rp, rp]

    @test Kσp ≈ fill(-1.0, 4, 1)
    @test Kpσ ≈ fill(1.0, 1, 4)
    @test norm(Kpp) == 0.0
    @test Kσσ ≈ Kσσ'

    λ = eigvals(Symmetric(Kσσ))
    @test all(λ .> 0)
end

@testset "DarcyMixedRT0P0Kernel K_uu scales with inv_k" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)

    function sigma_block(inv_k)
        kernel = DarcyMixedRT0P0Kernel(; inv_k)
        asm = DOFBasedCOOAssembler()
        cache = create_cache(asm, elements, handler, mesh, kernel)
        assemble!(cache, asm, kernel, mesh)
        Ksp, _ = extract_system(cache)
        K = Matrix(Ksp)
        rσ, _ = global_field_ranges(handler)
        return K[rσ, rσ]
    end

    A1 = sigma_block(1.0)
    A2 = sigma_block(3.0)
    @test A2 ≈ 3.0 .* A1
end

@testset "DarcyMixedRT0P0Kernel solveable with pressure gauge" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    Ksp, f = extract_system(cache)
    fill!(f, 0.0)

    p_dof = default_pressure_gauge_dof(handler; field_pressure = 2, elem_id = 1)
    bc = PenaltyDirichlet([p_dof], [0.0]; penalty = 1e14)
    Kc = copy(Ksp)
    fc = copy(f)
    apply_constraint!(Kc, bc)
    apply_constraint!(fc, bc)

    x = Kc \ fc
    @test length(x) == handler.total_dofs
    @test abs(x[p_dof]) < 1e-6
end

@testset "two Tet4 sharing a face — DOF count and SPD flux block" begin
    mesh = _two_tets_shared_face_mesh()
    @test nelements(mesh) == 2

    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)

    rσ, rp = global_field_ranges(handler)
    @test length(rσ) == 7
    @test length(rp) == 2
    @test handler.total_dofs == 9

    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    Ksp, _ = extract_system(cache)
    K = Matrix(Ksp)

    Kσσ = K[rσ, rσ]
    @test Kσσ ≈ Kσσ'
    @test all(eigvals(Symmetric(Kσσ)) .> 0)

    Kσp = K[rσ, rp]
    Kpσ = K[rp, rσ]
    # Each P₀ pressure row sums ∫ div φ over the four faces of its tet (+1 each).
    @test vec(sum(Kpσ; dims = 2)) ≈ fill(4.0, length(rp))
    # Each pressure column touches four face flux DOFs with −1.
    @test vec(sum(Kσp; dims = 1)) ≈ fill(-4.0, length(rp))
end

@testset "UniformMixedDarcySource — ∫ q f on cell pressures" begin
    mesh = _two_tets_shared_face_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)

    X1 = Vec{3, Float64}[mesh.nodes[Int(i)] for i in mesh.connectivity[1]]
    X2 = Vec{3, Float64}[mesh.nodes[Int(i)] for i in mesh.connectivity[2]]
    V1 = _tet_volume_analytical(X1)
    V2 = _tet_volume_analytical(X2)

    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)

    src = 3.5
    apply_load!(cache.f, UniformMixedDarcySource(src), cache, asm, kernel, mesh)

    pstarts = handler.field_starts[2]
    @test cache.f[pstarts[1]] ≈ src * V1
    @test cache.f[pstarts[2]] ≈ src * V2
    rσ, _ = global_field_ranges(handler)
    @test norm(cache.f[rσ]) == 0.0
end

@testset "tet_facet_gid_from_corners and global_facet_dof" begin
    mesh = _two_tets_shared_face_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    maps = handler.facet_maps::Tet4FacetMaps

    gid_shared = tet_facet_gid_from_corners(mesh, maps, (4, 2, 3))
    @test gid_shared != 0
    @test maps.elem_face_gid[3, 1] == gid_shared
    @test maps.elem_face_gid[4, 2] == gid_shared

    σ_expect = global_facet_dof(handler, 1, gid_shared)
    layout = local_dof_layout(elements[1])
    li_face3 = 0
    @inbounds for li in 1:length(layout)
        e = layout[li]
        if Int(field_idx(e)) == 1 && Int(entity_local(e)) == 3 && Int(component(e)) == 1
            li_face3 = li
            break
        end
    end
    @test li_face3 > 0
    @test σ_expect == Int(elements[1].dof_indices[li_face3])

    @test tet_facet_gid_from_corners(mesh, maps, (1, 2, 5)) == 0
end

@testset "mixed Darcy prescribed flux on facet (PenaltyDirichlet)" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    maps = handler.facet_maps::Tet4FacetMaps

    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    Ksp, f = extract_system(cache)
    fill!(f, 0.0)

    gid123 = tet_facet_gid_from_corners(mesh, maps, (2, 1, 3))
    σ_dof = global_facet_dof(handler, 1, gid123)
    p_dof = default_pressure_gauge_dof(handler; field_pressure = 2, elem_id = 1)

    bc_σ = PenaltyDirichlet([σ_dof], [2.25]; penalty = 1e14)
    bc_p = PenaltyDirichlet([p_dof], [0.0]; penalty = 1e14)
    Km = Matrix(Ksp)
    apply_constraint!(Km, bc_σ)
    apply_constraint!(Km, bc_p)
    fc = copy(f)
    apply_constraint!(fc, bc_σ)
    apply_constraint!(fc, bc_p)

    x = Km \ fc
    @test abs(x[σ_dof] - 2.25) < 1e-5
    @test abs(x[p_dof]) < 1e-5
end

@testset "DarcyMixedRT0P0Kernel tensor inv_K matches isotropic inv_k" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    asm = DOFBasedCOOAssembler()

    function K_dense(kern)
        cache = create_cache(asm, elements, handler, mesh, kern)
        assemble!(cache, asm, kern, mesh)
        Ksp, _ = extract_system(cache)
        return Matrix(Ksp)
    end

    Kiso = K_dense(DarcyMixedRT0P0Kernel(; inv_k = 2.25))
    I6 = one(SymmetricTensor{2,3, Float64, 6})
    Kten = K_dense(DarcyMixedRT0P0Kernel(2.25 * I6))
    @test Kiso ≈ Kten
end

@testset "DarcyMixedHex8RT0P0Kernel divergence coupling (one Hex8)" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Hex8, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)

    rσ, rp = global_field_ranges(handler)
    @test length(rσ) == 6
    @test length(rp) == 1

    kernel = DarcyMixedHex8RT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    Ksp, _ = extract_system(cache)
    K = Matrix(Ksp)

    Kσp = K[rσ, rp]
    Kpσ = K[rp, rσ]
    @test vec(sum(Kpσ; dims = 2)) ≈ fill(6.0, length(rp))
    @test sum(Kσp) ≈ -6.0
end

@testset "UniformMixedDarcySource on Hex8 brick" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Hex8, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    kernel = DarcyMixedHex8RT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)

    vol = sum(cache.geometry_caches[1].detJ_w)
    src = 1.25
    apply_load!(cache.f, UniformMixedDarcySource(src), cache, asm, kernel, mesh)
    pdof = handler.field_starts[2][1]
    @test cache.f[pdof] ≈ src * vol
end

@testset "MixedDarcyTet4BoundaryNormalFluxLoad — full boundary vs divergence theorem" begin
    mesh = _single_unit_tet_mesh()
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Tet4, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    kernel = DarcyMixedRT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    fill!(cache.f, 0.0)

    g = 1.625
    panels = Tuple{Int, Int}[(1, lf) for lf in 1:4]
    apply_load!(cache.f, MixedDarcyTet4BoundaryNormalFluxLoad(panels, g), cache, asm, kernel, mesh)

    rσ, _ = global_field_ranges(handler)
    # ∫_{∂Ω} g φᵢ·n dS = g ∫_{∂Ω} φᵢ·n dS = g ∫_Ω ∇·φᵢ dV = g · 1 (mixed kernel coupling row).
    @test all(isapprox.(cache.f[rσ], g; rtol = 1e-11, atol = 1e-11))
end

@testset "MixedDarcyHex8BoundaryNormalFluxLoad — uniform g on all faces" begin
    mesh = create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
    S = @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}}
    ET = Element{Hex8, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    kernel = DarcyMixedHex8RT0P0Kernel(HydraulicConductivity(K = 1.0))
    asm = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    fill!(cache.f, 0.0)

    g = 0.875
    panels = Tuple{Int, Int}[(1, lf) for lf in 1:6]
    apply_load!(cache.f, MixedDarcyHex8BoundaryNormalFluxLoad(panels, g), cache, asm, kernel, mesh)

    rσ, _ = global_field_ranges(handler)
    # For one `[0,1]³` Hex8 mapped from `[-1,1]³`, `∫_{∂Ω} φᵢ·n dS = 4` for each RT₀ face basis
    # (4-point quad face rule); differs from the lumped `±1` pressure–flux coupling row.
    scale_flux = 4.0
    @test all(isapprox.(cache.f[rσ], g * scale_flux; rtol = 1e-11, atol = 1e-11))
end
