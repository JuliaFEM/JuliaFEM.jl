# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Incompressible mixed u–p (`inv_bulk = 0`): pressure gauge, sparse direct solve,
matrix-free vs assembled mat-vec consistency, and sparse GMRES with field-split
Schur preconditioners (`ApproxSchurDiagBlockPreconditioner`,
`ApproxSchurICholDiagBlockPreconditioner`).
"""

using Test
using JuliaFEM
using JuliaFEM: MixedUPKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_structured_box_mesh, create_elements!, get_nodes_in_set, get_node_dofs
using JuliaFEM: default_pressure_gauge_dof, matrix_free_op
using JuliaFEM: global_field_ranges, ApproxSchurDiagBlockPreconditioner,
    ApproxSchurICholDiagBlockPreconditioner
using JuliaFEM: PenaltyDirichlet, apply_constraint!, apply_load!
using JuliaFEM: compute_diagonal!, apply_constraint_diag!
using JuliaFEM: NodalForce
using JuliaFEM: @DOFSet, DOF, Displacement, Vertex, Hex8
using LinearAlgebra
using SparseArrays
using IterativeSolvers: gmres!

function _mesh_up_box(nx::Int, ny::Int, nz::Int)
    return create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 1.0, nx = nx,
        ymin = 0.0, ymax = 1.0, ny = ny,
        zmin = 0.0, zmax = 1.0, nz = nz,
    )
end

function _collect_xmin_u_dofs(handler, mesh)
    d = Int[]
    for nid in get_nodes_in_set(mesh, :xmin)
        append!(d, get_node_dofs(handler, Int(nid)))
    end
    sort!(unique!(d))
    return d
end

"First mesh node with `x > 0.25` — not on `xmin` — return global uz DOF index."
function _interior_uz_dof(handler, mesh)
    for i in 1:length(mesh.nodes)
        if mesh.nodes[i][1] > 0.25
            nd = get_node_dofs(handler, i)
            return nd[3]
        end
    end
    error("no interior node for load")
end

@testset "MixedUP incompressible: direct sparse + pressure gauge" begin
    mesh = _mesh_up_box(2, 2, 2)
    mat = LinearElastic(E = 210e9, ν = 0.49)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 0.0)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = _collect_xmin_u_dofs(handler, mesh)
    p_pin = default_pressure_gauge_dof(handler; elem_id = 1)
    fixed = Int[vcat(u_fix, [p_pin])...]
    λ = 1e14
    bc = PenaltyDirichlet(fixed, zeros(Float64, length(fixed)); penalty = λ)

    uz = _interior_uz_dof(handler, mesh)
    apply_load!(f, NodalForce([uz], [1e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)
    uc = Kc \ f
    @test all(isfinite, uc)
    ru = norm(Kc * uc - f) / max(norm(f), 1.0)
    @test ru < 1e-8

    # Fixed displacement dofs ≈ 0 (penalty)
    @test maximum(abs.(uc[u_fix])) < 1e-4
    @test abs(uc[p_pin]) < 1e-4
end

@testset "MixedUP incompressible: matrix-free matches penalty-assembled K" begin
    mesh = _mesh_up_box(2, 2, 2)
    mat = LinearElastic(E = 210e9, ν = 0.49)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 0.0)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = _collect_xmin_u_dofs(handler, mesh)
    p_pin = default_pressure_gauge_dof(handler)
    fixed = Int[vcat(u_fix, [p_pin])...]
    bc = PenaltyDirichlet(fixed, zeros(Float64, length(fixed)); penalty = 1e14)

    uz = _interior_uz_dof(handler, mesh)
    apply_load!(f, NodalForce([uz], [1e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)

    mf = matrix_free_op(cache, asm, kernel, mesh; dirichlet = bc)
    @test isposdef(mf) == false

    n = length(f)
    x = randn(n)
    y_asm = Kc * x
    y_mf = zeros(n)
    mul!(y_mf, mf, x)
    @test norm(y_mf - y_asm) < 1e-10 * max(norm(y_asm), 1.0)

    # Jacobi diagonal matches assembled diagonal of the constrained operator
    d = zeros(n)
    compute_diagonal!(d, cache, asm, kernel, mesh)
    apply_constraint_diag!(d, bc)
    @test maximum(abs.(d .- diag(Matrix(Kc)))) < 1e-8 * max(maximum(abs.(diag(Matrix(Kc)))), 1.0)
end

@testset "MixedUP incompressible: GMRES + field-split Schur preconditioners" begin
    # Single Hex8 keeps `n` small so GMRES reaches tight tolerance with these
    # approximate Schur preconditioners (the 2³ brick needs many more iterations).
    mesh = _mesh_up_box(1, 1, 1)
    mat = LinearElastic(E = 210e9, ν = 0.49)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = MixedUPKernel(ContinuumFormulation{FullThreeD}(), mat; inv_bulk = 0.0)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    u_fix = _collect_xmin_u_dofs(handler, mesh)
    p_pin = default_pressure_gauge_dof(handler)
    fixed = Int[vcat(u_fix, [p_pin])...]
    bc = PenaltyDirichlet(fixed, zeros(Float64, length(fixed)); penalty = 1e14)

    uz = _interior_uz_dof(handler, mesh)
    apply_load!(f, NodalForce([uz], [1e4]), cache, asm, kernel, mesh)

    Kc = copy(K)
    apply_constraint!(Kc, bc)
    xref = Kc \ f

    n = length(f)
    ru, rp = global_field_ranges(handler)
    # IterativeSolvers.gmres! expects the matrix type `SparseMatrixCSC` here; the
    # constrained operator matches `matrix_free_op(...; dirichlet = bc)` mat-vecs
    # from `test_mixed_up_incompressible_solve` above.

    for PrType in (ApproxSchurDiagBlockPreconditioner, ApproxSchurICholDiagBlockPreconditioner)
        P = PrType(Kc, ru, rp)
        x = zeros(n)
        gmres!(
            x,
            Kc,
            f;
            Pl = P,
            restart = min(60, n),
            reltol = 1e-12,
            abstol = 1e-14,
            maxiter = 1500,
        )
        sol_err = norm(x - xref) / max(norm(xref), 1.0)
        @test sol_err < 1e-5
        res = Kc * x - f
        rrel = norm(res) / max(norm(f), 1.0)
        @test rrel < 1e-5
    end
end
