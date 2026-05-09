# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Analytical and manufactured-solution benchmarks for linear elasticity on `Hex8`.

External references: the same classical checks underpin many commercial validation suites,
including Code_Aster manual V6 (validation index:
https://www.code-aster.org/V2/doc/default/en/index.php?man=V ; consolidated table of
contents: https://biba1632.gitlab.io/code-aster-manuals/docs/validation/v6_toc.html).

Cases implemented here:

1. Uniform traction rod — Saint–Venant-type uniaxial stress σ on the free end of a slender
   bar with fixed root; centre-line extension scales as u_z ≈ σ L / E (Timoshenko and
   Goodier, Theory of Elasticity).

2. Manufactured linear displacement — u = (a x, b y, c z) on the entire boundary; the
   exact field lies in the trilinear Hex8 space on a box, so the FE solve recovers it
   (patch consistency; Irons and Razzaque, Int. J. Num. Meth. Eng., 1972).

3. Simple shear linear field — u = (γ y, 0, 0) on the boundary; shear-dominated kinematics
   with no body force.

All solves use `EliminatedDirichlet` on structured meshes from
[`create_structured_box_mesh`](@ref).
"""

using Test
using LinearAlgebra
using Tensors
using JuliaFEM
using JuliaFEM: ContinuumKernel, ContinuumFormulation, FullThreeD, LinearElastic
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: EliminatedDirichlet, apply_constraint!, apply_load!
using JuliaFEM: SurfaceLoad, @DOFSet, DOF, Displacement, Vertex, Hex8
using JuliaFEM: create_elements!, create_structured_box_mesh, Element, Lagrange
using JuliaFEM: get_nodes_in_set, get_node_dofs

function _collect_vertex_dofs_on_nodes(handler, mesh, nodes::AbstractVector{<:Integer})
    d = Int[]
    for nid in nodes
        append!(d, get_node_dofs(handler, Int(nid)))
    end
    sort!(unique!(d))
    return d
end

function _collect_boundary_vertex_nodes(mesh)
    s = Set{UInt32}()
    for sym in (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
        union!(s, get_nodes_in_set(mesh, sym))
    end
    return collect(s)
end

"""Quad faces on `z = zmax` for a structured `Hex8` box (`nx`, `ny`, `nz` segments)."""
function _zmax_quad_faces(nx::Int, ny::Int, nz::Int)
    faces = NTuple{4,Int}[]
    nix(i, j, k) = (k - 1) * (nx + 1) * (ny + 1) + (j - 1) * (nx + 1) + i
    for j in 1:ny, i in 1:nx
        push!(faces, (
            Int(nix(i, j, nz + 1)),
            Int(nix(i + 1, j, nz + 1)),
            Int(nix(i + 1, j + 1, nz + 1)),
            Int(nix(i, j + 1, nz + 1)),
        ))
    end
    return faces
end

@testset "Reference elasticity: uniform traction rod (σ L / E)" begin
    # Slender bar along Z; traction σ on zmax; clamp zmin (all components).
    L = 10.0
    σ = 1.0e6
    E = 210.0e9
    ν = 0.3
    nx = ny = 2
    nz = 40
    mesh = create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 1.0, nx = nx,
        ymin = 0.0, ymax = 1.0, ny = ny,
        zmin = 0.0, zmax = L, nz = nz,
    )
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), LinearElastic(E = E, ν = ν))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    faces = _zmax_quad_faces(nx, ny, nz)
    apply_load!(f, SurfaceLoad(faces, Vec((0.0, 0.0, σ))), cache, asm, kernel, mesh)

    fix = _collect_vertex_dofs_on_nodes(handler, mesh, get_nodes_in_set(mesh, :zmin))
    bc = EliminatedDirichlet(fix, zeros(Float64, length(fix)))

    Kc = copy(K)
    apply_constraint!(Kc, f, bc)
    u = Kc \ f

    @test all(isfinite, u)
    ru = norm(Kc * u - f) / max(norm(f), 1.0)
    @test ru < 1.0e-8

    uz_tip_expected = σ * L / E
    mid_i, mid_j = div(nx, 2) + 1, div(ny, 2) + 1
    nix(i, j, k) = (k - 1) * (nx + 1) * (ny + 1) + (j - 1) * (nx + 1) + i
    nid = Int(nix(mid_i, mid_j, nz + 1))
    ud = get_node_dofs(handler, nid)
    uz_num = u[Int(ud[3])]
    @test abs(uz_num - uz_tip_expected) / uz_tip_expected < 5.0e-3
end

@testset "Reference elasticity: manufactured u = (a x, b y, c z)" begin
    a, b, c = 1.0e-4, -2.0e-4, 3.0e-4
    nx = ny = nz = 4
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    E = 70.0e9
    ν = 0.33
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), LinearElastic(E = E, ν = ν))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    bn = _collect_boundary_vertex_nodes(mesh)
    dof_ids = Int[]
    vals = Float64[]
    for nid in bn
        x, y, z = mesh.nodes[Int(nid)][1], mesh.nodes[Int(nid)][2], mesh.nodes[Int(nid)][3]
        d = get_node_dofs(handler, Int(nid))
        push!(dof_ids, Int(d[1]), Int(d[2]), Int(d[3]))
        push!(vals, a * x, b * y, c * z)
    end
    bc = EliminatedDirichlet(dof_ids, vals)

    Kc = copy(K)
    apply_constraint!(Kc, f, bc)
    u = Kc \ f

    err_max = 0.0
    for k in 1:length(mesh.nodes)
        x, y, z = mesh.nodes[k][1], mesh.nodes[k][2], mesh.nodes[k][3]
        d = get_node_dofs(handler, k)
        ue = (a * x, b * y, c * z)
        for α in 1:3
            err_max = max(err_max, abs(u[Int(d[α])] - ue[α]))
        end
    end
    @test err_max < 1.0e-9 * max(abs(a), abs(b), abs(c))
end

@testset "Reference elasticity: manufactured simple shear u = (γ y, 0, 0)" begin
    γ = 2.5e-4
    nx = 3
    ny = 5
    nz = 4
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
        LinearElastic(E = 200.0e9, ν = 0.29))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    bn = _collect_boundary_vertex_nodes(mesh)
    dof_ids = Int[]
    vals = Float64[]
    for nid in bn
        y = mesh.nodes[Int(nid)][2]
        d = get_node_dofs(handler, Int(nid))
        push!(dof_ids, Int(d[1]), Int(d[2]), Int(d[3]))
        push!(vals, γ * y, 0.0, 0.0)
    end
    bc = EliminatedDirichlet(dof_ids, vals)

    Kc = copy(K)
    apply_constraint!(Kc, f, bc)
    u = Kc \ f

    err_max = 0.0
    for k in 1:length(mesh.nodes)
        y = mesh.nodes[k][2]
        d = get_node_dofs(handler, k)
        ue = (γ * y, 0.0, 0.0)
        for α in 1:3
            err_max = max(err_max, abs(u[Int(d[α])] - ue[α]))
        end
    end
    scale = max(abs(γ), 1.0)
    @test err_max < 1.0e-9 * scale
end
