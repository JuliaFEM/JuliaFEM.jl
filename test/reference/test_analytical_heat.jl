# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Steady heat conduction on a `Hex8` box with Dirichlet temperatures on two opposing faces
and natural (insulating) conditions elsewhere.

Exact solution (piecewise-linear in z): `T(z) = T1 * (z - z_min) / (z_max - z_min)`, which
satisfies Laplace's equation and the boundary data. Diffusion analogues appear alongside
mechanical cases in Code_Aster validation manual V (table of contents:
https://biba1632.gitlab.io/code-aster-manuals/docs/validation/v_toc.html).
"""

using Test
using LinearAlgebra
using JuliaFEM
using JuliaFEM: ContinuumFormulation, FullThreeD, Temperature, Vertex
using JuliaFEM: @DOFSet, DOF, HeatConductivity, HeatKernel
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: EliminatedDirichlet, apply_constraint!
using JuliaFEM: create_elements!, create_structured_box_mesh, Element, Lagrange, Hex8
using JuliaFEM: get_nodes_in_set, get_node_dofs

@testset "Reference heat: linear profile T(z) between zmin and zmax" begin
    zmin_v = 0.1
    zmax_v = 0.65
    T1 = 57.0
    kcond = 45.0
    nx = 4
    ny = 5
    nz = 12
    mesh = create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 2.3, nx = nx,
        ymin = 0.0, ymax = 1.7, ny = ny,
        zmin = zmin_v, zmax = zmax_v, nz = nz,
    )
    S = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), HeatConductivity(k = kcond))
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)

    n_zmin = sort!(collect(get_nodes_in_set(mesh, :zmin)))
    n_zmax = sort!(collect(get_nodes_in_set(mesh, :zmax)))
    dof_fix = Int[]
    val_fix = Float64[]
    for nid in n_zmin
        push!(dof_fix, Int(only(get_node_dofs(handler, Int(nid)))))
        push!(val_fix, 0.0)
    end
    for nid in n_zmax
        push!(dof_fix, Int(only(get_node_dofs(handler, Int(nid)))))
        push!(val_fix, T1)
    end
    bc = EliminatedDirichlet(dof_fix, val_fix)

    Kc = copy(K)
    apply_constraint!(Kc, f, bc)
    Tvec = Kc \ f

    @test all(isfinite, Tvec)
    ru = norm(Kc * Tvec - f) / max(norm(f), 1.0)
    @test ru < 1.0e-10

    denom = zmax_v - zmin_v
    err_max = 0.0
    for k in 1:length(mesh.nodes)
        z = mesh.nodes[k][3]
        d = Int(only(get_node_dofs(handler, k)))
        Tex = T1 * (z - zmin_v) / denom
        err_max = max(err_max, abs(Tvec[d] - Tex))
    end
    @test err_max < 1.0e-9 * max(abs(T1), 1.0)
end
