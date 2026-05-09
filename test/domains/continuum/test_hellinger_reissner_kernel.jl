# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`HellingerReissnerKernel`: displacement + piecewise-constant symmetric stress
on the DOF-based assembler.

1. Global `K` is numerically symmetric.
2. `K_uu` is zero (no displacement Laplace term in classical HR).
3. `K_σσ` equals `-vol · (G M⁻¹ G)` on the stress block.
4. `assemble!` allocates 0 bytes after warmup.
"""

using Test
using JuliaFEM
using JuliaFEM: HellingerReissnerKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex
using LinearAlgebra
using Tensors

function _hex8_unit_cube_single_element()
    return create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
end

@testset "HellingerReissnerKernel symmetry + empty displacement block" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, sig::DOF{SymmetricTensor{2,3}, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HellingerReissnerKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    assemble!(cache, asm, k, mesh)
    K, _ = extract_system(cache)

    R = K - transpose(K)
    @test norm(R) <= 1e-8 * max(1.0, norm(K))

    n_u = 3 * length(mesh.nodes)
    Kuu = K[1:n_u, 1:n_u]
    @test norm(Kuu) <= 1e-10 * max(1.0, norm(K))
end

@testset "HellingerReissnerKernel K_σσ vs −vol·(G M⁻¹ G)" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, sig::DOF{SymmetricTensor{2,3}, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HellingerReissnerKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    assemble!(cache, asm, k, mesh)
    K, _ = extract_system(cache)

    n_u = 3 * length(mesh.nodes)
    Kss = K[(n_u + 1):end, (n_u + 1):end]
    vol = 1.0  # unit cube
    ref = -vol * k.σσ
    @test size(Kss) == (6, 6)
    @test norm(Kss - ref) / max(norm(ref), 1.0) < 1e-12
end

@testset "HellingerReissnerKernel assemble! zero allocations" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}, sig::DOF{SymmetricTensor{2,3}, Cell}}
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HellingerReissnerKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    for _ in 1:3
        assemble!(cache, asm, k, mesh)
    end
    GC.gc()
    @test (@allocated assemble!(cache, asm, k, mesh)) == 0
end

@testset "HellingerReissnerKernel API guards" begin
    mat = LinearElastic(E = 210e9, ν = 0.3)
    k = HellingerReissnerKernel(ContinuumFormulation{FullThreeD}(), mat)
    @test dofs_per_node(k) == 5
    @test_throws ErrorException get_field(k)
end
