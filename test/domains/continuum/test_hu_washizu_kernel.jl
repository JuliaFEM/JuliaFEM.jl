# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`HuWashizuKernel`: three-field (u, ε̃, σ̃) on the DOF-based assembler.

1. Strain–strain block matches `vol * M`.
2. Strain–stress / stress–strain blocks match `−vol*G` and `+vol*G`.
3. `K_uu`, `K_uε`, `K_εu`, `K_σσ` are zero.
4. Coupling satisfies `K_uσ + K_σu^\\mathsf{T} \\approx 0` (symmetric global `K`).
5. `assemble!` allocates 0 bytes after warmup.
"""

using Test
using JuliaFEM
using JuliaFEM: HuWashizuKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_unit_cube_mesh, create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: matrix_free_op
using LinearAlgebra
using SparseArrays
using Tensors

function _hex8_unit_cube_single_element()
    return create_unit_cube_mesh(Hex8; nx = 1, ny = 1, nz = 1)
end

@testset "HuWashizuKernel Voigt blocks vs vol·M, vol·G" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{
        u::DOF{Displacement{3}, Vertex},
        eps::DOF{SymmetricTensor{2,3}, Cell},
        sig::DOF{SymmetricTensor{2,3}, Cell},
    }
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HuWashizuKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    assemble!(cache, asm, k, mesh)
    K, _ = extract_system(cache)

    n_u = 3 * length(mesh.nodes)
    r_ε = (n_u + 1):(n_u + 6)
    r_σ = (n_u + 7):(n_u + 12)
    vol = 1.0

    Kee = Matrix(K[r_ε, r_ε])
    @test norm(Kee - vol * k.M) / max(norm(k.M), 1.0) < 1e-12

    Kes = Matrix(K[r_ε, r_σ])
    @test norm(Kes + vol * k.G) / max(norm(k.G), 1.0) < 1e-12

    Kse = Matrix(K[r_σ, r_ε])
    @test norm(Kse - vol * k.G) / max(norm(k.G), 1.0) < 1e-12
end

@testset "HuWashizuKernel empty blocks + u–σ skew coupling" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{
        u::DOF{Displacement{3}, Vertex},
        eps::DOF{SymmetricTensor{2,3}, Cell},
        sig::DOF{SymmetricTensor{2,3}, Cell},
    }
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HuWashizuKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    assemble!(cache, asm, k, mesh)
    K, _ = extract_system(cache)

    n_u = 3 * length(mesh.nodes)
    r_u = 1:n_u
    r_ε = (n_u + 1):(n_u + 6)
    r_σ = (n_u + 7):(n_u + 12)

    @test norm(K[r_u, r_u]) < 1e-14
    @test norm(K[r_u, r_ε]) < 1e-14
    @test norm(K[r_ε, r_u]) < 1e-14
    @test norm(K[r_σ, r_σ]) < 1e-14

    Kus = Matrix(K[r_u, r_σ])
    Ksu = Matrix(K[r_σ, r_u])
    skew = Kus + transpose(Ksu)
    @test norm(skew) / max(norm(Kus), 1.0) < 1e-10

    R = K - transpose(K)
    @test norm(R) <= 1e-8 * max(1.0, norm(K))
end

@testset "HuWashizuKernel MatrixFreeOperator isposdef" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{
        u::DOF{Displacement{3}, Vertex},
        eps::DOF{SymmetricTensor{2,3}, Cell},
        sig::DOF{SymmetricTensor{2,3}, Cell},
    }
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HuWashizuKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    op = matrix_free_op(cache, asm, k, mesh)
    @test LinearAlgebra.isposdef(op) == false
end

@testset "HuWashizuKernel assemble! zero allocations" begin
    mesh = _hex8_unit_cube_single_element()
    mat = LinearElastic(E = 210e9, ν = 0.3)
    S = @DOFSet{
        u::DOF{Displacement{3}, Vertex},
        eps::DOF{SymmetricTensor{2,3}, Cell},
        sig::DOF{SymmetricTensor{2,3}, Cell},
    }
    elem, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    k = HuWashizuKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elem, handler, mesh, k)
    for _ in 1:3
        assemble!(cache, asm, k, mesh)
    end
    GC.gc()
    @test (@allocated assemble!(cache, asm, k, mesh)) == 0
end

@testset "HuWashizuKernel API guards" begin
    mat = LinearElastic(E = 210e9, ν = 0.3)
    k = HuWashizuKernel(ContinuumFormulation{FullThreeD}(), mat)
    @test dofs_per_node(k) == 6
    @test_throws ErrorException get_field(k)
end
