# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Reference tests for distributed-style matrix-free matvec **without MPI**:

`apply_K!` matches the sum of `apply_K_contributions!` over disjoint element
sets that partition the mesh, and [`prepare_multiply_workspace!`](@ref) /
[`MeshPartitionLayout`](@ref) behave as documented.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_K_contributions!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: MeshPartitionLayout, uniform_single_partition
using JuliaFEM: element_indices_for_part, validate_partition
using JuliaFEM: LocalMultiplyLayout, prepare_multiply_workspace!
using JuliaFEM: MatrixFreeOperator
using LinearAlgebra
using Random
using Tensors

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

function _build_single_tet4()
    nodes = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.5, 1.0, 0.0)),
        Vec{3}((0.5, 0.5, 1.0)),
    ]
    conns = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
    return Mesh{Tetrahedron{4}}(nodes, conns)
end

function _setup(mesh, ::Type{Topo}) where {Topo}
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Topo, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh, elements
end

"""Two-block element partition: ids 1:half → part 1, rest → part 2."""
function _two_block_layout(nelements::Int)
    nelements ≥ 1 || throw(ArgumentError("need at least one element"))
    half = nelements ÷ 2
    ids = ones(Int, nelements)
    @inbounds for i in (half + 1):nelements
        ids[i] = 2
    end
    return MeshPartitionLayout(ids), 1:half, (half + 1):nelements
end

@testset "partitioned matvec reference (two fake ranks)" begin
    Random.seed!(20260509)

    @testset "single Tet4 — empty block + full block" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m, elements = _setup(mesh, Tetrahedron{4})
        ne = length(elements)
        @test ne == 1
        layout, r1, r2 = _two_block_layout(ne)
        validate_partition(layout, ne)
        @test isempty(r1)
        @test collect(r2) == [1]

        e1 = element_indices_for_part(layout, 1)
        e2 = element_indices_for_part(layout, 2)
        @test isempty(e1)
        @test e2 == [1]

        n = cache.ndofs
        x = randn(n)
        y_full = zeros(n)
        apply_K!(y_full, cache, asm, kernel, m, x)

        ya = zeros(n)
        yb = zeros(n)
        apply_K_contributions!(ya, cache, asm, kernel, m, x, e1)
        apply_K_contributions!(yb, cache, asm, kernel, m, x, e2)
        @test ya + yb ≈ y_full rtol = 1e-12 atol = 1e-12
    end

    @testset "Hex8 box — two non-empty blocks" begin
        mesh = _build_hex8_box(3, 2, 2)
        cache, asm, kernel, m, elements = _setup(mesh, Hexahedron{8})
        ne = length(elements)
        layout, r1, r2 = _two_block_layout(ne)
        validate_partition(layout, ne)
        @test sort!(collect(union(Set(r1), Set(r2)))) == collect(1:ne)

        e1 = element_indices_for_part(layout, 1)
        e2 = element_indices_for_part(layout, 2)
        @test sort!(vcat(e1, e2)) == collect(1:ne)

        n = cache.ndofs
        for trial in 1:6
            x = randn(n)
            y_full = zeros(n)
            apply_K!(y_full, cache, asm, kernel, m, x)

            ya = zeros(n)
            yb = zeros(n)
            apply_K_contributions!(ya, cache, asm, kernel, m, x, e1)
            apply_K_contributions!(yb, cache, asm, kernel, m, x, e2)
            @test ya + yb ≈ y_full rtol = 1e-11 atol = 1e-11
        end
    end

    @testset "uniform_single_partition matches apply_K!" begin
        mesh = _build_hex8_box(2, 2, 2)
        cache, asm, kernel, m, elements = _setup(mesh, Hexahedron{8})
        ne = length(elements)
        layout = uniform_single_partition(ne)
        validate_partition(layout, ne)
        eall = element_indices_for_part(layout, 1)
        @test length(eall) == ne

        n = cache.ndofs
        x = randn(n)
        y_full = zeros(n)
        apply_K!(y_full, cache, asm, kernel, m, x)
        yc = zeros(n)
        apply_K_contributions!(yc, cache, asm, kernel, m, x, eall)
        @test yc ≈ y_full rtol = 1e-12 atol = 1e-12
    end

    @testset "invalid element id" begin
        mesh = _build_single_tet4()
        cache, asm, kernel, m, _ = _setup(mesh, Tetrahedron{4})
        n = cache.ndofs
        x = zeros(n)
        y = zeros(n)
        @test_throws ArgumentError apply_K_contributions!(y, cache, asm, kernel, m, x, [0])
        @test_throws ArgumentError apply_K_contributions!(y, cache, asm, kernel, m, x, [2])
    end
end

@testset "multiply workspace layout + MatrixFreeOperator default" begin
    mesh = _build_hex8_box(2, 2, 2)
    cache, asm, kernel, m, _ = _setup(mesh, Hexahedron{8})
    n = cache.ndofs
    x = randn(n)
    work = zeros(n)
    prepare_multiply_workspace!(work, x, LocalMultiplyLayout())
    @test work ≈ x

    work2 = copy(x)
    prepare_multiply_workspace!(work2, x, LocalMultiplyLayout())
    @test work2 ≈ x

    op = MatrixFreeOperator(cache, asm, kernel, m)
    y = zeros(n)
    mul!(y, op, x)
    yref = zeros(n)
    apply_K!(yref, cache, asm, kernel, m, x)
    @test y ≈ yref rtol = 1e-12 atol = 1e-12
end
