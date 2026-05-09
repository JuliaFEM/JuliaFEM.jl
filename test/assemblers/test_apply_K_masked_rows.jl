# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
`apply_K_masked_rows!`: disjoint owned-row masks (vertex-field ownership) sum to
the global [`apply_K!`](@ref) when `x` is the full vector on each fake rank.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: apply_K!, apply_K_masked_rows!, apply_K_owned_rows!
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: brick_hex_partition_slabs, validate_partition
using JuliaFEM: node_partition_owner_min!, mark_owned_vertex_field_dofs!
using Random
using Tensors

@testset "apply_K_masked_rows!: partition masks sum to apply_K!" begin
    Random.seed!(20260511)
    nx, ny, nz = 3, 4, 2
    nparts = 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

    layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
    validate_partition(layout, length(elements))

    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    n = cache.ndofs
    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    y_sum = zeros(n)
    own = falses(n)
    for part in 1:nparts
        mark_owned_vertex_field_dofs!(own, handler, node_own, part)
        y_part = zeros(n)
        apply_K_masked_rows!(y_part, own, cache, asm, kernel, mesh, x)
        y_sum .+= y_part
    end
    @test y_sum ≈ y_ref rtol = 1e-11 atol = 1e-11
end

@testset "apply_K_masked_rows!: DimensionMismatch on mask length" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    n = cache.ndofs
    @test_throws DimensionMismatch apply_K_masked_rows!(
        zeros(n), falses(n - 1), cache, asm, kernel, mesh, zeros(n))
end

@testset "apply_K_owned_rows!: matches apply_K! on owned rows" begin
    Random.seed!(20260512)
    nx, ny, nz = 3, 3, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    n = cache.ndofs
    owned = falses(n)
    mark_owned_vertex_field_dofs!(owned, handler, node_own, 1)

    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    y_own = zeros(n)
    apply_K_owned_rows!(y_own, owned, cache, asm, kernel, mesh, x)

    @inbounds for i in 1:n
        if owned[i]
            @test y_own[i] ≈ y_ref[i]
        else
            @test y_own[i] == 0.0
        end
    end
end

@testset "apply_K_owned_rows!: partition sum equals apply_K!" begin
    Random.seed!(20260513)
    nx, ny, nz = 3, 4, 2
    nparts = 3
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

    layout = brick_hex_partition_slabs(nx, ny, nz, nparts; axis = :y)
    validate_partition(layout, length(elements))

    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)

    n = cache.ndofs
    x = randn(n)
    y_ref = zeros(n)
    apply_K!(y_ref, cache, asm, kernel, mesh, x)

    y_sum = zeros(n)
    own = falses(n)
    for part in 1:nparts
        mark_owned_vertex_field_dofs!(own, handler, node_own, part)
        y_part = zeros(n)
        apply_K_owned_rows!(y_part, own, cache, asm, kernel, mesh, x)
        y_sum .+= y_part
    end
    @test y_sum ≈ y_ref rtol = 1e-11 atol = 1e-11
end

@testset "apply_K_owned_rows!: agrees with apply_K_masked_rows!" begin
    nx, ny, nz = 2, 3, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :z)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    n = cache.ndofs
    keep = falses(n)
    mark_owned_vertex_field_dofs!(keep, handler, node_own, 2)
    x = randn(n)
    y_m = zeros(n)
    y_o = zeros(n)
    apply_K_masked_rows!(y_m, keep, cache, asm, kernel, mesh, x)
    apply_K_owned_rows!(y_o, keep, cache, asm, kernel, mesh, x)
    @test y_m ≈ y_o
end

@testset "apply_K_owned_rows!: DimensionMismatch on mask length" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    n = cache.ndofs
    @test_throws DimensionMismatch apply_K_owned_rows!(
        zeros(n), falses(n - 1), cache, asm, kernel, mesh, zeros(n))
end

@testset "apply_K_masked_rows!: zero allocations after warmup" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    n = cache.ndofs
    keep = falses(n)
    mark_owned_vertex_field_dofs!(keep, handler, node_own, 1)
    y = zeros(n)
    x = randn(n)
    apply_K_masked_rows!(y, keep, cache, asm, kernel, mesh, x)
    a = @allocated apply_K_masked_rows!(y, keep, cache, asm, kernel, mesh, x)
    @test a == 0
end

@testset "apply_K_owned_rows!: zero allocations after warmup" begin
    nx, ny, nz = 2, 2, 2
    mesh = create_structured_box_mesh(Hex8; nx = nx, ny = ny, nz = nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    layout = brick_hex_partition_slabs(nx, ny, nz, 2; axis = :x)
    validate_partition(layout, length(elements))
    nnodes = length(mesh.nodes)
    node_own = Vector{Int}(undef, nnodes)
    node_partition_owner_min!(node_own, layout, mesh)
    n = cache.ndofs
    keep = falses(n)
    mark_owned_vertex_field_dofs!(keep, handler, node_own, 1)
    y = zeros(n)
    x = randn(n)
    apply_K_owned_rows!(y, keep, cache, asm, kernel, mesh, x)
    a = @allocated apply_K_owned_rows!(y, keep, cache, asm, kernel, mesh, x)
    @test a == 0
end
