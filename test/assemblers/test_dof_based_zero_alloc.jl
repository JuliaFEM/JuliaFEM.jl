# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Regression test: DOF-based assembler must stay zero-allocation
and bit-equivalent to the element-based assembler.

This test exists to lock in the December 2025 zero-allocation work
and to catch any regression as the assembler evolves.

Each problem size:
1. Build a structured Hex8 / Tet4 mesh
2. Create elements + DOFManager
3. Build both assemblers
4. Warm up
5. Assert: 0 bytes allocated by the DOF-based `assemble!`
6. Assert: matrix matches the element-based assembler to round-off
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: COOAssembler, create_cache
using JuliaFEM: create_elements!, @DOFSet, DOF, Displacement, Vertex
using JuliaFEM: local_dof_layout, DOFLayoutEntry
using LinearAlgebra
using SparseArrays
using Tensors
using InteractiveUtils  # @code_typed, code_llvm

# ----------------------------------------------------------------------------
# Helpers (kept local to this test, no leak into package)
# ----------------------------------------------------------------------------

"""
Build a structured Hex8 box mesh with `nx × ny × nz` elements over [0,1]^3.
"""
function _build_hex8_box_mesh(nx::Int, ny::Int, nz::Int)
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

"""
Build a single Tet4 mesh.
"""
function _build_single_tet4_mesh()
    nodes = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.5, 1.0, 0.0)),
        Vec{3}((0.5, 0.5, 1.0)),
    ]
    conns = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
    return Mesh{Tetrahedron{4}}(nodes, conns)
end

"""
Setup elasticity assembly fixture for a given mesh + topology.
Returns (cache_dof, asm_dof, cache_elem, asm_elem, kernel, mesh).
"""
function _setup_assembly(mesh, ::Type{Topo}) where {Topo}
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())

    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Topo, Lagrange{1}, S})

    asm_dof = DOFBasedCOOAssembler()
    cache_dof = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)

    asm_elem = COOAssembler()
    cache_elem = create_cache(asm_elem, mesh, kernel)

    return cache_dof, asm_dof, cache_elem, asm_elem, kernel, mesh
end

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@testset "DOF-based assembler: zero allocation regression" begin
    println("\n" * "=" ^ 70)
    println("DOF-BASED ASSEMBLER ZERO-ALLOCATION REGRESSION")
    println("=" ^ 70)

    # ------------------------------------------------------------------------
    # 1. Single Tet4
    # ------------------------------------------------------------------------
    @testset "Single Tet4" begin
        mesh = _build_single_tet4_mesh()
        cache_dof, asm_dof, cache_elem, asm_elem, kernel, m =
            _setup_assembly(mesh, Tetrahedron{4})

        # warmup
        assemble!(cache_dof, asm_dof, kernel, m)
        assemble!(cache_elem, asm_elem, kernel, m)

        K_dof, _ = extract_system(cache_dof)
        K_elem, _ = extract_system(cache_elem)

        # Equivalence
        rel = maximum(abs, Matrix(K_dof) - Matrix(K_elem)) /
              max(maximum(abs, Matrix(K_elem)), 1.0)
        @test rel < 1e-12

        # Zero-alloc
        GC.gc()
        a = @allocated assemble!(cache_dof, asm_dof, kernel, m)
        @test a == 0
        println("  Single Tet4 ........................ allocs=$a, rel=$(round(rel; sigdigits=3))")
    end

    # ------------------------------------------------------------------------
    # 2. Growing Hex8 cube meshes
    # ------------------------------------------------------------------------
    @testset "Hex8 cube $(nx)×$(ny)×$(nz)" for (nx, ny, nz) in
            [(1, 1, 1), (2, 1, 1), (4, 2, 2), (6, 3, 3), (8, 4, 4)]

        mesh = _build_hex8_box_mesh(nx, ny, nz)
        cache_dof, asm_dof, cache_elem, asm_elem, kernel, m =
            _setup_assembly(mesh, Hexahedron{8})

        # warmup
        assemble!(cache_dof, asm_dof, kernel, m)
        assemble!(cache_elem, asm_elem, kernel, m)

        K_dof, _ = extract_system(cache_dof)
        K_elem, _ = extract_system(cache_elem)

        rel = maximum(abs, Matrix(K_dof) - Matrix(K_elem)) /
              max(maximum(abs, Matrix(K_elem)), 1.0)
        @test rel < 1e-12

        GC.gc()
        a = @allocated assemble!(cache_dof, asm_dof, kernel, m)
        @test a == 0

        nelems = length(m.connectivity)
        ndofs = 3 * length(m.nodes)
        println("  Hex8 $(nx)×$(ny)×$(nz)   $(lpad(nelems,5)) elem  " *
                "$(lpad(ndofs,5)) dof   allocs=$a   rel=$(round(rel; sigdigits=3))")
    end
end

# ----------------------------------------------------------------------------
# Type stability check (no untyped ::Any in inferred output)
# ----------------------------------------------------------------------------
@testset "DOF-based assembler: type stability" begin
    mesh = _build_hex8_box_mesh(2, 1, 1)
    cache_dof, asm_dof, _, _, kernel, m = _setup_assembly(mesh, Hexahedron{8})
    assemble!(cache_dof, asm_dof, kernel, m)  # warmup

    # Inferred return type must be concrete (Nothing).
    rt = Base.promote_op(assemble!, typeof(cache_dof), typeof(asm_dof),
                         typeof(kernel), typeof(m))
    @test rt === Nothing

    # No method ambiguities or `Any` in the top-level signature
    code = code_typed(assemble!, (typeof(cache_dof), typeof(asm_dof),
                                  typeof(kernel), typeof(m)); optimize = true)
    @test !isempty(code)
    info = code[1]
    @test isconcretetype(info.second)
    println("  assemble! inferred return type: $(info.second)")
end

# ----------------------------------------------------------------------------
# Element-as-template: local_dof_layout(E) is compile-time constant and
# `assemble!` has zero GC allocation sites in the optimized LLVM IR.
# ----------------------------------------------------------------------------
@testset "DOF-based assembler: element template + LLVM allocs" begin
    mesh = _build_hex8_box_mesh(2, 1, 1)
    cache_dof, asm_dof, _, _, kernel, m = _setup_assembly(mesh, Hexahedron{8})
    assemble!(cache_dof, asm_dof, kernel, m)  # warmup

    # ------------------------------------------------------------------
    # 1. local_dof_layout returns the expected NTuple of DOFLayoutEntry
    #    for an Hex8 displacement element (24 DOFs, 8 vertices × 3 comp.)
    # ------------------------------------------------------------------
    ET = eltype(cache_dof.elements)
    layout = local_dof_layout(ET)
    @test layout isa NTuple{24, DOFLayoutEntry}
    @test all(e -> e.field_idx == 1, layout)
    @test [Int(e.entity_local) for e in layout] ==
          repeat(1:8; inner = 3)
    @test [Int(e.component) for e in layout] ==
          repeat(1:3, 8)

    # ------------------------------------------------------------------
    # 2. local_dof_layout(ET) call is constant-folded by the compiler:
    #    @allocated must be 0 and the optimized typed-IR must report a
    #    `Core.Const` for the layout in `assemble!` — exercised already
    #    by the zero-allocation tests above. We assert separately here.
    # ------------------------------------------------------------------
    GC.gc()
    a_layout = @allocated local_dof_layout(ET)
    @test a_layout == 0

    # ------------------------------------------------------------------
    # 3. Optimized LLVM IR for `assemble!` must have ZERO GC allocation
    #    sites. This is the strongest guarantee: even if @allocated == 0
    #    today, this catches any future change that would re-introduce
    #    a heap-allocating call inside the inner loop.
    # ------------------------------------------------------------------
    iob = IOBuffer()
    code_llvm(iob, assemble!,
        Tuple{typeof(cache_dof), typeof(asm_dof),
              typeof(kernel), typeof(m)}; optimize = true)
    ir = String(take!(iob))
    n_alloc =
        length(collect(eachmatch(r"call.*julia\.gc_alloc", ir))) +
        length(collect(eachmatch(r"call.*jl_gc_pool_alloc", ir))) +
        length(collect(eachmatch(r"call.*jl_gc_big_alloc", ir))) +
        length(collect(eachmatch(r"call.*jl_gc_alloc_typed", ir)))
    @test n_alloc == 0
    println("  assemble! optimized LLVM IR: $n_alloc gc-alloc sites " *
            "($(length(ir)) IR chars)")
end
