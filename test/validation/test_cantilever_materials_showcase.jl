# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# # Cantilever beam: elasticity materials + heat (showcase / regression)
#
# This script is a normal `Test` module (run via `test/validation/runtests.jl`).
# It is also written in [Literate.jl](https://github.com/JuliaDocs/Literate.jl)
# style: lines starting with `# #` become markdown headings if you pass this
# file through `Literate.markdown` from the `docs` environment.
#
# Goals:
#
# 1. One small structured Hex8 cantilever (`create_cantilever_mesh`): fixed
#    at `:xmin`, transverse load on `:xmax`.
# 2. Every **solid** constitutive model used by `ContinuumKernel` today:
#    `LinearElastic`, `NeoHookean`, `PerfectPlasticity` — same mesh and BCs.
# 3. `HeatConductivity` with `HeatKernel` on the **same geometry** (scalar
#    temperature at vertices), because there is no meaningful “cantilever”
#    for Fourier’s equation — only the same assembly pipeline.
# 4. After warmup, `assemble!(cache, asm, kernel, mesh)` must allocate
#    **0 bytes** (same contract as `test/assemblers/test_dof_based_zero_alloc.jl`).
#
# The sparse **solve** (`K_ff \\ f_f`) is not part of the zero-allocation
# contract; only the DOF-based assembly hot path is.

using Test
using JuliaFEM
using JuliaFEM: ContinuumKernel, ContinuumFormulation, FullThreeD, Displacement
using JuliaFEM: HeatKernel, HeatConductivity, Temperature
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache, assemble!, extract_system
using JuliaFEM: create_cantilever_mesh, create_elements!, get_nodes_in_set, get_node_dofs
using JuliaFEM: @DOFSet, DOF, Vertex, Hex8
using LinearAlgebra
using SparseArrays

# ## Shared mesh and BC helpers

function _small_cantilever_mesh()
    return create_cantilever_mesh(Hex8;
        length = 5.0,
        width = 1.0,
        height = 1.0,
        nx = 4,
        ny = 1,
        nz = 1,
    )
end

function _collect_fixed_dofs(handler, mesh)
    fixed = Int[]
    for nid_raw in get_nodes_in_set(mesh, :xmin)
        nid = Int(nid_raw)
        append!(fixed, get_node_dofs(handler, nid))
    end
    sort!(unique!(fixed))
    return fixed
end

function _apply_tip_shear!(f, handler, mesh; Fz::Float64)
    loaded = get_nodes_in_set(mesh, :xmax)
    nL = length(loaded)
    @assert nL > 0
    fz = Fz / nL
    for nid_raw in loaded
        nd = get_node_dofs(handler, Int(nid_raw))
        @assert length(nd) == 3
        f[nd[3]] += fz
    end
    return nothing
end

function _solve_eliminated(K::SparseMatrixCSC, f::Vector{Float64}, fixed_dofs::Vector{Int})
    ndofs = length(f)
    all_idx = 1:ndofs
    is_fixed = falses(ndofs)
    for d in fixed_dofs
        is_fixed[d] = true
    end
    free = Int[d for d in all_idx if !is_fixed[d]]
    Kff = K[free, free]
    ff = f[free]
    uf = Kff \ ff
    u = zeros(ndofs)
    u[free] = uf
    return u, free
end

@testset "Cantilever showcase: materials + zero-allocation assembly" begin
    mesh = _small_cantilever_mesh()
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    fixed_dofs = _collect_fixed_dofs(handler, mesh)

    E_young = 210e9
    ν = 0.3
    materials = (
        ("LinearElastic", LinearElastic(E = E_young, ν = ν)),
        ("NeoHookean", NeoHookean(E_mod = E_young, nu = ν)),
        ("PerfectPlasticity", PerfectPlasticity(E = E_young, ν = ν, σ_y = 350e6, H = 1e9)),
    )

    asm = DOFBasedCOOAssembler()

    for (name, mat) in materials
        @testset "$name — assemble! 0 allocs, finite solve" begin
            kernel = ContinuumKernel(ContinuumFormulation{FullThreeD}(), mat)
            cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

            for _ in 1:3
                assemble!(cache, asm, kernel, mesh)
            end
            GC.gc()
            bytes = @allocated assemble!(cache, asm, kernel, mesh)
            @test bytes == 0

            K, f0 = extract_system(cache)
            f = copy(f0)
            _apply_tip_shear!(f, handler, mesh; Fz = -50_000.0)
            u, _free = _solve_eliminated(K, f, fixed_dofs)

            @test all(isfinite, u)
            @test norm(u) > 1e-12
            # Tip should move in the direction of the applied shear (negative Z load).
            tip_ids = get_nodes_in_set(mesh, :xmax)
            uz_sum = 0.0
            for nid in tip_ids
                nd = get_node_dofs(handler, Int(nid))
                uz_sum += u[nd[3]]
            end
            @test uz_sum / length(tip_ids) < 0.0
        end
    end
end

@testset "Same geometry: HeatConductivity — assemble! 0 allocs" begin
    mesh = _small_cantilever_mesh()
    S = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
    mat = HeatConductivity(k = 45.0)
    kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), mat)
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)

    for _ in 1:3
        assemble!(cache, asm, kernel, mesh)
    end
    GC.gc()
    @test (@allocated assemble!(cache, asm, kernel, mesh)) == 0

    K, _ = extract_system(cache)
    @test size(K, 1) == handler.total_dofs
    @test nnz(K) > 0
    R = K - transpose(K)
    @test norm(R) <= 1e-8 * max(1.0, norm(K))
end
