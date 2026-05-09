# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Regression tests for the typed `MatrixFreeOperator` /
`MatrixFreeMassOperator`. The contract is:

  1. `mul!(y, op, x)` matches `K * x` (or `M * x`) to round-off, and
     also matches the closure-style `op(y, x)` invocation.
  2. `op * x` allocates an output vector and produces the same answer.
  3. `eltype(op) == Float64` and `size(op) == (n, n)`.
  4. The operator plugs into `LinearOperators.LinearOperator(...)` and
     drives `IterativeSolvers.cg!` to a correct solution.
  5. A `MatrixFreeOperator` built with `dirichlet = PenaltyDirichlet`
     reproduces `K + λ · diag(eᵈ)` row-by-row.
  6. `mul!` after the warmup is allocation-free.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFBasedCOOAssembler, DOFBasedCOOCache
using JuliaFEM: MatrixFreeOperator, MatrixFreeMassOperator, matrix_free_op
using JuliaFEM: PenaltyDirichlet
using JuliaFEM: extract_system, apply_K!, apply_M!
using JuliaFEM: ContinuumKernel, ContinuumFormulation, FullThreeD
using JuliaFEM: HeatKernel, HeatConductivity
using JuliaFEM: LinearElastic, Displacement, Vertex, Temperature
using JuliaFEM: create_elements!, @DOFSet, DOF
using LinearAlgebra
using SparseArrays
using Tensors
using Random
using LinearOperators
using IterativeSolvers

# ----------------------------------------------------------------------------
# Mesh helper
# ----------------------------------------------------------------------------

function _hex8_unit_box(nx::Int, ny::Int, nz::Int)
    nodes = Vec{3,Float64}[]
    nidx(i, j, k) = (i - 1) + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1) + 1
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:(nx + 1)
        push!(nodes, Vec{3}((Float64(i - 1) / nx,
                             Float64(j - 1) / ny,
                             Float64(k - 1) / nz)))
    end
    conns = NTuple{8,UInt32}[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        n1 = nidx(i,     j,     k); n2 = nidx(i + 1, j,     k)
        n3 = nidx(i + 1, j + 1, k); n4 = nidx(i,     j + 1, k)
        n5 = nidx(i,     j,     k + 1); n6 = nidx(i + 1, j,     k + 1)
        n7 = nidx(i + 1, j + 1, k + 1); n8 = nidx(i,     j + 1, k + 1)
        push!(conns, (UInt32(n1), UInt32(n2), UInt32(n3), UInt32(n4),
                      UInt32(n5), UInt32(n6), UInt32(n7), UInt32(n8)))
    end
    return Mesh{8,Hexahedron{8}}(nodes, conns)
end

function _setup_elasticity(nx::Int, ny::Int, nz::Int)
    mesh     = _hex8_unit_box(nx, ny, nz)
    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())
    S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

function _setup_heat(nx::Int, ny::Int, nz::Int)
    mesh   = _hex8_unit_box(nx, ny, nz)
    cond   = HeatConductivity(k = 5.0)
    kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), cond)
    S = @DOFSet{T::DOF{Temperature, Vertex}}
    elements, dof_mgr = create_elements!(mesh, Element{Hexahedron{8}, Lagrange{1}, S})
    asm   = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, dof_mgr, mesh, kernel)
    return cache, asm, kernel, mesh
end

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@testset "MatrixFreeOperator: contract and behaviour" begin
    Random.seed!(20260508)

    @testset "Stiffness operator: mul! matches K * x" begin
        cache, asm, kernel, mesh = _setup_elasticity(3, 2, 2)
        assemble!(cache, asm, kernel, mesh)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        op = MatrixFreeOperator(cache, asm, kernel, mesh)

        @test eltype(op) == Float64
        @test size(op) == (n, n)
        @test size(op, 1) == n
        @test size(op, 2) == n

        for trial in 1:5
            x      = randn(n)
            y_ref  = K * x

            y_mul  = zeros(n)
            mul!(y_mul, op, x)
            @test norm(y_mul - y_ref) / norm(y_ref) < 1e-12

            y_call = zeros(n)
            op(y_call, x)
            @test y_call == y_mul

            y_mat  = op * x
            @test norm(y_mat - y_ref) / norm(y_ref) < 1e-12
        end
    end

    @testset "matrix_free_op factory returns MatrixFreeOperator" begin
        cache, asm, kernel, mesh = _setup_elasticity(2, 2, 2)
        op = matrix_free_op(cache, asm, kernel, mesh)
        @test op isa MatrixFreeOperator
    end

    @testset "Mass operator: mul! matches M * x" begin
        # Use heat kernel because the mass term is non-zero when the
        # heat capacity is enabled; ContinuumKernel/HeatConductivity
        # default mass entries are zero unless density is configured.
        cache, asm, kernel, mesh = _setup_heat(2, 2, 2)

        # Build the matrix-free mass operator and reference M via
        # apply_M! (rather than the full assemble_M!) so we are
        # comparing the matrix-free path against itself column-wise.
        op_M = MatrixFreeMassOperator(cache, asm, kernel, mesh)
        n = size(op_M, 1)

        @test size(op_M) == (n, n)
        @test eltype(op_M) == Float64

        # Build a dense reference by mat-vec on canonical basis.
        M_ref = zeros(n, n)
        e_j   = zeros(n)
        for j in 1:n
            fill!(e_j, 0.0); e_j[j] = 1.0
            apply_M!(view(M_ref, :, j) |> collect, cache, asm, kernel, mesh, e_j)
            tmp = zeros(n)
            apply_M!(tmp, cache, asm, kernel, mesh, e_j)
            M_ref[:, j] .= tmp
        end

        for trial in 1:3
            x     = randn(n)
            y_ref = M_ref * x
            y_mul = zeros(n)
            mul!(y_mul, op_M, x)
            @test norm(y_mul - y_ref) / max(norm(y_ref), 1e-30) < 1e-10

            y_seed = randn(n)
            α = 1.7
            β = -0.25
            y_fused = copy(y_seed)
            mul!(y_fused, op_M, x, α, β)
            @test norm(y_fused - (α * y_ref + β * y_seed)) /
                  max(norm(y_ref), 1e-30) < 1e-10
        end
    end

    @testset "PenaltyDirichlet folded into mat-vec" begin
        cache, asm, kernel, mesh = _setup_heat(3, 2, 2)
        assemble!(cache, asm, kernel, mesh)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        fixed = [1, 4, 7]
        vals  = [1.0, 2.0, 3.0]
        λ     = 1.0e10
        c     = PenaltyDirichlet(fixed, vals; penalty = λ)

        K_pen = copy(K)
        for d in fixed
            K_pen[d, d] += λ
        end

        op = MatrixFreeOperator(cache, asm, kernel, mesh; dirichlet = c)
        x  = randn(n)
        y_ref = K_pen * x
        y_mf  = zeros(n)
        mul!(y_mf, op, x)
        @test norm(y_mf - y_ref) / norm(y_ref) < 1e-10
    end

    @testset "Plugs into LinearOperators + IterativeSolvers.cg!" begin
        cache, asm, kernel, mesh = _setup_heat(3, 3, 3)
        assemble!(cache, asm, kernel, mesh)
        K, _ = extract_system(cache)
        n    = size(K, 1)

        # Constrain the boundary so K is invertible.
        fixed = collect(1:n) |> dofs -> filter(d -> d in [1, n], dofs)
        c     = PenaltyDirichlet(fixed, fill(0.0, length(fixed)); penalty = 1.0e12)

        op    = MatrixFreeOperator(cache, asm, kernel, mesh; dirichlet = c)
        linop = LinearOperator(Float64, size(op, 1), size(op, 2), true, true, op)

        K_pen = copy(K)
        for d in fixed
            K_pen[d, d] += 1.0e12
        end
        b   = randn(n)
        u_ref = K_pen \ b

        u_cg  = zeros(n)
        cg!(u_cg, linop, b; abstol = 1e-12, reltol = 1e-12, maxiter = 4n)
        @test norm(u_cg - u_ref) / norm(u_ref) < 1e-6
    end

    @testset "mul! is allocation-free after warmup" begin
        cache, asm, kernel, mesh = _setup_elasticity(2, 2, 2)
        op = MatrixFreeOperator(cache, asm, kernel, mesh)
        n  = size(op, 1)
        x  = randn(n)
        y  = zeros(n)

        # Warmup (compile + first-call alloc on Pass 1).
        mul!(y, op, x)
        mul!(y, op, x)

        allocs = @allocated mul!(y, op, x)
        @test allocs == 0
    end
end
