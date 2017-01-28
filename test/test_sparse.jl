# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "Add to SparseMatrixCOO" begin
    A = SparseMatrixCOO()
    A2 = reshape(collect(1:9), 3, 3)
    add!(A, sparse(A2))
    @test isapprox(full(A), full(A2))
end

@testset "Add to SparseVectorCOO" begin
    b = SparseVectorCOO()
    b2 = collect(1:3)
    add!(b, sparse(b2))
    @test isapprox(full(b), full(b2))
end

@testset "Failure to add data to sparse vector due dimensino mismatch" begin
    b = SparseVectorCOO()
    @test_throws ErrorException add!(b, [1, 2], [1.0, 2.0, 3.0])
end

@testset "Test combining of SparseMatrixCOO" begin
    k = convert(Matrix{Float64}, reshape(collect(1:9), 3, 3))
    dofs1 = [1, 2, 3]
    dofs2 = [2, 3, 4]
    A = SparseMatrixCOO()
    add!(A, dofs1, dofs1, k)
    add!(A, dofs2, dofs2, k)
    A1 = full(A)
    optimize!(A)
    A2 = full(A)
    @test isapprox(A1, A2)
end

@testset "resize of sparse matrix and sparse vector" begin
    A = sparse(rand(3, 3))
    B = resize_sparse(A, 4, 4)
    @test size(B) == (4, 4)
    a = sparse(rand(3))
    b = resize_sparsevec(a, 4)
    @test size(b) == (4, )
end

