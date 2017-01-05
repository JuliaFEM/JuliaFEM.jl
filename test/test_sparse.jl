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
