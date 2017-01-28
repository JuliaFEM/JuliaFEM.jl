# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "discrete, constant, time invariant field" begin
    @test isa(DCTI(), DCTI)
    @test DCTI(0.0).data == 0.0
    @test isa(Field(0.0), DCTI)
    @test isa(Field(), DCTI)
    f = DCTI()
    update!(f, 1.0)
    @test f.data == 1.0
    @test f == DCTI(1.0)
    @test isapprox(f, DCTI(1.0))
    @test 2*f == 2.0 # multiply by constant
    @test f(1.0) == 1.0 # time interpolation
    @test isapprox([2.0]''*f, 2.0) # wanted behavior?
    @test length(f) == 1 # always
end

@testset "discrete, variable, time invariant field" begin
    @test isa(DVTI(), DVTI)
    @test DVTI([1.0, 2.0]).data == [1.0, 2.0]
    @test isa(Field([1.0, 2.0]), DVTI)
    
    f = DVTI()
    update!(f, [2.0, 3.0])
    @test isapprox(f.data, [2.0, 3.0])
    @test length(f) == 2

    # slicing
    @test isapprox(f[1], 2.0)
    @test isapprox(f[[1, 2]], [2.0, 3.0])

    # boolean comparison and multiplying by a constant
    @test f == DVTI([2.0, 3.0])
    @test isapprox(2*f, [4.0, 6.0])

    f3 = 2*f
    @test isa(f3, DVTI)
    @test f3+f == 3*f
    @test f3-f == f

    # spatial interpolation
    N = [1.0, 2.0]
    @test isapprox(N*f, 8.0)

    # time interpolation
    @test isapprox(f(1.0), [2.0, 3.0])

    # spatial interpolation of vector valued variable field
    f2 = DVTI(Vector{Float64}[[1.0, 2.0], [3.0, 4.0]])
    @test length(f2) == 2
    @test isapprox(N*f2, [1.0, 2.0] + [6.0, 8.0])

    @test vec(f2) == [1.0, 2.0, 3.0, 4.0]
    @test isapprox([1.0 2.0]*f, [8.0]'')
end
    
#@test isa(DVTI(), DVTI)
    #@test isa(DCTV(), DCTV)
    #@test isa(DVTV(), DVTV)
    # continous fields
    #@test isa(CCTI(), CCTI)
    #@test isa(CVTI(), CVTI)
    #@test isa(CCTV(), CCTV)
    #@test isa(CVTV(), CVTV)

#=
@testset "updating time dependent fields" begin
    f = Field(0.0 => 1.0)
    @test last(f).time == 0.0
    @test last(f).data == 1.0
    update!(f, 0.0 => 2.0)
    @test last(f).time == 0.0
    @test last(f).data == 2.0
    @test length(f) == 1
    update!(f, 1.0 => 3.0)
    @test last(f).time == 1.0
    @test last(f).data == 3.0
    @test length(f) == 2
end

@testset "updating time invariant fields" begin
    f = Field(1.0)
    @test f.data == 1.0
    update!(f, 2.0)
    @test f.data == 2.0
end

@testset "field defined using function" begin
    g(xi, t) = xi[1]*t
    f = Field(g)
    v = f([1.0], 2.0)
    @test isapprox(v, 2.0)
end

@testset "dictionary fields" begin
    f1 = Dict{Int64, Vector{Float64}}(1 => [0.0, 0.0], 2 => [0.0, 0.0])
    f2 = Dict{Int64, Vector{Float64}}(1 => [1.0, 1.0], 2 => [1.0, 1.0])
    f = Field(0.0 => f1, 1.0 => f2)
    @test isa(f, DVTV)
    @test isapprox(f(0.0)[1], [0.0, 0.0])
    @test isapprox(f(1.0)[2], [1.0, 1.0])

    f = Field(0.0 => f1)
    update!(f, 1.0 => f2)
    @test isa(f, DVTV)
    @test isapprox(f(0.0)[1], [0.0, 0.0])
    @test isapprox(f(1.0)[2], [1.0, 1.0])

    f = Field(f1)
    @test isapprox(f(0.0)[1], [0.0, 0.0])
    @test isapprox(f[1], [0.0, 0.0])

    f = Field(f1)
    @test isa(f, DVTI)
end
=#

