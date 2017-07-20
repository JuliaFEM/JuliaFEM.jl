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
    @test DCTI(1) == 1
    @test length(DCTI(1)) == 1
    @test f == DCTI(1.0)
    @test isapprox(f, DCTI(1.0))
    @test isapprox(f, 1.0)
    @test 2*f == 2.0 # multiply by constant
    @test f(1.0) == 1.0 # time interpolation
    @test isapprox(reshape([2.0],1,1)*f, 2.0) # wanted behavior?
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
    f2 = DVTI(Vector[[1.0, 2.0], [3.0, 4.0]])
    @test isapprox(f2[1], [1.0, 2.0])
    @test isapprox(f2[2], [3.0, 4.0])
    @test length(f2) == 2
    @test isapprox(N*f2, [1.0, 2.0] + [6.0, 8.0])

    # iteration of DVTI field
    s = zeros(2)
    for j in f2
        s += j
    end
    @test isapprox(s, [4.0, 6.0])

    @test vec(f2) == [1.0, 2.0, 3.0, 4.0]
    @test isapprox([1.0 2.0]*f, [8.0]'')

    new_data = [2.0, 3.0, 4.0, 5.0]
    f4 = similar(f2, new_data)
    @test isa(f4, DVTI)
    @test isapprox(f4.data[1], [2.0, 3.0])
    @test isapprox(f4.data[2], [4.0, 5.0])
end

@testset "discrete, constant, time-variant field" begin
    @test isa(DCTV(), DCTV)
    f = Field(0.0 => 1.0)
    @test isa(f, DCTV)
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

    @testset "interpolation in time direction" begin
        @test isa(f(0.0), DCTI) # converts to time-invariant after time interpolation
        @test isapprox(f(-1.0), 2.0)
        @test isapprox(f(0.0), 2.0)
        @test isapprox(f(0.5), 2.5)
        @test isapprox(f(1.0), 3.0)
        @test isapprox(f(2.0), 3.0)
    end

    # create several time steps at once
    f = DCTV(0.0 => 1.0, 1.0 => 2.0)
    @test isapprox(f(0.5), 1.5)

end

@testset "discrete, variable, time-variant field" begin
    @test isa(DVTV(), DVTV)
    f = Field(0.0 => [1.0, 2.0])
    @test isa(f, DVTV)
    @test last(f).time == 0.0
    @test last(f).data == [1.0, 2.0]
    update!(f, 0.0 => [2.0, 3.0])
    @test last(f).time == 0.0
    @test last(f).data == [2.0, 3.0]
    @test length(f) == 1
    update!(f, 1.0 => [3.0, 4.0])
    @test last(f).time == 1.0
    @test last(f).data == [3.0, 4.0]
    @test length(f) == 2
    
    @testset "interpolation in time direction" begin
        @test isa(f(0.0), DVTI) # converts to time-invariant after time interpolation
        @test isapprox(f(-1.0), [2.0, 3.0])
        @test isapprox(f(0.0), [2.0, 3.0])
        @test isapprox(f(0.5), [2.5, 3.5])
        @test isapprox(f(1.0), [3.0, 4.0])
        @test isapprox(f(2.0), [3.0, 4.0])
    end

    # create several time steps at once
    f = DVTV(0.0 => [1.0, 2.0], 1.0 => [2.0, 3.0])
    @test isapprox(f(0.5), [1.5, 2.5])
end

@testset "continuous, constant, time-invariant field" begin
    f = Field(() -> 2.0)
    @test isapprox(f([1.0], 2.0), 2.0)

end

@testset "continuous, constant, time variant field" begin
    f = Field((time::Float64) -> 2.0*time)
    @test isapprox(f([1.0], 2.0), 4.0)

end

@testset "continuous, variable, time invariant field" begin
    f = Field((xi::Vector) -> sum(xi))
    @test isapprox(f([1.0, 2.0], 2.0), 3.0)
end

@testset "continuous, variable, time variant field" begin
    f = Field((xi::Vector, t::Float64) -> xi[1]*t)
    @test isapprox(f([1.0], 2.0), 2.0)
end

@testset "unknown function argument for continuous field" begin
    @test_throws ErrorException Field((a, b, c) -> a*b*c)
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
