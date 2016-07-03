# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test interpolation of discrete constant time-variant field" begin
    f = DCTV(0.0 => 0.0, 1.0 => 1.0)
    # time interpolation of time-variant fields results it's
    # time-invariant counterpart
    @test isapprox(f(-1.0), DCTI(0.0))
    @test isapprox(f( 0.0), DCTI(0.0))
    @test isapprox(f( 0.3), DCTI(0.3))
    @test isapprox(f( 0.5), DCTI(0.5))
    @test isapprox(f( 0.9), DCTI(0.9))
    @test isapprox(f( 1.0), DCTI(1.0))
    @test isapprox(f( 1.5), DCTI(1.0))

    f2 = DCTV(0.0 => 0.0, 0.25 => -0.1, 0.50 => -0.1)
    @test isapprox(f2(0.0), DCTI(0.0))
    @test isapprox(f2(0.25), DCTI(-0.1))
    @test isapprox(f2(0.50), DCTI(-0.1))
    @test isapprox(f2(0.35), DCTI(-0.1))
end

