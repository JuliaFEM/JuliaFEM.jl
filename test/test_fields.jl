# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test updating time dependent fields" begin
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

@testset "test updating time invariant fields" begin
    f = Field(1.0)
    @test f.data == 1.0
    update!(f, 2.0)
    @test f.data == 2.0
end
