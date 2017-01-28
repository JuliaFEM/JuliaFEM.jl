# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "NSeg interpolate" begin
    element = Element(NSeg, [1, 2])
    @test element([0.0], 0.0) == [0.5 0.5]
end

@testset "NSurf interpolate" begin
    element = Element(NSurf, [1, 2, 3, 4])
    @test element([0.0, 0.0], 0.0) == [0.25 0.25 0.25 0.25]
end

@testset "NSolid interpolate" begin
    element = Element(NSolid, [1, 2, 3, 4, 5, 6, 7, 8])
    @test element([0.0, 0.0, 0.0], 0.0) == [0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125]
end
