# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test

using JuliaFEM.Core: Quad4, update!

@testset "add new discrete constant time-variant field and interpolate it" begin
    element = Quad4([1, 2, 3, 4])
    element["my field"] = (0.0 => 0.0, 1.0 => 1.0)
    @test isapprox(element("my field", [0.0, 0.0], 0.5), 0.5)
    update!(element, "my field 2", 0.0 => 0.0, 1.0 => 1.0)
    @test isapprox(element("my field 2", [0.0, 0.0], 0.5), 0.5)
end
