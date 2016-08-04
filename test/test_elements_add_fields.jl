# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "dict field" begin
    el = Element(Seg2, 1, [1, 2])
    X = Dict{Int64, Vector{Float64}}(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [0.5, 0.5])
    f = Field(X)
    debug("field = $f")
    #update!(el, "geometry", X)
    el["geometry"] = f
    @test isapprox(el("geometry")[1], [0.0, 0.0])
    @test isapprox(el("geometry", 0.0)[1], [0.0, 0.0])
    @test isapprox(el("geometry", 0.0)[3], [0.5, 0.5])
    @test isapprox(el("geometry", [0.0], 0.0), [0.5, 0.0])
end
