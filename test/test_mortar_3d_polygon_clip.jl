# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM: get_polygon_clip, calculate_polygon_area
using JuliaFEM.Testing

@testset "polygon clipping" begin
    
    S = Vector[
        [0.375, 0.0, 0.5],
        [0.6, 0.0, 0.5],
        [0.5, 0.25, 0.5]]
    M = Vector[
        [0.50, 0.0, 0.5],
        [0.25, 0.0, 0.5],
        [0.375, 0.25, 0.5]]
    n0 = [0.0, 0.0, 1.0]
    P = get_polygon_clip(S, M, n0)
    @test length(P) == 3
    @test isapprox(calculate_polygon_area(P), 1/128)
    
    S = Vector[
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.50, 0.25, 0.5]]
    M = Vector[
        [0.50, 0.0, 0.5],
        [0.25, 0.0, 0.5],
        [0.375, 0.25, 0.5]]
    n0 = [0.0, 0.0, 1.0]
    P = get_polygon_clip(S, M, n0)
    @test length(P) == 3
    @test isapprox(calculate_polygon_area(P), 1/48)

end
