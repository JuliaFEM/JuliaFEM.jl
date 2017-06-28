# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "get nodal field from boundary condition if geometry is not defined" begin
    bc = Problem(Dirichlet, "bc without geometry", 3, "displacement")
    bc.elements = [Element(Poi1, [1])]
    @test bc("geometry", 0.0) == nothing
    s = Solver(Linear, "test solver")
    s.problems = [bc]
    @test length(s("geometry", 0.0)) == 0
end
