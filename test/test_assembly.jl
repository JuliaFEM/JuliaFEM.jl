# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test static condensation" begin
    nodes = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    el1 = Quad4([1, 2, 3, 4])
    el1["geometry"] = Vector[nodes[1], nodes[2], nodes[3], nodes[4]]
    el1["temperature thermal conductivity"] = 6.0
    el1["temperature load"] = [12.0, 12.0, 12.0, 12.0]
    el2 = Seg2([1, 2])
    el2["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0]]
    el2["temperature flux"] = 6.0
    field_problem = HeatProblem()
    push!(field_problem, el1)
    push!(field_problem, el1)

    el3 = Seg2([3, 4])
    el3["geometry"] = Vector[nodes[3], nodes[4]]
    el3["temperature"] = 0.0
    boundary_problem = DirichletProblem("temperature", 1)
    push!(boundary_problem, el3)

    fass = assemble(field_problem, 0.0)
    bass = assemble(boundary_problem, 0.0)

#   interior_dofs = [1, 2]
    boundary_dofs = [3, 4]
    cass = condensate(fass, boundary_dofs)
    @test isapprox(full(cass.Kc), [
        0.0 0.0  0.0  0.0
        0.0 0.0  0.0  0.0
        0.0 0.0  4.8 -4.8
        0.0 0.0 -4.8  4.8])
   @test isapprox(full(cass.fc)', [0.0 0.0 12.0 12.0])
   @test cass.interior_dofs == [1, 2]

   x = sparse(zeros(4))'
   la = sparse(zeros(4))'
   la[3] = la[4] = 24.0
   reconstruct!(cass, x)
   x = full(x)
   info(la)
   info(x)
   @test isapprox(x[1], 1.0)
   @test isapprox(x[2], 1.0)
end

