# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "write simple Xdmf file" begin

    X = Dict(
             1 => [0.0, 0.0, 0.0],
             2 => [1.0, 0.0, 0.0],
             3 => [1.0, 1.0, 0.0],
             4 => [0.0, 1.0, 0.0],
             5 => [0.0, 0.0, 0.5],
             6 => [1.0, 0.0, 0.5],
             7 => [1.0, 1.0, 0.5],
             8 => [0.0, 1.0, 0.5],
             9 => [0.0, 0.0, 1.0],
             10 => [1.0, 0.0, 1.0],
             11 => [1.0, 1.0, 1.0],
             12 => [0.0, 1.0, 1.0])

    u = Dict()
    u[0] = Dict(
                1 => [0.0, 0.0, 0.0],
                2 => [0.0, 0.0, 0.0],
                3 => [0.0, 0.0, 0.0],
                4 => [0.0, 0.0, 0.0],
                5 => [0.0, 0.0, 0.0],
                6 => [0.0, 0.0, 0.0],
                7 => [0.0, 0.0, 0.0],
                8 => [0.0, 0.0, 0.0],
                9 => [0.0, 0.0, 0.0],
                10 => [0.0, 0.0, 0.0],
                11 => [0.0, 0.0, 0.0],
                12 => [0.0, 0.0, 0.0])
    u[1] = Dict(
                1 => [0.0, 0.0, 0.0],
                2 => [0.0, 0.0, 0.0],
                3 => [0.0, 0.0, 0.0],
                4 => [0.0, 0.0, 0.0],
                5 => [0.0, 0.0, -0.1],
                6 => [0.0, 0.0, -0.1],
                7 => [0.0, 0.0, -0.1],
                8 => [0.0, 0.0, -0.1],
                9 => [0.0, 0.0, -0.2],
                10 => [0.0, 0.0, -0.2],
                11 => [0.0, 0.0, -0.2],
                12 => [0.0, 0.0, -0.2])

    T = Dict(
             1 => 10.0,
             2 => 10.0,
             3 => 10.0,
             4 => 10.0,
             5 => 20.0,
             6 => 20.0,
             7 => 20.0,
             8 => 20.0,
             9 => 30.0,
             10 => 30.0,
             11 => 30.0,
             12 => 30.0)

    rf = Dict()
    rf[0] = Dict(
                 1 => [0.0, 0.0, 0.0],
                 2 => [0.0, 0.0, 0.0],
                 3 => [0.0, 0.0, 0.0],
                 4 => [0.0, 0.0, 0.0])
    rf[1] = Dict(
                 1 => [0.0, 0.0, 1.0],
                 2 => [0.0, 0.0, 1.0],
                 3 => [0.0, 0.0, 1.0],
                 4 => [0.0, 0.0, 1.0])

    e1 = Element(Hex8, [1, 2, 3, 4, 5, 6, 7, 8])
    e2 = Element(Hex8, [5, 6, 7, 8, 9, 10, 11, 12])
    e3 = Element(Quad4, [1, 2, 3, 4])
    update!([e1, e2, e3], "geometry", X)
    update!([e1, e2, e3], "displacement", 0.0 => u[0])
    update!([e1, e2, e3], "displacement", 1.0 => u[1])
    update!([e1, e2, e3], "temperature", T)
    update!(e3, "reaction force", 0.0 => rf[0])
    update!(e3, "reaction force", 1.0 => rf[1])

    p1 = Problem(Elasticity, "lower", 3)
    p1.elements = [e1]
    p2 = Problem(Elasticity, "upper", 3)
    p2.elements = [e2]
    p3 = Problem(Dirichlet, "bc", 3, "displacement")
    p3.elements = [e3]

    xdmf = Xdmf()
    xdmf.format = "XML"
    update_xdmf!(xdmf, p1, 0.0, ["displacement", "temperature"])
    update_xdmf!(xdmf, p2, 0.0, ["displacement"])
    update_xdmf!(xdmf, p3, 0.0, ["reaction force"])
    update_xdmf!(xdmf, p1, 1.0, ["displacement", "temperature"])
    update_xdmf!(xdmf, p2, 1.0, ["displacement"])
    update_xdmf!(xdmf, p3, 1.0, ["reaction force"])
    @test read(xdmf, "/Domain/Grid/Grid/Time/Value") == "0.0"
    @test read(xdmf, "/Domain/Grid/Grid[2]/Time/Value") == "1.0"
end

