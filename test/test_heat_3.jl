# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

@testset "2d poisson problem with known analytical solution" begin
    # from FENiCS tutorial, u(x,y) = 1 + x² + 2y² on [0x1]×[0,1]
    # and u₀(x,y) = 1 + x² + 2y², f(x,y) = -6

    mesh_file = Pkg.dir("JuliaFEM")*"/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "UNITSQUARE_6X4")

    field = Problem(Heat, "unit square, 6x4 triangular mesh", 1)
    field.elements = create_elements(mesh, "UNITSQUARE")
    field.properties.formulation = "2D"
    update!(field, "temperature thermal conductivity", 1.0)
    update!(field, "temperature load", -6.0)

    bc = Problem(Dirichlet, "u₀(x,y) = 1 + x² + 2y²", 1, "temperature")
    #bc.properties.order = 2
    #bc.properties.dual_basis = true
    bc.properties.variational = false
    bc.elements = create_elements(mesh, "FACE1", "FACE2", "FACE3", "FACE4")
    function u0(element, ip, time)
        x, y = element("geometry", ip, time)
        return 1 + x^2 + 2*y^2
    end
    update!(bc, "temperature 1", u0)

    solver = LinearSolver(field, bc)
    solver()

    T_fem = Float64[]
    T_acc = Float64[]
    for (nid, X) in field("geometry")
#       info("$nid -> $X")
        push!(T_fem, field("temperature", X)[1])
        push!(T_acc, 1.0 + X[1]^2 + 2*X[2]^2)
    end

    for element in bc.elements
        for (X, T_fem) in zip(element("geometry", 0.0), element("temperature", 0.0))
            x, y = X
            T_acc = 1.0 + x^2 + 2*y^2
#           info("(x,y) = ($x,$y), T_acc = $T_acc, T_fem = $T_fem")
        end
    end

    @test maximum(abs(T_fem-T_acc)) < 1.0e-12
end
