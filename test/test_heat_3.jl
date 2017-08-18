# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "2d poisson problem with known analytical solution" begin
    # from FENiCS tutorial, u(x,y) = 1 + x² + 2y² on [0x1]×[0,1]
    # and u₀(x,y) = 1 + x² + 2y², f(x,y) = -6

    mesh_file = @__DIR__()*"/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "UNITSQUARE_6X4")

    field = Problem(Heat, "unit square, 6x4 triangular mesh", 1)
    field.elements = create_elements(mesh, "UNITSQUARE")
    field.properties.formulation = "2D"
    update!(field, "thermal conductivity", 1.0)
    update!(field, "heat source", -6.0)

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
        push!(T_fem, field("temperature", X)[1])
        push!(T_acc, 1.0 + X[1]^2 + 2*X[2]^2)
    end

    @test maximum(abs.(T_fem-T_acc)) < 1.0e-12

    # gradient of field is
    gradT(X) = [2*X[1] 4*X[2]]
    X = [0.5, 0.5]
    gradT1 = gradT(X)
    gradT2 = field("temperature", X, solver.time, Val{:Grad})
    info("gradT1 = $gradT1, gradT2 = $gradT2")
    # [1.1666666666666625 1.5000000000000018] quite big difference ..?
    @test isapprox(gradT1, gradT2; rtol=25.0e-2)
end
