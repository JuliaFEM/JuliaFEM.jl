# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "3d rod" begin
    mesh = aster_read_mesh(Pkg.dir("JuliaFEM")*"/test/testdata/primitives.med", "CYLINDER_20_TET4")
    problem = Problem(Heat, "rod of length 20", 1)
    problem.elements = create_elements(mesh, "CYLINDER")
    update!(problem, "temperature thermal conductivity", 200.0)
    outer = Problem(Heat, "outer surface", 1)
    outer.elements = create_elements(mesh, "FACE2", "FACE3")
    update!(outer, "temperature external temperature", 20.0)
    update!(outer, "temperature heat transfer coefficient", 1.0)
    #midline = Problem(Heat, "midline of rod", 1)
    #midline.elements = create_elements(mesh, "INNER_LINE")
    boundary = Problem(Dirichlet, "homogeneous dirichlet boundary", 1, "temperature")
    boundary.elements = create_elements(mesh, "FACE1")
    update!(boundary, "temperature 1", 100.0)
    #solver = LinearSolver(problem, outer, boundary, midline)
    solver = LinearSolver(problem, outer, boundary)
    solver()

    L = 20
    k = 200.0
    Tu = 20.0
    h = 1.0
    P = 2*pi
    A = pi
    α = h
    β = sqrt((h*P)/(k*A))
    T0 = 100.0
    C = [1.0 1.0; (α+k*β)*exp(β*L) (α-k*β)*exp(-β*L)] \ [T0-Tu, 0]
    T(x) = dot(C, [exp(β*x), exp(-β*x)]) + Tu

    T_diff = []
    for x in linspace(0, 20)
        T_FEM = problem("temperature", [x, 0.0, 0.0])[1]
        T_ACC = T(x)
        push!(T_diff, norm(T_FEM - T_ACC))
        info("x = $x, T_FEM = $T_FEM, T_ACC = $T_ACC")
    end
    info("mean diff = ", mean(T_diff))
    # mean diff = 1.14
    @test mean(T_diff) < 1.2
end

