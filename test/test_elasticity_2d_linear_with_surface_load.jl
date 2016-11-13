# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

#=
- solve 2d plane stress problem with known solution:
    surface traction force in 2d
    volume load in 2d
    reaction force
- test postprocessing of nodal fields:
    geometry
    displacement
    reaction force
    concentrated force
=#
@testset "test 2d linear elasticity with surface + volume load" begin
    meshfile = "/geometry/2d_block/BLOCK_1elem.med"
    mesh = aster_read_mesh(Pkg.dir("JuliaFEM")*meshfile)

    # field problem
    block = Problem(Elasticity, "BLOCK", 2)
    block.properties.formulation = :plane_stress
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false
    block.elements = create_elements(mesh, "BLOCK")
    update!(block.elements, "youngs modulus", 288.0)
    update!(block.elements, "poissons ratio", 1/3)
    update!(block.elements, "displacement load 2", 576.0)

    # traction
    traction = Problem(Elasticity, "TRACTION", 2)
    traction.properties.formulation = :plane_stress
    traction.properties.finite_strain = false
    traction.properties.geometric_stiffness = false
    traction.elements = create_elements(mesh, "TOP")
    update!(traction, "displacement traction force 2", 288.0)

    # boundary conditions
    bc_sym_23 = Problem(Dirichlet, "symmetry bc 23", 2, "displacement")
    bc_sym_23.elements = create_elements(mesh, "LEFT")
    update!(bc_sym_23, "displacement 1", 0.0)
    bc_sym_13 = Problem(Dirichlet, "symmetry bc 13", 2, "displacement")
    bc_sym_13.elements = create_elements(mesh, "BOTTOM")
    update!(bc_sym_13, "displacement 2", 0.0)

    solver = Solver(Linear, block, traction, bc_sym_23, bc_sym_13)
    solver()

    info("u = ", block.assembly.u)
    info("λ = ", block.assembly.la)

    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
#=
    # fetch nodal results X + u and join them into one table using DataFrames
    X = solver(DataFrame, "geometry", :COOR)
    u = solver(DataFrame, "displacement", :U)
    la = solver(DataFrame, "reaction force", :RF)
    f = solver(DataFrame, "concentrated force", :CF)
    results = join(X, u, on=:NODE, kind=:outer)
    results = join(results, la, on=:NODE, kind=:outer)
    length(f) != 0 && (results = join(results, f, on=:NODE, kind=:outer))
    sort!(results, cols=[:NODE])
    println(results)

    u3 = extract(results, NODE=:N3, :U1, :U2)
    @test isapprox(u3, u3_expected)

    # element details
    el = first(block.elements)
    X = el("geometry", 0.0)
    debug("X = $X")
    S1 = block(el, [0.0, 0.0], 0.0, Val{:S})
    S1 = S1[[1,4,2]]
    E1 = block(el, [0.0, 0.0], 0.0, Val{:E})
    E1 = E1[[1,4,2]]
    C1 = block(el, [0.0, 0.0], 0.0, Val{:COORD})
    info("strain = $E1, stress = $S1, at $C1")
    @test isapprox(E1, [-2/3, 2.0, 0.0])
    @test isapprox(S1, [0.0, 576.0, 0.0])
    @test isapprox(C1, [0.5, 0.5])

    S1 = block(DataFrame, 0.0, Val{:S})
    E1 = block(DataFrame, 0.0, Val{:E})
    C1 = block(DataFrame, 0.0, Val{:COORD})
    println(S1)
    println(E1)
    println(C1)
    S = solver(DataFrame, 0.0, Val{:S})
    println(S)

    info(solver("displacement", 0.0))
    solver()
    info(solver("displacement", 0.0))
    u = solver("displacement", 0.0)[3]
    info("u3 = $u")
    @test isapprox(u, u3_expected)

#   info("calling nonlinear solver")
#   solver2 = NonlinearSolver(block, traction, bc_sym_23, bc_sym_13)
#   solver2()
#   u = solver2("displacement", 0.0)[3]
#   info("nlsolver u3 = $u, expected = $u3_expected")
#   @test isapprox(u, u3_expected; rtol=1.0e-5)
=#
end

#= TODO: to other file
@testset "test dump model to disk and read back before and after solution" begin
    solver = get_model("test 2d linear elasticity with surface + volume load")
    save("/tmp/model.jld", "linear_model", solver)
    solver2 = load("/tmp/model.jld")["linear_model"]
    solver2()
    save("/tmp/model.jld", "results", solver2)
    solver3 = load("/tmp/model.jld")["results"]
    block = solver3["BLOCK"]
    u3 = reshape(block.assembly.u, 2, 4)[:,3]
    f = 288.0
    g = 576.0
    E = 288.0
    nu = 1/3
    u3_expected = f/E*[-nu, 1] + g/(2*E)*[-nu, 1]
    @test isapprox(u3, u3_expected)
end
=#
