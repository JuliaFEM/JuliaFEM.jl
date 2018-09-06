# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# test continuum 3d linear elasticity with surface load

X = Dict(1 => [0.0, 0.0, 0.0],
         2 => [1.0, 0.0, 0.0],
         3 => [1.0, 1.0, 0.0],
         4 => [0.0, 1.0, 0.0],
         5 => [0.0, 0.0, 1.0],
         6 => [1.0, 0.0, 1.0],
         7 => [1.0, 1.0, 1.0],
         8 => [0.0, 1.0, 1.0])

element1 = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
element2 = Element(Quad4, (5, 6, 7, 8))
update!((element1, element2), "geometry", X)
update!(element1, "youngs modulus", 288.0)
update!(element1, "poissons ratio", 1/3)
update!(element2, "displacement traction force 3", 288.0)
update!(element1, "displacement load 3", 576.0)

problem = Problem(Elasticity, "solve continuum block", 3)
problem.properties.finite_strain = false
add_elements!(problem, element1, element2)

symxy = Element(Quad4, (1, 2, 3, 4))
symxz = Element(Quad4, (1, 2, 6, 5))
symyz = Element(Quad4, (1, 4, 8, 5))
update!([symxy, symxz, symyz], "geometry", X)
update!(symyz, "displacement 1", 0.0)
update!(symxz, "displacement 2", 0.0)
update!(symxy, "displacement 3", 0.0)

bc = Problem(Dirichlet, "symmetry boundary conditions", 3, "displacement")
add_elements!(bc, symxy, symxz, symyz)

analysis = Analysis(Linear)
add_problems!(analysis, problem, bc)
run!(analysis)

u = element1("displacement", [1.0, 1.0, 1.0], 0.0)
u_expected = 2.0 * [-1/3, -1/3, 1.0]
@debug("displacement at tip", u, u_expected)
@test isapprox(u, u_expected)

# run similar analysis for different meshes

function solve_rod_model_elasticity(eltype)
    fn = @__DIR__() * "/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, eltype)
    element_sets = join(keys(mesh.element_sets), ", ")
    @debug("element sets", element_sets)
    p1 = Problem(Elasticity, "rod", 3)
    p2 = Problem(Elasticity, "trac", 3)
    p3 = Problem(Dirichlet, "fixed", 3, "displacement")
    p4 = Problem(Dirichlet, "fixed", 3, "displacement")
    p5 = Problem(Dirichlet, "fixed", 3, "displacement")
    p1_elements = create_elements(mesh, "ROD")
    p2_elements = create_elements(mesh, "FACE2")
    p3_elements = create_elements(mesh, "FACE1")
    p4_elements = create_elements(mesh, "FACE3")
    p5_elements = create_elements(mesh, "FACE5")
    update!(p1_elements, "youngs modulus", 96.0)
    update!(p1_elements, "poissons ratio", 1/3)
    update!(p2_elements, "displacement traction force 1", 96.0)
    update!(p3_elements, "displacement 1", 0.0)
    update!(p4_elements, "displacement 2", 0.0)
    update!(p5_elements, "displacement 3", 0.0)
    add_elements!(p1, p1_elements)
    add_elements!(p2, p2_elements)
    add_elements!(p3, p3_elements)
    add_elements!(p4, p4_elements)
    add_elements!(p5, p5_elements)
    analysis = Analysis(Linear)
    add_problems!(analysis, p1, p2, p3, p4, p5)
    run!(analysis)
    u = p1("displacement", 0.0)
    u_max = maximum(maximum(ui for ui in values(u)))
    @debug("analysis of block", eltype, u_max)
    return u_max
end

@test isapprox(solve_rod_model_elasticity("Tet4"), 0.2)
@test isapprox(solve_rod_model_elasticity("Tet10"), 0.2)
@test isapprox(solve_rod_model_elasticity("Hex8"), 0.2)
@test isapprox(solve_rod_model_elasticity("Hex20"), 0.2)
@test isapprox(solve_rod_model_elasticity("Hex27"), 0.2)
