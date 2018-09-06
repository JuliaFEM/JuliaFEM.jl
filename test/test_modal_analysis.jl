# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

X = Dict(
    1 => [2.0, 3.0, 4.0],
    2 => [6.0, 3.0, 2.0],
    3 => [2.0, 5.0, 1.0],
    4 => [4.0, 3.0, 6.0])
u = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [0.0, 0.0, 0.0],
    3 => [0.0, 0.0, 0.0],
    4 => [0.25, 0.25, 0.25])
element1 = Element(Tet4, (1, 2, 3, 4))
element2 = Element(Tri3, (1, 2, 3))
update!((element1, element2), "geometry", X)
update!((element1, element2), "displacement", 0.0 => u)
update!(element1, "youngs modulus", 96.0)
update!(element1, "poissons ratio", 1.0/3.0)
update!(element1, "density", 420.0)
update!(element2, "displacement 1", 0.0)
update!(element2, "displacement 2", 0.0)
update!(element2, "displacement 3", 0.0)
problem1 = Problem(Elasticity, "test problem", 3)
problem1.properties.finite_strain = false
problem1.properties.geometric_stiffness = false
problem2 = Problem(Dirichlet, "boundary condition", 3, "displacement")
add_elements!(problem1, element1)
add_elements!(problem2, element2)
analysis = Analysis(Modal)
analysis.properties.which = :LM
add_problems!(analysis, problem1, problem2)

# test eigenvalues for single tet4 element
run!(analysis)
@test isapprox(analysis.properties.eigvals, [4/3, 1/3])

# test eigenvalues for single tet4 element, with geometric stiffness
problem1.properties.geometric_stiffness = true
analysis.properties.geometric_stiffness = true
run!(analysis)
@test isapprox(analysis.properties.eigvals, [5/3, 2/3])

# test poisson problem modal analysis without tie
X = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 3.0],
    4 => [0.0, 3.0],
    5 => [0.0, 3.0],
    6 => [1.0, 3.0],
    7 => [1.0, 9.0],
    8 => [0.0, 9.0])
element1 = Element(Quad4, (1, 2, 3, 4))
element2 = Element(Quad4, (4, 3, 7, 8))
element3 = Element(Seg2, (1, 2))
element4 = Element(Seg2, (7, 8))
update!((element1, element2, element3, element4), "geometry", X)
update!((element1, element2), "density", 6.0)
update!((element1, element2), "thermal conductivity", 36.0)
update!((element3, element4), "temperature 1", 0.0)
problem1 = Problem(PlaneHeat, "combined body", 1)
problem2 = Problem(Dirichlet, "fixed ends", 1, "temperature")
add_elements!(problem1, element1, element2)
add_elements!(problem2, element3, element4)
analysis = Analysis(Modal)
add_problems!(analysis, problem1, problem2)
run!(analysis)
@test isapprox(first(analysis.properties.eigvals), 1.0)

#=
@testset "test poisson modal problem with mesh tie" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 3.0],
        4 => [0.0, 3.0],
        5 => [0.0, 3.0],
        6 => [1.0, 3.0],
        7 => [1.0, 9.0],
        8 => [0.0, 9.0])
    el1 = Element(Quad4, [1, 2, 3, 4])
    el2 = Element(Quad4, [5, 6, 7, 8])
    el3 = Element(Seg2, [1, 2])
    el4 = Element(Seg2, [7, 8])
    el5 = Element(Seg2, [3, 4])
    el6 = Element(Seg2, [5, 6])
    update!([el1, el2, el3, el4, el5, el6], "geometry", X)
    update!([el1, el2], "density", 6.0)
    update!([el1, el2], "thermal conductivity", 36.0)
    update!([el3, el4], "temperature 1", 0.0)
    update!(el5, "master elements", [el6])
    p1 = Problem(PlaneHeat, "body 1", 1)
    add_elements!(p1, [el1])
    p2 = Problem(PlaneHeat, "body 2", 1)
    add_elements!(p2, [el2])
    p3 = Problem(Dirichlet, "fixed ends", 1, "temperature")
    add_elements!(p3, [el3, el4])
    p4 = Problem(Mortar2D, "interface between bodies", 1, "temperature")
    add_slave_elements!(p4, [el5])
    add_master_elements!(p4, [el6])

    solver = Solver(Modal)
    push!(solver, p1, p2, p3, p4)
    solver()
    @test isapprox(solver.properties.eigvals[1], 1.0)
end
=#
