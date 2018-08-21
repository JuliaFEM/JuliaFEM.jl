# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

function get_test_model()
    X = Dict(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 0.5],
        4 => [0.0, 0.5],
        5 => [0.0, 0.6],
        6 => [1.0, 0.6],
        7 => [1.0, 1.1],
        8 => [0.0, 1.1])
    el1 = Element(Quad4, (1, 2, 3, 4))
    el2 = Element(Quad4, (5, 6, 7, 8))
    el3 = Element(Seg2, (1, 2))
    el4 = Element(Seg2, (7, 8))
    el5 = Element(Seg2, (4, 3))
    el6 = Element(Seg2, (5, 6))
    update!((el1, el2, el3, el4, el5, el6), "geometry", X)
    update!((el1, el2), "youngs modulus", 96.0)
    update!((el1, el2), "poissons ratio", 1/3)
    update!(el3, "displacement 1", 0.0)
    update!(el3, "displacement 2", 0.0)
    update!(el4, "displacement 1", 0.0)
    update!(el4, "displacement 2", 0.0)
    update!(el6, "master elements", [el5])
    p1 = Problem(Elasticity, "body1", 2)
    p2 = Problem(Elasticity, "body2", 2)
    p3 = Problem(Dirichlet, "fixed", 2, "displacement")
    p4 = Problem(Mortar2D, "interface", 2, "displacement")
    add_elements!(p1, el1)
    add_elements!(p2, el2)
    add_elements!(p3, el3, el4)
    add_elements!(p4, el5, el6)
    return p1, p2, p3, p4
end

## test adjust setting in 2d tie contact
p1, p2, p3, p4 = get_test_model()
p1.properties.formulation = :plane_stress
p2.properties.formulation = :plane_stress
#p4.properties.adjust = true
#p4.properties.rotate_normals = false
analysis = Analysis(Linear)
add_problems!(analysis, p1, p2, p3, p4)
run!(analysis)
el5 = first(get_elements(p4))
xi, time = (0.0, ), 0.0
u = el5("displacement", xi, time)
@debug("u = $u")
@test_broken isapprox(u, [0.0, 0.05])


## test that interface transfers constant field without error
meshfile = @__DIR__() * "/testdata/block_2d.med"
mesh = aster_read_mesh(meshfile)

upper = Problem(PlaneHeat, "upper", 1)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "thermal conductivity", 1.0)
add_elements!(upper, upper_elements)

lower = Problem(PlaneHeat, "lower", 1)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "thermal conductivity", 1.0)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "upper boundary", 1, "temperature")
bc_upper_elements = create_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "temperature 1", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "lower boundary", 1, "temperature")
bc_lower_elements = create_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "temperature 1", 1.0)
add_elements!(bc_lower, bc_lower_elements)

interface = Problem(Mortar2D, "interface between upper and lower block", 1, "temperature")
interface_slave_elements = create_elements(mesh, "LOWER_TOP")
interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
add_master_elements!(interface, interface_master_elements)
add_slave_elements!(interface, interface_slave_elements)

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface)
run!(analysis)

#interface_norm = norm(interface.assembly)
# for bi-orthogonal:
#interface_norm_expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.44870723441585775, 0.44870723441585775, 0.0, 0.0, 0.0]
#interface_norm_expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.39361633468943247, 0.39361633468943247, 0.0, 0.0, 0.0]
#@info("Interface norm:          $interface_norm")
#@info("Interface norm expected: $interface_norm_expected")
#@test isapprox(interface_norm, interface_norm_expected)

xi, time = (0.0, ), 0.0
T_upper = first(bc_upper_elements)("temperature", xi, time)
T_lower = first(bc_lower_elements)("temperature", xi, time)
T_middle = first(interface_slave_elements)("temperature", xi, time)
@debug("T upper: $T_upper, T lower: $T_lower, T interface: $T_middle")

node_ids, temperature = get_nodal_vector(interface_slave_elements, "temperature", 0.0)
T = [t[1] for t in temperature]
minT = minimum(T)
maxT = maximum(T)
@debug("minT = $minT, maxT = $maxT")
@test isapprox(minT, 0.5)
@test isapprox(maxT, 0.5)


## test mesh tie with splitted block and plane stress elasticity

meshfile = @__DIR__() * "/testdata/block_2d.med"
mesh = aster_read_mesh(meshfile)

upper = Problem(Elasticity, "upper", 2)
upper.properties.formulation = :plane_stress
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "youngs modulus", 100.0)
update!(upper_elements, "poissons ratio", 1/3)
add_elements!(upper, upper_elements)

lower = Problem(Elasticity, "lower", 2)
lower.properties.formulation = :plane_stress
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "youngs modulus", 100.0)
update!(lower_elements, "poissons ratio", 1/3)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
bc_upper_elements = create_elements(mesh, "UPPER_TOP")
#   update!(bc_upper.elements, "displacement 1", 0.1)
update!(bc_upper_elements, "displacement 2", -0.1)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
bc_lower_elements = create_elements(mesh, "LOWER_BOTTOM")
#   update!(bc_lower.elements, "displacement 1", 0.0)
update!(bc_lower_elements, "displacement 2", 0.0)
add_elements!(bc_lower, bc_lower_elements)

bc_corner = Problem(Dirichlet, "fix model from lower left corner to prevent singularity", 2, "displacement")
node_ids = find_nearest_nodes(mesh, [0.0, 0.0])
bc_corner_elements = [Element(Poi1, node_ids)]
update!(bc_corner_elements, "geometry", mesh.nodes)
update!(bc_corner_elements, "displacement 1", 0.0)
add_elements!(bc_corner, bc_corner_elements)

interface = Problem(Mortar2D, "interface between upper and lower block", 2, "displacement")
interface_slave_elements = create_elements(mesh, "LOWER_TOP")
interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
add_master_elements!(interface, interface_master_elements)
add_slave_elements!(interface, interface_slave_elements)

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface, bc_corner)
run!(analysis)

slave_elements = get_slave_elements(interface)
node_ids, la = get_nodal_vector(slave_elements, "lambda", 0.0)
for lai in la
    @test isapprox(lai, [0.0, 10.0])
end
