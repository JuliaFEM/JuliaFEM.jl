# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test, Statistics

### temperature patch tests, sl tet4, dl tet4, sl tet10, dl tet 10

tet4_meshfile = joinpath("test_problems_mortar_3d", "tet4.inp")
tet10_meshfile = joinpath("test_problems_mortar_3d", "tet10.inp")

# patch test temperature + abaqus inp + tet4
mesh = abaqus_read_mesh(tet4_meshfile)

upper = Problem(Heat, "UPPER", 1)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "thermal conductivity", 1.0)
add_elements!(upper, upper_elements)

lower = Problem(Heat, "LOWER", 1)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "thermal conductivity", 1.0)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "temperature 1", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "temperature 1", 1.0)
add_elements!(bc_lower, bc_lower_elements)

interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

JuliaFEM.diagnose_interface(interface, 0.0)
analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface)
xdmf = Xdmf("sl_lin_temp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

T = values(interface("temperature", 0.0))
minT = minimum(T)
maxT = maximum(T)
@info("minT = $minT, maxT = $maxT")
@test isapprox(minT, 0.5)
@test isapprox(maxT, 0.5)

#=
initialize!(solver)
assemble!(solver)
M, K, Kg, f, fg = get_field_assembly(solver)
Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
K = K + Kg + Kb
f = f + fg + fb
K = 1/2*(K + K')
M = 1/2*(M + M')
=#


# patch test temperature + abaqus inp + tet4 + dual basis + adjust
mesh = abaqus_read_mesh(tet4_meshfile)

upper = Problem(Heat, "UPPER", 1)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "thermal conductivity", 1.0)
add_elements!(upper, upper_elements)

lower = Problem(Heat, "LOWER", 1)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "thermal conductivity", 1.0)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "temperature 1", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "temperature 1", 1.0)
add_elements!(bc_lower, bc_lower_elements)

interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_master_elements, interface_slave_elements)
interface.properties.dual_basis = true
#interface.properties.adjust = true

JuliaFEM.diagnose_interface(interface, 0.0)
analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface)
xdmf = Xdmf("dl_lin_temp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

T = values(interface("temperature", 0.0))
minT = minimum(T)
maxT = maximum(T)
@debug("minT = $minT, maxT = $maxT")
@test isapprox(minT, 0.5)
@test isapprox(maxT, 0.5)


# patch test temperature + abaqus inp + tet10, quadratic surface elements
mesh = abaqus_read_mesh(tet10_meshfile)

upper = Problem(Heat, "UPPER", 1)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "thermal conductivity", 1.0)
add_elements!(upper, upper_elements)

lower = Problem(Heat, "LOWER", 1)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "thermal conductivity", 1.0)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "temperature 1", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "temperature 1", 1.0)
add_elements!(bc_lower, bc_lower_elements)

interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_master_elements, interface_slave_elements)

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.alpha = 0.0
#   JuliaFEM.diagnose_interface(interface, 0.0)

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface)
xdmf = Xdmf("sl_quad_temp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

T = values(interface("temperature", 0.0))
minT = minimum(T)
maxT = maximum(T)
stdT = std(T)
@info("minT = $minT, maxT = $maxT, stdT = $stdT")
@test maxT - minT < 1.0e-6
@test isapprox(stdT, 0.0; atol=1.0e-6)


# patch test temperature + abaqus inp + tet10 + quadratic surface elements + dual basis + alpha=0.2
mesh = abaqus_read_mesh(tet10_meshfile)

upper = Problem(Heat, "UPPER", 1)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "thermal conductivity", 1.0)
add_elements!(upper, upper_elements)

lower = Problem(Heat, "LOWER", 1)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "thermal conductivity", 1.0)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 1, "temperature")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "temperature 1", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 1, "temperature")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "temperature 1", 1.0)
add_elements!(bc_lower, bc_lower_elements)

interface = Problem(Mortar, "interface between upper and lower block", 1, "temperature")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.dual_basis = true
interface.properties.alpha = 0.2

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, interface)
xdmf = Xdmf("dl_quad_temp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

T = values(interface("temperature", 0.0))
minT = minimum(T)
maxT = maximum(T)
stdT = std(T)
@info("minT = $minT, maxT = $maxT, stdT = $stdT")
@test maxT - minT < 1.0e-10
@test isapprox(stdT, 0.0; atol=1.0e-10)

### displacement patch tests, sl tet4, dl tet4, sl tet10, dl tet 10

# patch test displacement + abaqus inp + tet4 + adjust

mesh = abaqus_read_mesh(tet4_meshfile)
# modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
JuliaFEM.create_node_set_from_element_set!(mesh, :UPPER)
for nid in mesh.node_sets[:UPPER]
    mesh.nodes[nid][3] += 0.2
end

upper = Problem(Elasticity, "UPPER", 3)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "youngs modulus", 288.0)
update!(upper_elements, "poissons ratio", 1/3)
add_elements!(upper, upper_elements)

lower = Problem(Elasticity, "LOWER", 3)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "youngs modulus", 288.0)
update!(lower_elements, "poissons ratio", 1/3)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "displacement 3", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "displacement 3", 0.0)
add_elements!(bc_lower, bc_lower_elements)

bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
bc_sym13_elements_1 = create_surface_elements(mesh, "LOWER_SYM13")
bc_sym13_elements_2 = create_surface_elements(mesh, "UPPER_SYM13")
update!(bc_sym13_elements_1, "displacement 2", 0.0)
update!(bc_sym13_elements_2, "displacement 2", 0.0)
add_elements!(bc_sym13, bc_sym13_elements_1, bc_sym13_elements_2)

bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
bc_sym23_elements_1 = create_surface_elements(mesh, "LOWER_SYM23")
bc_sym23_elements_2 = create_surface_elements(mesh, "UPPER_SYM23")
update!(bc_sym23_elements_1, "displacement 1", 0.0)
update!(bc_sym23_elements_2, "displacement 1", 0.0)
add_elements!(bc_sym23, bc_sym23_elements_1, bc_sym23_elements_2)

interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

append!(interface.assembly.removed_dofs, [1316, 1319, 1358, 1387, 1388, 1492, 1597, 1627])

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.adjust = true
interface.properties.dual_basis = false

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
xdmf = Xdmf("sl_lin_disp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

displacement = interface("displacement", 0.0)
u3 = [u[3] for u in values(displacement)]
maxabsu3 = maximum(abs.(u3))
stdabsu3 = std(abs.(u3))
@info("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
@test isapprox(stdabsu3, 0.0; atol=1.0e-10)


# patch test displacement + abaqus inp + tet4 + adjust + dual basis

mesh = abaqus_read_mesh(tet4_meshfile)
# modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
JuliaFEM.create_node_set_from_element_set!(mesh, :UPPER)
for nid in mesh.node_sets[:UPPER]
    mesh.nodes[nid][3] += 0.2
end

upper = Problem(Elasticity, "UPPER", 3)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "youngs modulus", 288.0)
update!(upper_elements, "poissons ratio", 1/3)
add_elements!(upper, upper_elements)

lower = Problem(Elasticity, "LOWER", 3)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "youngs modulus", 288.0)
update!(lower_elements, "poissons ratio", 1/3)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "displacement 3", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "displacement 3", 0.0)
add_elements!(bc_lower, bc_lower_elements)

bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
bc_sym13_elements_1 = create_surface_elements(mesh, "LOWER_SYM13")
bc_sym13_elements_2 = create_surface_elements(mesh, "UPPER_SYM13")
update!(bc_sym13_elements_1, "displacement 2", 0.0)
update!(bc_sym13_elements_2, "displacement 2", 0.0)
add_elements!(bc_sym13, bc_sym13_elements_1, bc_sym13_elements_2)

bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
bc_sym23_elements_1 = create_surface_elements(mesh, "LOWER_SYM23")
bc_sym23_elements_2 = create_surface_elements(mesh, "UPPER_SYM23")
update!(bc_sym23_elements_1, "displacement 1", 0.0)
update!(bc_sym23_elements_2, "displacement 1", 0.0)
add_elements!(bc_sym23, bc_sym23_elements_1, bc_sym23_elements_2)

interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

append!(interface.assembly.removed_dofs, [1316, 1319, 1358, 1387, 1388, 1492, 1597, 1627])

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.adjust = true
interface.properties.dual_basis = true

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
xdmf = Xdmf("dl_lin_disp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

displacement = interface("displacement", 0.0)
u3 = [u[3] for u in values(displacement)]
maxabsu3 = maximum(abs.(u3))
stdabsu3 = std(abs.(u3))
@debug("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
@test isapprox(stdabsu3, 0.0; atol=1.0e-10)


# patch test displacement + abaqus inp + tet10 + adjust

mesh = abaqus_read_mesh(tet10_meshfile)
# modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
JuliaFEM.create_node_set_from_element_set!(mesh, :UPPER)
for nid in mesh.node_sets[:UPPER]
    mesh.nodes[nid][3] += 0.2
end

upper = Problem(Elasticity, "UPPER", 3)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "youngs modulus", 288.0)
update!(upper_elements, "poissons ratio", 1/3)
add_elements!(upper, upper_elements)

lower = Problem(Elasticity, "LOWER", 3)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "youngs modulus", 288.0)
update!(lower_elements, "poissons ratio", 1/3)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "displacement 3", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "displacement 3", 0.0)
add_elements!(bc_lower, bc_lower_elements)

bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
bc_sym13_elements_1 = create_surface_elements(mesh, "LOWER_SYM13")
bc_sym13_elements_2 = create_surface_elements(mesh, "UPPER_SYM13")
update!(bc_sym13_elements_1, "displacement 2", 0.0)
update!(bc_sym13_elements_2, "displacement 2", 0.0)
add_elements!(bc_sym13, bc_sym13_elements_1, bc_sym13_elements_2)

bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
bc_sym23_elements_1 = create_surface_elements(mesh, "LOWER_SYM23")
bc_sym23_elements_2 = create_surface_elements(mesh, "UPPER_SYM23")
update!(bc_sym23_elements_1, "displacement 1", 0.0)
update!(bc_sym23_elements_2, "displacement 1", 0.0)
add_elements!(bc_sym23, bc_sym23_elements_1, bc_sym23_elements_2)

interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

removed_dofs = [1316, 1319, 1325, 1358, 1361, 1387, 1388, 1391, 1492, 1597, 1600, 1627, 1630, 1657]
append!(interface.assembly.removed_dofs, removed_dofs)

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.adjust = true
interface.properties.dual_basis = false

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
xdmf = Xdmf("sl_quad_disp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

displacement = interface("displacement", 0.0)
u3 = [u[3] for u in values(displacement)]
maxabsu3 = maximum(abs.(u3))
stdabsu3 = std(abs.(u3))
@debug("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
@test isapprox(stdabsu3, 0.0; atol=1.0e-6)


# patch test displacement + abaqus inp + tet10 + adjust + dual basis + alpha=0.2

mesh = abaqus_read_mesh(tet10_meshfile)
# modify mesh a bit, find all nodes in elements in element set UPPER and put 0.2 to X3 to test adjust
JuliaFEM.create_node_set_from_element_set!(mesh, :UPPER)
for nid in mesh.node_sets[:UPPER]
    mesh.nodes[nid][3] += 0.2
end

upper = Problem(Elasticity, "UPPER", 3)
upper_elements = create_elements(mesh, "UPPER")
update!(upper_elements, "youngs modulus", 288.0)
update!(upper_elements, "poissons ratio", 1/3)
add_elements!(upper, upper_elements)

lower = Problem(Elasticity, "LOWER", 3)
lower_elements = create_elements(mesh, "LOWER")
update!(lower_elements, "youngs modulus", 288.0)
update!(lower_elements, "poissons ratio", 1/3)
add_elements!(lower, lower_elements)

bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
bc_upper_elements = create_surface_elements(mesh, "UPPER_TOP")
update!(bc_upper_elements, "displacement 3", 0.0)
add_elements!(bc_upper, bc_upper_elements)

bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
bc_lower_elements = create_surface_elements(mesh, "LOWER_BOTTOM")
update!(bc_lower_elements, "displacement 3", 0.0)
add_elements!(bc_lower, bc_lower_elements)

bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
bc_sym13_elements_1 = create_surface_elements(mesh, "LOWER_SYM13")
bc_sym13_elements_2 = create_surface_elements(mesh, "UPPER_SYM13")
update!(bc_sym13_elements_1, "displacement 2", 0.0)
update!(bc_sym13_elements_2, "displacement 2", 0.0)
add_elements!(bc_sym13, bc_sym13_elements_1, bc_sym13_elements_2)

bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
bc_sym23_elements_1 = create_surface_elements(mesh, "LOWER_SYM23")
bc_sym23_elements_2 = create_surface_elements(mesh, "UPPER_SYM23")
update!(bc_sym23_elements_1, "displacement 1", 0.0)
update!(bc_sym23_elements_2, "displacement 1", 0.0)
add_elements!(bc_sym23, bc_sym23_elements_1, bc_sym23_elements_2)

interface = Problem(Mortar, "LOWER_TO_UPPER", 3, "displacement")
interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
update!(interface_slave_elements, "master elements", interface_master_elements)
add_elements!(interface, interface_slave_elements, interface_master_elements)

removed_dofs = [1316, 1319, 1325, 1358, 1361, 1387, 1388, 1391, 1492, 1597, 1600, 1627, 1630, 1657]
append!(interface.assembly.removed_dofs, removed_dofs)

interface.properties.linear_surface_elements = false
interface.properties.split_quadratic_slave_elements = false
interface.properties.split_quadratic_master_elements = false
interface.properties.adjust = true
interface.properties.dual_basis = true
interface.properties.alpha = 0.2

analysis = Analysis(Linear)
add_problems!(analysis, upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)
xdmf = Xdmf("dl_quad_disp_results"; overwrite=true)
add_results_writer!(analysis, xdmf)
run!(analysis)

displacement = interface("displacement", 0.0)
u3 = [u[3] for u in values(displacement)]
maxabsu3 = maximum(abs.(u3))
stdabsu3 = std(abs.(u3))
@debug("tet10 block: max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
@test isapprox(stdabsu3, 0.0; atol=1.0e-6)
