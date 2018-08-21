# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test
using AsterReader: RMEDFile, aster_read_nodes, aster_read_data

#=
Two rings, RING1 is inner, RING2 is outer. Inner diameter is from 0.8 .. 0.9 and
outer ring is 0.9 .. 1.0. Contact surface pair is RING1_OUTER <- RING2_INNER.
Put constant temperature 1.0 for inner surface of inner ring and 2.0 for outer
surface of outer ring. We should expect constant temperature in contact surface.
This is conforming mesh so result should match to the conforming situation.
=#

## test that curved interface transfers constant field without error, two rings problem

meshfile = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(meshfile, "RINGS")

ring1 = Problem(Heat, "RING1", 1)
ring1_elements = create_elements(mesh, "RING1")
update!(ring1_elements, "thermal conductivity", 1.0)
add_elements!(ring1, ring1_elements)

ring2 = Problem(Heat, "RING2", 1)
ring2_elements = create_elements(mesh, "RING2")
update!(ring2_elements, "thermal conductivity", 1.0)
add_elements!(ring2, ring2_elements)

bc_inner = Problem(Dirichlet, "INNER SURFACE", 1, "temperature")
bc_inner_elements = create_elements(mesh, "RING1_INNER")
update!(bc_inner_elements, "temperature 1", 1.0)
add_elements!(bc_inner, bc_inner_elements)

bc_outer = Problem(Dirichlet, "OUTER SURFACE", 1, "temperature")
bc_outer_elements = create_elements(mesh, "RING2_OUTER")
update!(bc_outer_elements, "temperature 1", 2.0)
add_elements!(bc_outer, bc_outer_elements)

interface = Problem(Mortar, "interface between rings", 1, "temperature")
interface_slave = create_elements(mesh, "RING1_OUTER")
interface_master = create_elements(mesh, "RING2_INNER")
update!(interface_slave, "master elements", interface_master)
add_elements!(interface, interface_slave, interface_master)

analysis = Analysis(Linear)
add_problems!(analysis, ring1, ring2, bc_inner, bc_outer, interface)
run!(analysis)

fn = @__DIR__() * "/testdata/rings.rmed"
results = RMEDFile(fn)
nodes = aster_read_nodes(results)
temp_ca = aster_read_data(results, "TEMP")

sorted_node_ids = sort(collect(keys(nodes)))
node_coords = [nodes[j] for j in sorted_node_ids]
node_temps_jf = [analysis("temperature", node_coords[j], 0.0) for j in sorted_node_ids]
node_temps_ca = [temp_ca[j] for j in sorted_node_ids]
@test isapprox(node_temps_jf, node_temps_ca; rtol=1.0e-12)
