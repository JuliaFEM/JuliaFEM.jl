# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test
using AsterReader: RMEDFile, aster_read_nodes, aster_read_data

## Two rings
# RING1 = inner, RING2 = outer, RINGS combined mesh. Set T=1.0 for
# inner ring and T=2.0 for outer ring, measure temperature from middle of ring.
# Results are calculated using Code Aster for comparison.

# test 3d heat, two rings, and compare to CA solution
meshfile = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(meshfile, "RINGS_UNION")

rings = Problem(Heat, "RINGS", 1)
rings_elements_1 = create_elements(mesh, "RING1")
rings_elements_2 = create_elements(mesh, "RING2")
update!(rings_elements_1, "thermal conductivity", 1.0)
update!(rings_elements_2, "thermal conductivity", 1.0)
add_elements!(rings, rings_elements_1)
add_elements!(rings, rings_elements_2)

bc_inner = Problem(Dirichlet, "INNER SURFACE", 1, "temperature")
bc_inner_elements = create_elements(mesh, "RING1_INNER")
update!(bc_inner_elements, "temperature 1", 1.0)
add_elements!(bc_inner, bc_inner_elements)

bc_outer = Problem(Dirichlet, "OUTER SURFACE", 1, "temperature")
bc_outer_elements = create_elements(mesh, "RING2_OUTER")
update!(bc_outer_elements, "temperature 1", 2.0)
add_elements!(bc_outer, bc_outer_elements)

analysis = Analysis(Linear)
add_problems!(analysis, rings, bc_inner, bc_outer)
run!(analysis)

temp_jf = rings("temperature", 0.0)

fn = @__DIR__() * "/testdata/rings.rmed"
results = RMEDFile(fn)
nodes = aster_read_nodes(results)
temp_ca = aster_read_data(results, "TEMP")

rtol = [norm(temp_jf[j]-temp_ca[j]) / max(temp_jf[j],temp_ca[j]) for j in keys(temp_jf)]
@test maximum(rtol) < 1.0e-12
