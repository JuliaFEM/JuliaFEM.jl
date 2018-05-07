# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

pkgdir = Pkg.dir("JuliaFEM")
datadir = joinpath(pkgdir, "test", first(splitext(basename(@__FILE__))))

# from fenet d3613 advanced finite element contact benchmarks
# a = 6.21 mm, pmax = 3585 MPa
# this is a very sparse mesh and for that reason pmax is not very accurate
# (only 6 elements in -20 .. 20 mm contact zone, 3 elements in contact

meshfile = joinpath(datadir, "hertz_2d_full.med")
mesh = aster_read_mesh(meshfile)

upper = Problem(Elasticity, "CYLINDER", 2)
upper.properties.formulation = :plane_strain
upper.elements = create_elements(mesh, "CYLINDER")
update!(upper, "youngs modulus", 70.0e3)
update!(upper, "poissons ratio", 0.3)

lower = Problem(Elasticity, "BLOCK", 2)
lower.properties.formulation = :plane_strain
lower.elements = create_elements(mesh, "BLOCK")
update!(lower, "youngs modulus", 210.0e3)
update!(lower, "poissons ratio", 0.3)

# support block to ground
bc_fixed = Problem(Dirichlet, "fixed", 2, "displacement")
bc_fixed.elements = create_elements(mesh, "FIXED")
update!(bc_fixed, "displacement 2", 0.0)

# symmetry line
bc_sym_23 = Problem(Dirichlet, "symmetry line 23", 2, "displacement")
bc_sym_23.elements = create_elements(mesh, "SYM23")
update!(bc_sym_23, "displacement 1", 0.0)

nid = find_nearest_node(mesh, [0.0, 100.0])
#load = Problem(Dirichlet, "load", 2, "displacement")
load = Problem(Elasticity, "point load", 2)
load.properties.formulation = :plane_strain
load.elements = [Element(Poi1, [nid])]
#update!(load.elements, "displacement 2", -10.0)
update!(load, "displacement traction force 2", -35.0e3)

contact = Problem(Contact2D, "contact between block and cylinder", 2, "displacement")
contact.properties.rotate_normals = true
contact_slave_elements = create_elements(mesh, "CYLINDER_TO_BLOCK")
contact_master_elements = create_elements(mesh, "BLOCK_TO_CYLINDER")
add_master_elements!(contact, contact_master_elements)
add_slave_elements!(contact, contact_slave_elements)

solver = Solver(Nonlinear)
push!(solver, upper, lower, bc_fixed, bc_sym_23, load, contact)
solver()

node_ids, la = get_nodal_vector(contact_slave_elements, "lambda", 0.0)
node_ids, n = get_nodal_vector(contact_slave_elements, "normal", 0.0)
pres = [dot(ni, lai) for (ni, lai) in zip(n, la)]
#@test isapprox(maximum(pres), 4060.010799583303)
# 12 % error in maximum pressure
# integrate pressure in normal and tangential direction
Rn = 0.0
Rt = 0.0
Q = [0.0 -1.0; 1.0 0.0]
time = 0.0
for sel in contact_slave_elements
    for ip in get_integration_points(sel)
        w = ip.weight*sel(ip, time, Val{:detJ})
        n = sel("normal", ip, time)
        t = Q'*n
        la = sel("lambda", ip, time)
        Rn += w*dot(n, la)
        Rt += w*dot(t, la)
    end
end
info("2d hertz: Rn = $Rn, Rt = $Rt")
info("2d hertz: maximum pressure pmax = ", maximum(pres))
@test isapprox(maximum(pres), 3585.0; rtol = 0.13)
# under 0.15 % error in resultant force
@test isapprox(Rn, 35.0e3; rtol=0.020)
@test isapprox(Rt, 0.0; atol=200.0)
