# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# # Hertzian contact, 2d

# - from fenet d3613 advanced finite element contact benchmarks
# - a = 6.21 mm, pmax = 3585 MPa
# - this is a very sparse mesh and for that reason pmax is not very accurate
# - (only 6 elements in -20 .. 20 mm contact zone, 3 elements in contact

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using Logging
Logging.configure(level=INFO)
add_elements! = JuliaFEM.add_elements!

# read mesh

datadir = Pkg.dir("JuliaFEM", "examples", "2d_hertz_contact")
meshfile = joinpath(datadir, "hertz_2d_full.med")
mesh = aster_read_mesh(meshfile)

# define upper and lower bodies

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

# boundary conditions: support block to ground

bc_fixed = Problem(Dirichlet, "fixed", 2, "displacement")
bc_fixed.elements = create_elements(mesh, "FIXED")
update!(bc_fixed, "displacement 2", 0.0)

# symmetry boundary condition

bc_sym_23 = Problem(Dirichlet, "symmetry line 23", 2, "displacement")
bc_sym_23.elements = create_elements(mesh, "SYM23")
update!(bc_sym_23, "displacement 1", 0.0)

# add point load in negative y-direction, 35.0e3 kN

nid = find_nearest_node(mesh, [0.0, 100.0])
load = Problem(Elasticity, "point load", 2)
load.properties.formulation = :plane_strain
load.elements = [Element(Poi1, [nid])]
update!(load.elements, "geometry", mesh.nodes)
update!(load, "displacement traction force 2", -35.0e3)

# define contact

contact = Problem(Contact2D, "contact", 2, "displacement")
contact.properties.rotate_normals = true
contact_slave_elements = create_elements(mesh, "BLOCK_TO_CYLINDER")
contact_master_elements = create_elements(mesh, "CYLINDER_TO_BLOCK")
add_master_elements!(contact, contact_master_elements)
add_slave_elements!(contact, contact_slave_elements)

# nonlinear, quasistatic analysis

step = Analysis(Nonlinear)
add_problems!(step, [upper, lower, bc_fixed, bc_sym_23, load, contact])
xdmf = Xdmf("2d_hertz_results"; overwrite=true)
# todo for 2d
# for body in (upper, lower)
#     push!(body.postprocess_fields, "stress")
# end
add_results_writer!(step, xdmf)

# run the analysis

step()

# do some postprocessing, integrate resultant force in normal and tangential
# direction

Rn = 0.0
Rt = 0.0
time = 0.0
for sel in contact_slave_elements
    for ip in get_integration_points(sel)
        w = ip.weight*sel(ip, time, Val{:detJ})
        n = sel("normal", ip, time)
        t = sel("tangent", ip, time)
        la = sel("lambda", ip, time)
        Rn += w*dot(n, la)
        Rt += w*dot(t, la)
    end
end

println("2d hertz: Rn = $Rn, Rt = $Rt")

close(xdmf.hdf)

# ![](2d_hertz_contact/results_displacement.png)

