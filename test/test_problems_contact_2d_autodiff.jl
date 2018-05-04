# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

pkg_dir = Pkg.dir("JuliaFEM")
datadir = joinpath(pkg_dir, "test", first(splitext(basename(@__FILE__))))

meshfile = joinpath(datadir, "block_2d.med")
mesh = aster_read_mesh(meshfile)

upper = Problem(mesh, Elasticity, "UPPER", 2)
lower = Problem(mesh, Elasticity, "LOWER", 2)

for body in [upper, lower]
    body.properties.formulation = :plane_stress
    update!(body, "youngs modulus", 288.0)
    update!(body, "poissons ratio", 1/3)
end

load = Problem(mesh, Elasticity, "UPPER_TOP", 2)
load.properties.formulation = :plane_stress
update!(load, "displacement traction force 2", 0.0 => 0.0)
update!(load, "displacement traction force 2", 1.0 => -28.8)
bc1 = Problem(mesh, Dirichlet, "LOWER_BOTTOM", 2, "displacement")
update!(bc1, "displacement 2", 0.0)
bc2 = Problem(mesh, Dirichlet, "LOWER_LEFT", 2, "displacement")
update!(bc2, "displacement 1", 0.0)
bc3 = Problem(mesh, Dirichlet, "UPPER_LEFT", 2, "displacement")
update!(bc3, "displacement 1", 0.0)

interface = Problem(Contact2DAD, "interface", 2, "displacement")
interface_slave_elements = create_elements(mesh, "LOWER_TOP")
interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
add_slave_elements!(interface, interface_slave_elements)
add_master_elements!(interface, interface_master_elements)
interface.properties.rotate_normals = true

# in LOWER_LEFT we have node belonging also to contact interface
# let's remove it from dirichlet bc
create_node_set_from_element_set!(mesh, "LOWER_LEFT")
nid = find_nearest_node(mesh, [0.0, 0.5]; node_set="LOWER_LEFT")
coords = mesh.nodes[nid]
info("nearest node to (0.0, 0.5) = $nid, coordinates = $coords")
dofs = [2*(nid-1)+1, 2*(nid-1)+2]
info("removing nid $nid, dofs $dofs from LOWER_LEFT")
push!(bc2.assembly.removed_dofs, dofs...)

solver = Solver(Nonlinear)
push!(solver, upper, lower, load, bc1, bc2, bc3, interface)

interface.assembly.u = zeros(48)
interface.assembly.la = zeros(48)

for body in [upper, lower]
    body.properties.geometric_stiffness = true
    body.properties.finite_strain = true
end

for time in [0.0, 1/3, 2/3, 1.0]
    interface.properties.iteration = 0
    solve!(solver, time)
end

node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 1.0)
node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 1.0)
node_ids, la = get_nodal_vector(interface.elements, "lambda", 1.0)
u2 = [u[2] for u in displacement]
f2 = [f[2] for f in la]
maxabsu2 = maximum(abs.(u2))
stdabsu2 = std(abs.(u2))
info("max(abs(u2)) = $maxabsu2, std(abs(u2)) = $stdabsu2")
@test isapprox(stdabsu2, 0.0; atol=1.0e-12)
maxabsf2 = maximum(abs.(f2))
stdabsf2 = std(abs.(f2))
info("max(abs(f2)) = $maxabsf2, std(abs(f2)) = $stdabsf2")
@test isapprox(stdabsf2, 0.0; atol=1.0e-12)
# for linear case pressure 28.8
@test isapprox(mean(abs.(f2)), 27.76616800689944; rtol=1.0e-3)
