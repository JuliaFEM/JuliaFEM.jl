# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMCoupling.jl/blob/master/LICENSE

# # Cylinder in torsion using distributed coupling

# Distributed coupling can be used to distribute point moments as point forces
# to nodes which dont have rotational degrees of freedom.

# The model is a 3d cylinder, shown in picture.

# ![](example_dcoupling_cylindrical_beam/example_dcoupling_cylindrical_beam.png)

# Distributed coupling is used to distribute a torque load to the end of the
# cylinder. The other end is fixed. Node sets for coupling nodes, reference
# node and fixed face were made in the ABAQUS input file. Reference node is
# located in the centrum of the coupling nodes.

using JuliaFEM
using JuliaFEM: add_elements!, Problem
using JuliaFEM.Preprocess
using JuliaFEM.Abaqus: create_surface_elements
using FEMBase
using FEMCoupling
using FEMCoupling: add_reference_node!, add_coupling_nodes!

# reading mesh from ABAQUS input file

datadir = Pkg.dir("JuliaFEM", "examples", "distributed_coupling_cylindrical_beam")
mesh = abaqus_read_mesh(joinpath(datadir, "example_dcoupling_cylindrical_beam.inp"))
println("Number of nodes in a model: ", length(mesh.nodes))

# # Elements
# Creating elements for the body. Mesh and element types are defined in
# the ABAQUS input file. The cylinder body is named "Body1" in the input file.

cylinder_body = create_elements(mesh,"Body1")

# Updating material properties of steel for elements.

update!(cylinder_body, "youngs modulus", 210e3)
update!(cylinder_body, "poissons ratio", 0.3)

# Creating an elasticity problem and adding the elements to it.

cylinder_problem = Problem(Elasticity,"cylinder_problem",3)
add_elements!(cylinder_problem, cylinder_body)

# # Boundary conditions
# Creating Poi1-type elements as boundary condition elements to nodes of the
# node set Fixed_face_set. Poi1 element is a one node element.

bc_elements = [Element(Poi1, [j]) for j in mesh.node_sets[:Fixed_face_set]]

# Updating geometry for the bc elements

update!(bc_elements, "geometry", mesh.nodes)

# Fixing all displacements for the bc elements.

for i=1:3
    update!(bc_elements, "displacement $i", 0.0)
end

# Creating a bc problem and adding the bc elements to it.

bc = Problem(Dirichlet, "fixed", 3, "displacement")
add_elements!(bc, bc_elements)

# # Distributed coupling
# Creating Poi1 elements to nodes in coupling nodes set.

coupling_nodes = [Element(Poi1, [j]) for j in mesh.node_sets[:Coupling_nodes_set]]

# Updating geometry for the coupling nodes.

update!(coupling_nodes, "geometry", mesh.nodes)

# Creating Poi1 element for the reference node.

reference_node_id = collect(mesh.node_sets[:ref_node_set])
reference_node = Element(Poi1, reference_node_id)

# Updating geometry and applying a point moment for the reference node.

update!(reference_node, "geometry", mesh.nodes)
update!(reference_node, "point moment 3", 1500.0)

# Creating a coupling problem and adding coupling nodes and reference nodes to
# it.

coupling = Problem(Coupling, "cylind", 3, "displacement")
add_coupling_nodes!(coupling, coupling_nodes)
add_reference_node!(coupling, reference_node)

# # Analysis
# Creating a step and running the analysis. The cylinder_problem contains
# information about the body, its elements and their values. The bc contains
# information about boundary conditions and coupling contains information
# about distributed coupling.

step = Analysis(Nonlinear)
add_problems!(step, [cylinder_problem, bc, coupling])
run!(step)

# # Results

# Comparing calculated results with ABAQUS results. The node set
# circlenodes_set contains nodes which are on the outer face radius.
# These circle nodes should have the maximum displacement magnitude (norm(u)).
# With the first() function the first node in the set is chosen.

node_on_circle = first(mesh.node_sets[:circlenodes_set])

# declaring displacements at time 0.0 to variable u

time=0.0
u = cylinder_problem("displacement", time)[node_on_circle]
u_mag = norm(u)

# Making a testset.

using FEMBase.Test
@testset "displacement magnitude" begin
u_mag_expected=6.306e-4
@test isapprox(u_mag, u_mag_expected, rtol=1e-3)
end

# Printing node ids
println("reference node id = $(reference_node_id[1])")
println("node on circle id = $node_on_circle")
