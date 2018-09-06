# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# # JuliaFEM Linear Static Example

# ![](linear_static/freecad.png)

# ## Preprocessing

using JuliaFEM

# First we will read in the mesh. Geometry and mesh are greated with FreeCAD,
# where med format is selected for exporting. Mesh file consist also edge and
# surface mesh, which we will need to neglect later.

datadir = Pkg.dir("JuliaFEM", "examples", "linear_static")
meshfile = joinpath(datadir, "JuliaFEMSMP18.med")
mesh = aster_read_mesh(meshfile)

# Next we will create the model and define Elasticity. Also elements are added
# to the model.

model = Problem(Elasticity, "OTHER", 3)
model_elements = create_elements(mesh, "OTHER")

# Elements need material properties and they are defined next

update!(model_elements, "youngs modulus", 208.0E3)
update!(model_elements, "poissons ratio", 0.30)
update!(model_elements, "density", 7.80E-9)
add_elements!(model, model_elements)

# We can ignore Seg3 and Tri6 elements using `filter` with a special function
# returning true if element is something else than Seg3 or Tri6:

function is_not_Seg3_or_Tri6(element)
    return !isa(element, Union{Element{Seg3}, Element{Tri6}})
end

filter!(is_not_Seg3_or_Tri6, model.elements)

# The whole idea of the JuliaFEM input is to be a normal Julia script, where the
# user can freely define any functions needed to perform the task. Here we
# define a function, which finds nodes on the given plane yz, xz or xy from the
# given height.

function add_nodes_at_certain_plane_to_node_set!(mesh, name, vector_id, distance,
                                                 radius=6.0)
    for (node, coords) in mesh.nodes
        if isapprox(coords[vector_id], distance, atol=radius)
            add_node_to_node_set!(mesh, name, node)
        end
    end
    return nothing
end

# We will find nodes from the xz-plane going through point (0,50,0) or actually
# we previously defined the radius to be 6.0, which means (0,[44,56],0). In other
# words we will select each node, which second coordinate value is between 44
# and 56. This function will edit mesh object and add node set called `:mid_fixed`
# to it. 

add_nodes_at_certain_plane_to_node_set!(mesh, :mid_fixed, 2, 50.0)

# We need to somehow handle the i's dot. I looked the rough coordinates of the
# dot in FreeCAD and now we can search three closest nodes to these coordinates.
# Those will be added to the same set `:mid_fixed`.

ipoint = find_nearest_nodes(mesh, [165.0, 88.0, 10],3)
for poi in ipoint
    add_node_to_node_set!(mesh, :mid_fixed, poi)
end

# The fixed boundary conditions are defined next.

fixed = Problem(Dirichlet, "fixed", 3, "displacement")
fixed_elements = create_nodal_elements(mesh, "mid_fixed")
add_elements!(fixed, fixed_elements)
update!(fixed_elements, "displacement 1", 0.0)
update!(fixed_elements, "displacement 2", 0.0)
update!(fixed_elements, "displacement 3", 0.0)


# Let's use simple acceleration load.
update!(model_elements, "displacement load 1", 1.0)

# Finally the ´Analysis` couples everything togeter.
analysis = Analysis(Linear, model, fixed)

# Let's write resuls to Xdmf file

xdmf = Xdmf("model_results"; overwrite=true)
add_results_writer!(analysis, xdmf)

# This is how the stresses are requested
push!(model.postprocess_fields, "stress")

# Now we have all we need to run the analysis.

run!(analysis)

# ## Postprocessing

# In order to look the results, we will need to close the xdmf that it is actually
# written to the file from buffer.

close(xdmf)

# Finally when we open the model in ParaView and set some settings we have this
# end result.

# ![](linear_static/paraview.png)

# ## Testing

# First let's test that we have the output files writen to the disk

if VERSION < v"1.0.0"
    using Base.Test
else
    using Test
end

@test isfile("model_results.xmf")
@test isfile("model_results.h5")

# Secondly let's test that we have the same maximum displacement each time. 
# This is also an usefull example how to access the displacements values.

time = 0.0
u = analysis("displacement", time)
u_norms = Dict(i => norm(j) for (i, j) in u)
@test isapprox(maximum(values(u_norms)),2.4052929896922337)
