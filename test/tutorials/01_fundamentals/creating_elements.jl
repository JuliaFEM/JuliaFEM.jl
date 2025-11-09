# # Tutorial 1: Creating Elements and Fields
#
# This tutorial introduces the fundamental concepts of JuliaFEM:
# - Creating nodes and elements
# - Updating element fields
# - Understanding the field concept
#
# ## Prerequisites
# 
# Basic Julia knowledge. This is the first tutorial - no prior JuliaFEM experience needed!

using JuliaFEM
using Test

# ## Nodes and Coordinates
#
# In finite element analysis, everything starts with **nodes** - points in space
# where we'll compute displacements, temperatures, or other field values.
#
# In JuliaFEM, nodes are simply vectors of coordinates. Let's create a few nodes
# for a 2D square:

node1 = [0.0, 0.0]  # Bottom-left corner
node2 = [1.0, 0.0]  # Bottom-right corner  
node3 = [1.0, 1.0]  # Top-right corner
node4 = [0.0, 1.0]  # Top-left corner

# We typically store nodes in a dictionary for easy access by node ID:

X = Dict(
    1 => node1,
    2 => node2,
    3 => node3,
    4 => node4
)

# ## Elements
#
# An **element** connects nodes and defines:
# - The shape (linear, quadratic, etc.)
# - The interpolation (how fields vary within the element)
# - The physics (what equations to solve)
#
# Let's create a 4-node quadrilateral element (`Quad4`):

element = Element(Quad4, (1, 2, 3, 4))

# The tuple `(1, 2, 3, 4)` specifies which nodes this element connects.
# Node numbering follows a counter-clockwise convention:
#
# ```
# 4 ---- 3
# |      |
# |      |  
# 1 ---- 2
# ```

# ## The Field Concept
#
# In JuliaFEM, everything is a **field**. A field is data associated with an element
# that can vary in space and time:
#
# - **Geometry field:** Node coordinates
# - **Material fields:** Young's modulus, Poisson's ratio
# - **Load fields:** Body forces, surface tractions
# - **Solution fields:** Displacements, temperatures
#
# Fields are updated using the `update!` function.

# ### Geometry Field
#
# First, we tell the element where its nodes are located:

update!(element, "geometry", X)

# Now the element knows its shape and can compute things like its area,
# jacobian, etc.

# ### Material Fields
#
# Let's add material properties (for an elasticity problem):

update!(element, "youngs modulus", 200.0e3)  # 200 GPa steel
update!(element, "poissons ratio", 0.3)

# ### Load Fields
#
# We can apply loads as fields too. For example, a body force in the y-direction:

update!(element, "displacement load 2", 10.0)  # 10 N/m³ in y-direction

# The naming convention:
# - `"displacement load 1"` = body force in x-direction
# - `"displacement load 2"` = body force in y-direction  
# - `"displacement load 3"` = body force in z-direction (3D)

# ## Field Access
#
# We can retrieve fields using function call syntax:

geom = element("geometry", 0.0)  # Get geometry at time t=0.0
E = element("youngs modulus", 0.0)
ν = element("poissons ratio", 0.0)

# Fields can be time-dependent! The second argument is the time value.
# For static fields (like geometry), the time doesn't matter.

# ## Testing Our Understanding
#
# Let's verify everything works as expected:

@testset "Element Creation" begin
    # Element should have 4 nodes
    @test length(element.connectivity) == 4

    # Geometry field returns a tuple of node coordinates
    @test length(geom) == 4  # 4 nodes
    @test geom[1] == [0.0, 0.0]  # First node

    # Material properties should be retrievable
    @test E == 200.0e3
    @test ν == 0.3
end

# ## Multiple Elements
#
# Real problems have many elements. Let's create a vector of elements:

elements = [
    Element(Quad4, (1, 2, 3, 4)),
    Element(Quad4, (2, 5, 6, 3))  # Adjacent element (assuming nodes 5, 6 exist)
]

# We can update fields for all elements at once:

update!(elements, "youngs modulus", 200.0e3)
update!(elements, "poissons ratio", 0.3)

# This is more efficient than updating each element individually.

# ## What We Learned
#
# ✅ Nodes are just coordinate vectors  
# ✅ Elements connect nodes using connectivity tuples  
# ✅ Everything is a field (geometry, materials, loads, solutions)  
# ✅ Fields are updated with `update!(element, "field_name", value)`  
# ✅ Fields are accessed with `element("field_name", time)`  
# ✅ Multiple elements can be updated together
#
# ## Next Steps
#
# - **Tutorial 2:** Reading meshes from files (ABAQUS .inp format)
# - **Tutorial 3:** Basis functions and shape function evaluation
# - **Tutorial 4:** Solving your first problem (1D elasticity)
#
# ## Further Reading
#
# - JuliaFEM paper: [Frondelius & Aho (2017)](https://doi.org/10.23998/rm.64224)
# - Field concept: Based on Abaqus field definitions
