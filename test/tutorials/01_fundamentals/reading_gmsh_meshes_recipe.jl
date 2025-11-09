# Recipe to generate reading_gmsh_meshes.msh
# 
# This file shows how the test mesh was created using Gmsh.jl.
# Run this file to regenerate the mesh if needed.
#
# Mesh: 10-element square (2×5 structured quad mesh)
# Domain: [0,1] × [0,1]
# Element type: Quad4 (4-node quadrilaterals)

using Gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("square_10_elements")

# Geometry: Unit square
lc = 0.5  # Characteristic length (controls element size)

# Points (corners of square)
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc, 2)
gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc, 3)
gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc, 4)

# Lines (edges of square)
gmsh.model.geo.addLine(1, 2, 1)  # Bottom
gmsh.model.geo.addLine(2, 3, 2)  # Right
gmsh.model.geo.addLine(3, 4, 3)  # Top
gmsh.model.geo.addLine(4, 1, 4)  # Left

# Surface (the square itself)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Synchronize geometry
gmsh.model.geo.synchronize()

# Structured mesh (2 elements in x, 5 in y = 10 quads)
gmsh.model.mesh.setTransfiniteCurve(1, 3)  # Bottom: 3 nodes = 2 elements
gmsh.model.mesh.setTransfiniteCurve(3, 3)  # Top: 3 nodes = 2 elements
gmsh.model.mesh.setTransfiniteCurve(2, 6)  # Right: 6 nodes = 5 elements
gmsh.model.mesh.setTransfiniteCurve(4, 6)  # Left: 6 nodes = 5 elements
gmsh.model.mesh.setTransfiniteSurface(1)
gmsh.model.mesh.setRecombine(2, 1)  # Combine triangles into quads

# Physical groups (for boundary conditions)
gmsh.model.addPhysicalGroup(2, [1], 1, "DOMAIN")   # Surface elements
gmsh.model.addPhysicalGroup(1, [1], 2, "BOTTOM")   # Bottom edge
gmsh.model.addPhysicalGroup(1, [2], 3, "RIGHT")    # Right edge  
gmsh.model.addPhysicalGroup(1, [3], 4, "TOP")      # Top edge
gmsh.model.addPhysicalGroup(1, [4], 5, "LEFT")     # Left edge

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Save to file (v4.1 format - newest)
output_file = joinpath(@__DIR__, "reading_gmsh_meshes.msh")
gmsh.write(output_file)

# Print mesh statistics
println("Mesh generated successfully!")
println("Output file: $output_file")
println("Nodes: ", length(gmsh.model.mesh.getNodes()[1]))
entities = gmsh.model.getEntities(2)  # 2D entities (surfaces)
for entity in entities
    elements = gmsh.model.mesh.getElements(entity[1], entity[2])
    println("Elements: ", length(elements[2][1]))
end

# Cleanup
gmsh.finalize()

println("\nTo view the mesh:")
println("  gmsh $output_file")
