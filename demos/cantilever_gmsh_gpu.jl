# Cantilever Beam Demo - GPU Physics with Gmsh Mesh
#
# Uses Gmsh to generate a proper mesh with ~20 quadratic tetrahedrons (Tet10)
# Demonstrates:
# - Gmsh mesh generation (programmatic)
# - Quadratic elements (Tet10, Tri6)
# - Immutable element API
# - GPU solver with Physics{Elasticity}

using JuliaFEM
using CUDA
using Tensors

println("="^60)
println("Cantilever Beam - GPU Physics with Gmsh Mesh")
println("="^60)

# Include new GPU physics module
include("../src/gpu_physics_elasticity.jl")
using .GPUElasticityPhysics

# ============================================================================
# 1. Generate Mesh with Gmsh
# ============================================================================

println("\n[1] Generating mesh with Gmsh...")

# Check if gmsh is available
if !success(`which gmsh`)
    error("Gmsh not found! Install with: sudo apt install gmsh (Linux) or brew install gmsh (Mac)")
end

# Create Gmsh script
geo_file = "/tmp/cantilever_beam.geo"
msh_file = "/tmp/cantilever_beam.msh"

open(geo_file, "w") do f
    write(
        f,
        """
// Cantilever beam geometry
// Dimensions: L=4.0, W=1.0, H=1.0

SetFactory("OpenCASCADE");

// Create box
Box(1) = {0, 0, 0, 4.0, 1.0, 1.0};

// Define physical groups
Physical Volume("body") = {1};

// Fixed end (X=0)
Physical Surface("fixed") = {1};

// Free end (X=4) - not needed for load
Physical Surface("free") = {2};

// Top surface (Z=1) for pressure load
Physical Surface("pressure") = {6};

// Mesh settings
Mesh.CharacteristicLengthMin = 0.3;
Mesh.CharacteristicLengthMax = 0.5;
Mesh.ElementOrder = 2;  // Quadratic elements (Tet10)
Mesh.Algorithm3D = 4;   // Frontal Delaunay

// Generate 3D mesh
Mesh 3;
"""
    )
end

# Run Gmsh
println("  Running Gmsh...")
run(`gmsh $geo_file -3 -o $msh_file -format msh2`)

println("  ✓ Mesh generated: $msh_file")

# ============================================================================
# 2. Read Mesh (Simple Parser for MSH2 Format)
# ============================================================================

println("\n[2] Reading mesh...")

function read_msh2(filename::String)
    nodes = Dict{Int,Vector{Float64}}()
    elements = Dict{String,Vector{Vector{Int}}}()
    physical_groups = Dict{String,Vector{Vector{Int}}}()  # name => elements

    open(filename) do f
        section = ""
        while !eof(f)
            line = strip(readline(f))

            if line == "\$Nodes"
                section = "nodes"
                n_nodes = parse(Int, readline(f))
                for _ in 1:n_nodes
                    parts = split(readline(f))
                    node_id = parse(Int, parts[1])
                    x, y, z = parse(Float64, parts[2]), parse(Float64, parts[3]), parse(Float64, parts[4])
                    nodes[node_id] = [x, y, z]
                end
            elseif line == "\$Elements"
                section = "elements"
                n_elements = parse(Int, readline(f))
                for _ in 1:n_elements
                    parts = split(readline(f))
                    elem_id = parse(Int, parts[1])
                    elem_type = parse(Int, parts[2])
                    n_tags = parse(Int, parts[3])

                    # Read tags
                    physical_tag = n_tags >= 1 ? parse(Int, parts[4]) : 0

                    # Connectivity starts after tags
                    offset = 4 + n_tags

                    # Element type: 4=Tet4, 11=Tet10, 2=Tri3, 9=Tri6
                    if elem_type == 11  # Tet10
                        connectivity = [parse(Int, parts[i]) for i in offset:offset+9]
                        if !haskey(elements, "Tet10")
                            elements["Tet10"] = Vector{Int}[]
                        end
                        push!(elements["Tet10"], connectivity)

                        # Store by physical group
                        group_name = "body_$physical_tag"
                        if !haskey(physical_groups, group_name)
                            physical_groups[group_name] = Vector{Int}[]
                        end
                        push!(physical_groups[group_name], connectivity)

                    elseif elem_type == 9  # Tri6
                        connectivity = [parse(Int, parts[i]) for i in offset:offset+5]
                        if !haskey(elements, "Tri6")
                            elements["Tri6"] = Vector{Int}[]
                        end
                        push!(elements["Tri6"], connectivity)

                        # Store by physical group (surface elements)
                        group_name = "surface_$physical_tag"
                        if !haskey(physical_groups, group_name)
                            physical_groups[group_name] = Vector{Int}[]
                        end
                        push!(physical_groups[group_name], connectivity)

                    elseif elem_type == 2  # Tri3
                        connectivity = [parse(Int, parts[i]) for i in offset:offset+2]
                        if !haskey(elements, "Tri3")
                            elements["Tri3"] = Vector{Int}[]
                        end
                        push!(elements["Tri3"], connectivity)

                        # Store by physical group
                        group_name = "surface_$physical_tag"
                        if !haskey(physical_groups, group_name)
                            physical_groups[group_name] = Vector{Int}[]
                        end
                        push!(physical_groups[group_name], connectivity)
                    end
                end
            elseif line == "\$EndNodes" || line == "\$EndElements"
                section = ""
            end
        end
    end

    return nodes, elements, physical_groups
end

node_dict, elem_dict, physical_groups = read_msh2(msh_file)

println("\nPhysical groups found:")
for (name, elems) in physical_groups
    println("  $name: $(length(elems)) elements")
end

# Convert nodes to matrix
n_nodes = length(node_dict)
node_ids = sort(collect(keys(node_dict)))
node_map = Dict(id => i for (i, id) in enumerate(node_ids))

nodes = zeros(3, n_nodes)
for (id, i) in node_map
    nodes[:, i] = node_dict[id]
end

println("  Nodes: $n_nodes")
println("  Tet10 elements: $(length(get(elem_dict, "Tet10", [])))")
println("  Tri6 elements: $(length(get(elem_dict, "Tri6", [])))")

# ============================================================================
# 3. Create Body Elements (Tet10)
# ============================================================================

println("\n[3] Creating body elements...")

body_elements = Element[]

for conn_gmsh in elem_dict["Tet10"]
    # Renumber nodes (Gmsh → 1-based sequential)
    conn = [node_map[id] for id in conn_gmsh]

    # Extract coordinates
    X_elem = nodes[:, conn]

    # Create immutable element
    el = Element(Tet10, Lagrange{Tet10,2}, tuple(conn...);
        fields=(geometry=X_elem,
            youngs_modulus=210e9,    # Steel
            poissons_ratio=0.3))

    push!(body_elements, el)
end

println("  Body elements created: $(length(body_elements))")

# ============================================================================
# 4. Create Physics{Elasticity}
# ============================================================================

println("\n[4] Creating Physics{Elasticity}...")

physics = Physics(Elasticity, "cantilever beam", 3)
physics.properties.formulation = :continuum
physics.properties.finite_strain = false

add_elements!(physics, body_elements)

println("  Physics: $(physics.name)")
println("  Elements in physics: $(length(physics.body_elements))")

# ============================================================================
# 5. Add Boundary Conditions
# ============================================================================

println("\n[5] Adding boundary conditions...")

# Dirichlet BC: Use "fixed" physical surface (tag 1)
# Extract all unique nodes from fixed surface elements
fixed_nodes_set = Set{Int}()
if haskey(physical_groups, "surface_1")  # Physical tag 1 = "fixed"
    for conn_gmsh in physical_groups["surface_1"]
        for node_id in conn_gmsh
            push!(fixed_nodes_set, node_map[node_id])
        end
    end
end
fixed_nodes = sort(collect(fixed_nodes_set))

println("  Fixing $(length(fixed_nodes)) nodes from 'fixed' surface (all DOFs)")

for node in fixed_nodes
    add_dirichlet!(physics, [node], [1, 2, 3], 0.0)
end

# Neumann BC: Use "pressure" physical surface (tag 3)
println("  Applying pressure load on 'pressure' surface...")

pressure = -1e6  # -1 MPa in -Z direction
traction = Vec{3}((0.0, 0.0, pressure))

n_pressure_surfaces = 0
if haskey(physical_groups, "surface_3")  # Physical tag 3 = "pressure"
    for conn_gmsh in physical_groups["surface_3"]
        conn = [node_map[id] for id in conn_gmsh]
        X_surf = nodes[:, conn]

        surf_el = Element(Tri6, Lagrange{Tri6,2}, tuple(conn...);
            fields=(geometry=X_surf,))

        add_neumann!(physics, surf_el, traction)
        n_pressure_surfaces += 1
    end
end

println("  Dirichlet BCs: $(length(physics.bc_dirichlet.node_ids)) nodes")
println("  Neumann BCs: $n_pressure_surfaces surface elements")

# ============================================================================
# 6. Solve on GPU
# ============================================================================

println("\n[6] Solving on GPU...")
println("  DOFs: $(3 * n_nodes)")
println("  Initializing GPU data...")

result = solve_elasticity_gpu!(
    physics;
    time=0.0,
    tol=1e-6,
    max_iter=2000  # More DOFs, might need more iterations
)

println("\n" * "="^60)
println("SOLUTION")
println("="^60)
println("  Iterations: $(result.iterations)")
println("  Residual: $(result.residual)")

# ============================================================================
# 7. Post-Process Results
# ============================================================================

println("\n[7] Post-processing...")

u = result.u
n_dofs = length(u)

# Extract displacement components
u_x = u[1:3:end]
u_y = u[2:3:end]
u_z = u[3:3:end]

# Compute magnitude
u_mag = sqrt.(u_x .^ 2 + u_y .^ 2 + u_z .^ 2)

println("\nDisplacement Statistics:")
println("  Max |u|: $(maximum(u_mag) * 1000) mm")
println("  Max u_x: $(maximum(abs.(u_x)) * 1000) mm")
println("  Max u_y: $(maximum(abs.(u_y)) * 1000) mm")
println("  Max u_z: $(maximum(abs.(u_z)) * 1000) mm")

# Find max displacement location
max_idx = argmax(u_mag)
println("\nMax displacement at node $max_idx:")
println("  Location: $(nodes[:, max_idx])")
println("  u = [$(u_x[max_idx]*1000), $(u_y[max_idx]*1000), $(u_z[max_idx]*1000)] mm")

# ============================================================================
# 8. Validate
# ============================================================================

println("\n[8] Validation...")

# Find free end nodes from "free" physical surface (tag 2)
free_end_nodes_set = Set{Int}()
if haskey(physical_groups, "surface_2")  # Physical tag 2 = "free"
    for conn_gmsh in physical_groups["surface_2"]
        for node_id in conn_gmsh
            push!(free_end_nodes_set, node_map[node_id])
        end
    end
end
free_end_nodes = collect(free_end_nodes_set)

free_end_disp = maximum(u_mag[free_end_nodes])
fixed_end_disp = maximum(u_mag[fixed_nodes])

println("  Free end max |u|: $(free_end_disp * 1000) mm")
println("  Fixed end max |u|: $(fixed_end_disp * 1000) mm")

if fixed_end_disp < 1e-10
    println("  ✓ Fixed end has zero displacement (good!)")
else
    println("  ✗ Fixed end displacement > 0 (bad!)")
end

if free_end_disp > 1e-6
    println("  ✓ Free end has non-zero displacement (good!)")
else
    println("  ✗ Free end displacement ≈ 0 (bad!)")
end

println("\n" * "="^60)
println("Demo complete!")
println("="^60)

# ============================================================================
# Summary
# ============================================================================

println("\nMesh Statistics:")
println("  Nodes: $n_nodes")
println("  DOFs: $(3 * n_nodes)")
println("  Tet10 elements: $(length(body_elements))")
println("  Tri6 surface elements: $n_pressure_surfaces")
println()
println("Performance:")
println("  CG iterations: $(result.iterations)")
println("  Final residual: $(result.residual)")
println("  GPU solver: Pure device code (no CPU-GPU transfer)")
