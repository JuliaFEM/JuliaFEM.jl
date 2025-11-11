---
title: "Quick Start: Linear Elasticity"
date: 2025-11-10
author: "JuliaFEM Team"
status: "Draft"
tags: ["user-guide", "elasticity", "quickstart", "tutorial"]
---

## Overview

This guide shows how to solve a simple linear elasticity problem using JuliaFEM's GPU-accelerated solver.

**What you'll learn:**

- How to create a mesh with Gmsh
- How to define materials and boundary conditions
- How to solve and visualize results

**Example problem:** Cantilever beam with fixed end and pressure load.

## Step 1: Create the Mesh

First, create a 3D mesh using Gmsh:

```julia
using Gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("cantilever")

# Geometry: 10m × 1m × 1m beam
L, W, H = 10.0, 1.0, 1.0
box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
gmsh.model.occ.synchronize()

# Define physical groups for boundary conditions
surfaces = gmsh.model.getBoundary([(3, box)], false, false, true)

for surf in surfaces
    _, surf_id = surf
    # Get surface center to identify it
    com = gmsh.model.occ.getCenterOfMass(2, surf_id)
    
    if abs(com[1]) < 1e-6  # X = 0 (fixed end)
        gmsh.model.addPhysicalGroup(2, [surf_id], -1, "FixedEnd")
    elseif abs(com[3] - H) < 1e-6  # Z = H (top surface for pressure)
        gmsh.model.addPhysicalGroup(2, [surf_id], -1, "PressureSurface")
    end
end

# Add volume physical group
gmsh.model.addPhysicalGroup(3, [box], -1, "Volume")

# Generate 3D tetrahedral mesh
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
gmsh.model.mesh.generate(3)

# Save mesh
gmsh.write("cantilever_beam.msh")
gmsh.finalize()
```

**Key concepts:**

- Physical groups label surfaces/volumes for boundary conditions
- `"FixedEnd"` - nodes that will be constrained (Dirichlet BC)
- `"PressureSurface"` - nodes where pressure is applied (Neumann BC)
- `"Volume"` - elements for assembly

## Step 2: Load the Mesh

```julia
include("src/gpu_elasticity.jl")
using .GPUElasticity

# Read the mesh file
mesh = read_gmsh_mesh("cantilever_beam.msh")

println("Mesh info:")
println("  Nodes: $(size(mesh.nodes, 2))")
println("  Elements: $(size(mesh.elements, 2))")
```

**What you get:**

- `mesh.nodes` - 3×n_nodes matrix of coordinates
- `mesh.elements` - 4×n_elements matrix of connectivity (Tet4)
- `mesh.physical_groups` - Dictionary mapping names to node/element IDs

## Step 3: Extract Boundary Condition Nodes

```julia
using .GPUElasticity.GmshReader: get_surface_nodes

# Get nodes for each boundary condition
fixed_nodes = get_surface_nodes(mesh, "FixedEnd")
pressure_nodes = get_surface_nodes(mesh, "PressureSurface")

println("Boundary conditions:")
println("  Fixed nodes: $(length(fixed_nodes))")
println("  Pressure nodes: $(length(pressure_nodes))")
```

**Boundary condition types:**

1. **Dirichlet (fixed_nodes):** Zero displacement constraint
   - Nodes cannot move (u = 0)
   - Models supports, clamps, symmetry

2. **Neumann (pressure_nodes):** Applied force/pressure
   - External load on surface
   - Models traction, pressure, point forces

## Step 4: Define Material

```julia
# Create material (steel)
material = ElasticMaterial(
    210e9,  # E - Young's modulus [Pa]
    0.3     # ν - Poisson's ratio [-]
)
```

**Common materials:**

| Material | E (GPa) | ν |
|----------|---------|---|
| Steel | 200-210 | 0.27-0.30 |
| Aluminum | 69 | 0.33 |
| Concrete | 30-40 | 0.15-0.20 |
| Rubber | 0.01-0.1 | 0.48-0.50 |

## Step 5: Create Physics Problem

```julia
# Define the complete problem
physics = ElasticityPhysics(
    mesh,              # Mesh with geometry
    material,          # Material properties
    fixed_nodes,       # Dirichlet BC nodes
    pressure_nodes,    # Neumann BC nodes
    10e6               # Pressure magnitude [Pa] = 10 MPa
)
```

**What ElasticityPhysics contains:**

- Mesh (nodes, elements, connectivity)
- Material (E, ν for isotropic linear elasticity)
- Fixed nodes (where displacement = 0)
- Pressure nodes (where external load is applied)
- Pressure value (load magnitude)

## Step 6: Solve

```julia
# Solve on GPU with iterative solver
result = solve_elasticity_gpu(physics, tol=1e-6, max_iter=1000)

println("\nSolution converged!")
println("  CG iterations: $(result.iterations)")
println("  Final residual: $(result.residual)")
```

**Solver parameters:**

- `tol` - Convergence tolerance (default: 1e-6)
- `max_iter` - Maximum CG iterations (default: 1000)

**What you get:**

- `result.u` - Displacement field (3n_nodes vector)
- `result.iterations` - Number of CG iterations
- `result.residual` - Final residual norm

## Step 7: Post-Process Results

```julia
# Extract displacement components
n_nodes = size(mesh.nodes, 2)
u_x = result.u[1:3:end]
u_y = result.u[2:3:end]
u_z = result.u[3:3:end]

# Find maximum displacement
u_magnitude = sqrt.(u_x.^2 + u_y.^2 + u_z.^2)
max_disp = maximum(u_magnitude)
max_node = argmax(u_magnitude)

println("\nResults:")
println("  Max displacement: $(max_disp * 1000) mm")
println("  At node: $(max_node)")
println("  Location: $(mesh.nodes[:, max_node])")

# Compute stresses (requires element loop - TODO)
```

## Complete Example

Here's the full script combining all steps:

```julia
using Gmsh
include("src/gpu_elasticity.jl")
using .GPUElasticity
using .GPUElasticity.GmshReader: get_surface_nodes

# 1. Generate mesh
gmsh.initialize()
gmsh.model.add("cantilever")
L, W, H = 10.0, 1.0, 1.0
box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
gmsh.model.occ.synchronize()

# Label surfaces
surfaces = gmsh.model.getBoundary([(3, box)], false, false, true)
for surf in surfaces
    _, surf_id = surf
    com = gmsh.model.occ.getCenterOfMass(2, surf_id)
    if abs(com[1]) < 1e-6
        gmsh.model.addPhysicalGroup(2, [surf_id], -1, "FixedEnd")
    elseif abs(com[3] - H) < 1e-6
        gmsh.model.addPhysicalGroup(2, [surf_id], -1, "PressureSurface")
    end
end
gmsh.model.addPhysicalGroup(3, [box], -1, "Volume")

# Generate and save
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
gmsh.model.mesh.generate(3)
gmsh.write("cantilever.msh")
gmsh.finalize()

# 2. Load mesh
mesh = read_gmsh_mesh("cantilever.msh")

# 3. Define boundary conditions
fixed_nodes = get_surface_nodes(mesh, "FixedEnd")
pressure_nodes = get_surface_nodes(mesh, "PressureSurface")

# 4. Define material (steel)
material = ElasticMaterial(210e9, 0.3)

# 5. Create physics
physics = ElasticityPhysics(
    mesh,
    material,
    fixed_nodes,
    pressure_nodes,
    10e6  # 10 MPa pressure
)

# 6. Solve
result = solve_elasticity_gpu(physics)

# 7. Results
n_nodes = size(mesh.nodes, 2)
u_mag = sqrt.(
    result.u[1:3:end].^2 +
    result.u[2:3:end].^2 +
    result.u[3:3:end].^2
)

println("Max displacement: $(maximum(u_mag) * 1000) mm")
```

## Current Limitations (Linear Elasticity)

The current implementation (`gpu_elasticity.jl`) supports:

✅ **Working:**

- Linear elastic material (Hooke's law)
- Isotropic materials (E, ν constant)
- Small strain assumption
- Dirichlet BC (fixed displacement)
- Neumann BC (pressure on surfaces)
- Matrix-free CG solver
- GPU acceleration

❌ **Not yet implemented:**

- Nonlinear materials (plasticity, hyperelasticity)
- Large deformations (geometric nonlinearity)
- Material state variables (plastic strain, damage)
- Contact mechanics
- Dynamic analysis (time integration)
- Point forces (only surface pressure)

## Next Steps

**To extend to nonlinear elasticity**, we need:

1. **Material state at integration points**
   - Store plastic strain εₚ, hardening α, etc.
   - Update state during Newton iterations

2. **Newton-Raphson solver**
   - Replace CG with Newton loop
   - Compute tangent stiffness and residual
   - Line search for globalization

3. **Stress update algorithms**
   - Radial return for plasticity
   - Hyperelastic stress from strain energy
   - State management (old vs new state)

4. **Boundary condition updates**
   - Prescribed displacement (not just zero)
   - Follower forces (load direction changes)
   - Contact constraints

See `docs/src/book/` for design documents on these extensions.

## Troubleshooting

### Gmsh not found

```julia
using Pkg
Pkg.add("Gmsh")
```

### No CUDA device

CPU-only version coming soon. For now, requires NVIDIA GPU with CUDA.

### CG doesn't converge

- Increase `max_iter` parameter
- Check boundary conditions (mesh must be constrained)
- Add preconditioner (future work)

### Out of GPU memory

- Reduce mesh size (fewer elements)
- Use coarser mesh (`Mesh.MeshSizeMax` larger)
- Future: Distributed multi-GPU solver

## Where to Learn More

- **Theory:** `docs/src/book/elasticity_theory.md`
- **Implementation:** `src/gpu_elasticity.jl` (477 lines, well-commented)
- **Test:** `test/test_gpu_elasticity.jl`
- **Demo:** `demos/cantilever_beam_demo.jl`
- **Design:** `docs/src/book/design/gpu_elasticity_implementation.md`

## Summary

**Workflow:**

1. Generate mesh with Gmsh (label surfaces for BCs)
2. Load mesh into JuliaFEM
3. Extract boundary condition nodes
4. Define material properties
5. Create `ElasticityPhysics` struct
6. Solve with `solve_elasticity_gpu()`
7. Post-process displacement field

**Current status:** Linear elasticity works. Nonlinear extensions in progress.
