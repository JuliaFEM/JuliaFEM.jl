---
title: "From Gmsh to Physics: Complete Heat Equation Tutorial"
author: "Jukka Aho"
date: "2025-11-09"
categories: ["Tutorial", "Getting Started"]
tags: ["gmsh", "heat-equation", "academic-usage", "method-of-lines"]
issue: "183"
description: "Complete workflow from mesh generation to FEM assembly, addressing academic usage without built-in physics"
---

# From Gmsh to Physics: Complete Heat Equation Tutorial

**Purpose:** Demonstrate complete FEM workflow from mesh generation through physics, addressing [Issue #183](https://github.com/JuliaFEM/JuliaFEM.jl/issues/183) - using JuliaFEM for academic/research purposes without built-in physics.

**What You'll Learn:**
- Generate meshes with Gmsh
- Load meshes into JuliaFEM
- Assemble stiffness and mass matrices
- Extract matrices for external solvers (DifferentialEquations.jl, etc.)
- Solve the heat equation using method of lines

---

## Problem Statement

We'll solve the transient heat equation on a unit square Ω = [0,1] × [0,1]:

**Governing equation:**
```
∂u/∂t = α∇²u + f(x,y,t)    in Ω
```

**Boundary conditions:**
- Dirichlet: u = 0 on left edge (x = 0)
- Neumann: ∂u/∂n = 0 on other edges (natural BC)

**Initial condition:**
```
u(x,y,0) = sin(πx)sin(πy)
```

**Parameters:**
- α = 1.0 (thermal diffusivity)
- f = 0 (no heat source)

---

## Step 1: Mesh Generation with Gmsh

### Why Gmsh?

Gmsh is a free, open-source mesh generator that:
- Supports complex geometries (2D and 3D)
- Generates quality meshes with various element types
- Has scripting capabilities (`.geo` files)
- Widely used in academic and industrial FEM

### Creating the Geometry File

Create `unit_square.geo`:

```geo
// Gmsh geometry file: Unit square mesh
// Generate with: gmsh -2 unit_square.geo -o unit_square.msh

// Mesh element size
lc = 0.1;

// Corner points
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Edges
Line(1) = {1, 2};  // Bottom
Line(2) = {2, 3};  // Right
Line(3) = {3, 4};  // Top
Line(4) = {4, 1};  // Left

// Surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups for boundary conditions
Physical Line("bottom") = {1};
Physical Line("right") = {2};
Physical Line("top") = {3};
Physical Line("left") = {4};
Physical Surface("body") = {1};

// Use triangular elements
Mesh.ElementOrder = 1;  // Linear elements (Tri3)
Mesh.Algorithm = 6;     // Frontal-Delaunay for 2D
```

**Key concepts:**
- `lc` controls mesh density (smaller = finer mesh)
- Physical groups (`Physical Line`, `Physical Surface`) label regions for boundary conditions
- `ElementOrder = 1` gives linear triangular elements (Tri3)

### Generating the Mesh

```bash
gmsh -2 unit_square.geo -o unit_square.msh
```

**Options:**
- `-2`: Generate 2D mesh
- `-o`: Output file name

**Result:** `unit_square.msh` contains nodes and element connectivity.

### Understanding the Mesh Format

Gmsh `.msh` format (ASCII version 2.2):

```
$MeshFormat
2.2 0 8
$EndMeshFormat

$Nodes
121        # Number of nodes
1 0.0 0.0 0.0    # node_id x y z
2 0.1 0.0 0.0
...
$EndNodes

$Elements
240        # Number of elements
1 1 2 4 1 1 2      # elem_id type num_tags tags... node_ids...
2 2 2 1 1 1 2 13   # type=2 is Tri3
...
$EndElements
```

**Element types:**
- Type 1: 2-node line (Seg2)
- Type 2: 3-node triangle (Tri3)
- Type 3: 4-node quadrilateral (Quad4)

---

## Step 2: Weak Formulation (Theory)

### Strong Form

The strong form (classical PDE) is:

```
∂u/∂t - α∇²u = f    in Ω
u = g                on Γ_D (Dirichlet boundary)
∂u/∂n = h            on Γ_N (Neumann boundary)
```

### Weak Form

Multiply by test function v and integrate by parts:

```
∫_Ω v(∂u/∂t) dΩ + α∫_Ω ∇v·∇u dΩ = ∫_Ω vf dΩ + ∫_{Γ_N} vh dΓ
```

### Galerkin Approximation

Approximate u and v with finite element basis functions:

```
u(x,y,t) ≈ Σᵢ uᵢ(t) Nᵢ(x,y)
v(x,y)   ≈ Σⱼ vⱼ Nⱼ(x,y)
```

Substitute and collect terms:

```
M du/dt + K u = f
```

Where:
- **M** = mass matrix: `Mᵢⱼ = ∫_Ω Nᵢ Nⱼ dΩ`
- **K** = stiffness matrix: `Kᵢⱼ = α∫_Ω ∇Nᵢ·∇Nⱼ dΩ`
- **f** = force vector: `fᵢ = ∫_Ω Nᵢ f dΩ + ∫_{Γ_N} Nᵢ h dΓ`

This is a **system of ODEs** - the spatial discretization is complete, leaving only time dependence.

---

## Step 3: FEM Assembly in JuliaFEM

### Loading the Mesh

```julia
using JuliaFEM

# Read Gmsh mesh
mesh = abaqus_read_mesh("unit_square.msh")  # Or gmsh reader when available

# Inspect mesh
println("Nodes: ", length(mesh.nodes))
println("Elements: ", length(mesh.elements))
println("Sets: ", keys(mesh.element_sets))
```

### Creating Problem and Elements

```julia
# Heat transfer problem (domain)
body = Problem(Heat, "heat_body", 2)  # 2D problem
body_elements = create_elements(mesh, "body")

# Add material properties
thermal_conductivity = 1.0  # α
for element in body_elements
    update!(element, "thermal conductivity", thermal_conductivity)
end

add_elements!(body, body_elements)
```

**Key:** `update!` sets element properties. JuliaFEM stores these in element fields (now immutable NamedTuples for 130x speedup!).

### Boundary Conditions

```julia
# Dirichlet BC: u = 0 on left edge
bc = Problem(Dirichlet, "fixed_temp", 2, "temperature")
bc_elements = create_elements(mesh, "left")

for element in bc_elements
    update!(element, "temperature", 0.0)
end

add_elements!(bc, bc_elements)
```

**Neumann BC:** Natural boundary conditions (∂u/∂n = 0) require no explicit code - they're automatically satisfied by the weak form.

### Assembly Process

```julia
# Assemble at time t = 0
time = 0.0
assemble!(body, time)
assemble!(bc, time)
```

**What happens internally:**

1. **Loop over elements:**
   For each element e:
   
2. **Compute element matrices:**
   ```julia
   # Get element nodes and geometry
   X = [node.position for node in element.nodes]
   
   # Numerical integration over element
   for (ξ, w) in gauss_points(element)
       # Shape functions and derivatives
       N = basis(element, ξ)
       dN = grad_basis(element, ξ)
       
       # Jacobian (parametric → physical)
       J = dN' * X
       detJ = det(J)
       
       # Physical derivatives
       dN_dx = J \ dN
       
       # Local matrices
       Kₑ += w * detJ * α * (dN_dx * dN_dx')
       Mₑ += w * detJ * (N * N')
   end
   ```

3. **Scatter to global:**
   ```julia
   global_dofs = get_dofs(element)
   K[global_dofs, global_dofs] += Kₑ
   M[global_dofs, global_dofs] += Mₑ
   ```

4. **Apply Dirichlet BCs:**
   Zero out rows/columns for constrained DOFs.

---

## Step 4: Extracting Matrices (Issue #183)

**This is what Chris Rackauckas asked for!**

After assembly, extract matrices for use with external solvers:

```julia
# Get assembled system
K = body.assembly.K  # SparseMatrixCSC{Float64}
M = body.assembly.M  # SparseMatrixCSC{Float64}
f = body.assembly.f  # Vector{Float64}

# System is now: M * du/dt = -K * u + f
```

### Why Extract Matrices?

**Use cases:**
1. **DifferentialEquations.jl** - Advanced ODE solvers (Rosenbrock, IMEX, etc.)
2. **Krylov.jl** - Iterative linear solvers for large systems
3. **Custom time integration** - Research on novel time-stepping schemes
4. **Sensitivity analysis** - Automatic differentiation through solvers
5. **Optimal control** - Adjoint methods, PDE-constrained optimization

### Example: Solve with DifferentialEquations.jl

```julia
using OrdinaryDiffEq

# Initial condition: u₀ = sin(πx)sin(πy)
u0 = zeros(length(mesh.nodes))
for (i, node) in enumerate(mesh.nodes)
    x, y = node.position[1:2]
    u0[i] = sin(π*x) * sin(π*y)
end

# Apply boundary conditions to u0
apply_bc!(u0, bc)

# Define ODE: M * du/dt = -K * u + f
# Rearrange: du/dt = M \ (-K * u + f)
function heat_ode!(du, u, p, t)
    K, M, f = p
    du .= M \ (-K * u .+ f)
end

# Create ODE problem
tspan = (0.0, 1.0)
prob = ODEProblem(heat_ode!, u0, tspan, (K, M, f))

# Solve with adaptive Rosenbrock method
sol = solve(prob, Rosenbrock23())

# Solution is now in sol.u (array of states at different times)
```

**Advantages:**
- Adaptive time-stepping
- Stiff ODE solvers
- Event handling
- Sensitivity analysis
- GPU acceleration (CuArrays)

---

## Step 5: Complete Working Example

See `examples/gmsh_heat_equation/gmsh_heat_equation.jl` for full runnable code.

**Run it:**
```bash
cd examples/gmsh_heat_equation
gmsh -2 unit_square.geo -o unit_square.msh
julia --project=. gmsh_heat_equation.jl
```

**Output:**
- Mesh statistics
- Assembly information
- Extracted matrix sizes
- Instructions for next steps

---

## Method of Lines: Spatial Discretization Strategy

**Key insight:** FEM performs **spatial discretization only**, converting PDE → ODE system.

### Before FEM (PDE):
```
∂u(x,y,t)/∂t = α∇²u(x,y,t) + f(x,y,t)
```
- Infinite-dimensional: u depends on continuous (x,y)
- Cannot solve directly on computer

### After FEM (ODE):
```
M du(t)/dt = -K u(t) + f(t)
```
- Finite-dimensional: u is a vector of n nodal values
- Can solve with ODE integrators

**This separation is powerful:**
- **Space:** FEM handles complex geometry, boundary conditions
- **Time:** ODE solvers handle stiff systems, adaptivity, stability
- **Modularity:** Swap spatial/temporal methods independently

---

## Comparison: Built-in vs External Solvers

### Option 1: JuliaFEM Built-in (Easiest)

```julia
solver = Solver(Linear)
push!(solver, body, bc)
solver()  # Uses direct solver
results = solver.results
```

**Pros:** Simple, integrated, handles BCs automatically
**Cons:** Less control, fixed time-stepping, direct solver only

### Option 2: Extract Matrices (Flexible)

```julia
K, M, f = extract_matrices(body, bc)
prob = ODEProblem(heat_ode!, u0, tspan, (K, M, f))
sol = solve(prob, Tsit5())
```

**Pros:** Full control, adaptive methods, iterative solvers, research flexibility
**Cons:** More code, manual BC handling, need to understand ODE interface

**Recommendation:** Start with Option 1 for learning, move to Option 2 for research.

---

## Extensions and Next Steps

### 1. Nonlinear Problems

If α = α(u) (temperature-dependent conductivity):

```julia
function nonlinear_heat!(du, u, p, t)
    K, M, f = p
    # Reassemble K with current u
    K_nonlinear = assemble_stiffness(u)
    du .= M \ (-K_nonlinear * u .+ f)
end
```

### 2. Time-Dependent BCs

```julia
function bc_time(t)
    return sin(2π*t)  # Oscillating temperature
end

# Update f vector in ODE function
f_bc = apply_bc_vector(bc, t)
```

### 3. 3D Problems

Same workflow, just change:
- Geometry file to 3D (Tet4, Tet10 elements)
- Problem dimension: `Problem(Heat, "body", 3)`

### 4. Parallel Assembly

For large meshes (>1M elements):

```julia
using Threads
@threads for element in elements
    assemble_local!(element)
end
```

### 5. GPU Acceleration

```julia
using CUDA
K_gpu = CuSparseMatrixCSC(K)
u_gpu = CuArray(u0)
# Solve on GPU
```

---

## Troubleshooting

### Gmsh not found

**Error:** `gmsh: command not found`

**Solution:** Install Gmsh:
- **Ubuntu/Debian:** `sudo apt install gmsh`
- **macOS:** `brew install gmsh`
- **Windows:** Download from https://gmsh.info

### Mesh too coarse/fine

Adjust `lc` in `.geo` file:
- `lc = 0.05` → finer mesh (more elements, slower)
- `lc = 0.2` → coarser mesh (fewer elements, faster)

### Assembly takes forever

For large meshes (>100K elements):
1. Use iterative solvers (Krylov.jl)
2. Enable threading
3. Profile assembly (`@time`, `@profview`)

### Oscillations in solution

- Mesh too coarse → refine
- Time step too large → reduce or use adaptive solver
- Boundary conditions wrong → check Gmsh physical groups

---

## References

1. **Gmsh Documentation:** https://gmsh.info/doc/texinfo/gmsh.html
2. **JuliaFEM Documentation:** https://juliafem.github.io/JuliaFEM.jl
3. **DifferentialEquations.jl:** https://docs.sciml.ai/DiffEqDocs/
4. **Issue #183:** https://github.com/JuliaFEM/JuliaFEM.jl/issues/183

**Books:**
- Hughes, "The Finite Element Method" (theory)
- Zienkiewicz & Taylor, "The Finite Element Method" (comprehensive)
- Quarteroni et al., "Numerical Models for Differential Problems" (modern methods)

---

## Summary

**What we covered:**

1. ✓ Mesh generation with Gmsh (`.geo` → `.msh`)
2. ✓ Weak formulation and Galerkin method
3. ✓ FEM assembly (element → global matrices)
4. ✓ Extracting K, M, f for external solvers
5. ✓ Solving with DifferentialEquations.jl
6. ✓ Method of lines (PDE → ODE)

**Key takeaway:** JuliaFEM provides the spatial discretization machinery. You bring the physics and time integration. Perfect for academic/research flexibility.

**Next:** Try the example, modify the geometry, experiment with different BCs, explore nonlinear problems!

---

*This tutorial addresses Issue #183 and demonstrates JuliaFEM's "laboratory, not fortress" philosophy - educational, transparent, and flexible for research use.*
