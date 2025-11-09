# Heat Equation Example: From Gmsh to Physics
# Addresses Issue #183: Academic usage without built-in physics

This example demonstrates the complete workflow:
1. Generate mesh using Gmsh
2. Load mesh into JuliaFEM
3. Assemble stiffness matrix and mass matrix
4. Extract matrices for external solvers (e.g., DifferentialEquations.jl)
5. Solve the heat equation

## Problem Statement

Solve the transient heat equation on a unit square:

```
∂u/∂t = α∇²u + f(x,y,t)
```

with boundary conditions:
- u = 0 on left edge (Dirichlet)
- ∂u/∂n = 0 on other edges (Neumann, natural BC)

Initial condition: u(x,y,0) = sin(πx)sin(πy)

## Quick Start

### 1. Generate mesh

```bash
gmsh -2 unit_square.geo -o unit_square.msh
```

This creates a triangular mesh of the unit square.

### 2. Run the example

```bash
julia --project gmsh_heat_equation.jl
```

## What You Get

The example shows how to:
- Load Gmsh mesh files
- Create FEM elements with material properties
- Assemble global stiffness matrix K and mass matrix M
- Apply Dirichlet boundary conditions
- Extract the resulting ODE system: M du/dt = -K u + f
- Solve using your own time integrator

## For Academic Users (Issue #183)

If you want to use JuliaFEM just for discretization (not the built-in physics):

```julia
# After assembly, extract the matrices:
K = problem.assembly.K  # Stiffness matrix (SparseMatrixCSC)
M = problem.assembly.M  # Mass matrix (SparseMatrixCSC)
f = problem.assembly.f  # Force vector

# Now use these with DifferentialEquations.jl, Krylov.jl, etc.
# The ODE system is: M * du/dt = -K * u + f
```

## Files

- `unit_square.geo` - Gmsh geometry definition
- `gmsh_heat_equation.jl` - Complete working example
- `README.md` - This file

## See Also

- Tutorial: `docs/book/gmsh_tutorial.md` (comprehensive step-by-step)
- Issue #183: https://github.com/JuliaFEM/JuliaFEM.jl/issues/183
