# Gmsh Heat Equation Example

Complete workflow demonstration addressing [Issue #183](https://github.com/JuliaFEM/JuliaFEM.jl/issues/183).

## Quick Links

- **Working Example:** `gmsh_heat_equation.jl`
- **Gmsh Geometry:** `unit_square.geo`
- **Comprehensive Tutorial:** `../../docs/book/gmsh_tutorial.md`

## What's Here

This example shows:

1. Mesh generation with Gmsh
2. Loading mesh into JuliaFEM  
3. FEM assembly (K and M matrices)
4. Extracting matrices for external solvers
5. Integration with DifferentialEquations.jl

## Academic Usage (Issue #183)

If you want JuliaFEM for **discretization only** (not built-in physics):

```julia
# After assembly:
K = problem.assembly.K  # Stiffness matrix
M = problem.assembly.M  # Mass matrix
f = problem.assembly.f  # Force vector

# ODE system: M * du/dt = -K * u + f
# Now use with your own solver!
```

See tutorial for complete explanation.
