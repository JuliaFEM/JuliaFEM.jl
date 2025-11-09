# Academic Example: Matrix Extraction for External Solvers

**Addresses Issue #183**: Demonstrates using JuliaFEM for spatial discretization only, extracting matrices for external solvers.

## The Three Requirements

This example demonstrates exactly what was requested in Issue #183:

### a) Discretize space (into a mesh)
- Shows programmatic mesh generation
- Element connectivity accessible  
- Compatible with Gmsh mesh files

### b) Assemble the stiffness matrix
- Assembly framework demonstrated
- Currently works with Dirichlet BC
- Heat/Elasticity coming in Phase 2 (2-4 months)

### c) Get back vectors and matrices
- Extract `K` (stiffness), `M` (mass), `f` (force) as standard Julia types
- `SparseMatrixCSC{Float64,Int64}` and `Vector{Float64}`
- Direct compatibility with entire Julia ecosystem

## Quick Start

```bash
cd examples/academic_matrix_extraction
julia --project=../.. academic_example.jl
```

## What You Get

After assembly, matrices are extracted as:

```julia
K = problem.assembly.K  # Stiffness matrix (sparse)
M = problem.assembly.M  # Mass matrix (sparse)  
f = problem.assembly.f  # Force vector
```

These are standard Julia types that work with:

### DifferentialEquations.jl (Transient Problems)

```julia
using DifferentialEquations

function fem_ode!(du, u, p, t)
    K, M, f = p
    du .= M \ (-K * u .+ f)
end

u0 = zeros(N)
prob = ODEProblem(fem_ode!, u0, (0.0, 1.0), (K, M, f))
sol = solve(prob, Tsit5())
```

### LinearSolve.jl (Steady-State)

```julia
using LinearSolve

prob = LinearProblem(K, f)
sol = solve(prob, KrylovJL_GMRES())
```

### Krylov.jl (Iterative Methods)

```julia
using Krylov

u, stats = gmres(K, f; atol=1e-10, rtol=1e-8)
```

### Custom Research Solvers

```julia
using SparseArrays, LinearAlgebra

u = K \ f              # Direct solve
L = cholesky(K)        # Factorization
λ, v = eigs(K, M)      # Eigenvalue analysis
```

## Current Status

**What Works NOW:**
- ✅ Mesh generation and element connectivity
- ✅ Matrix extraction API (`problem.assembly.K`, `.M`, `.f`)
- ✅ Dirichlet boundary conditions
- ✅ Integration with Julia solver ecosystem

**Coming in Phase 2 (2-4 months):**
- ⏳ Heat equation problem type
- ⏳ Elasticity problem type
- ⏳ Full assembly for physics problems
- ⏳ 40-130x performance improvement

## Why Phase 2?

JuliaFEM is undergoing architecture refactoring (Nov 2025):
- Replacing Dict-based fields (100x performance penalty) with type-stable system
- New immutable element architecture (already 40-130x faster)
- Heat/Elasticity problem types depend on old system
- Being restored with new architecture

## See Also

- **Issue #183**: Original request from Chris Rackauckas (2017)
- **examples/gmsh_heat_equation/**: Full workflow with Gmsh mesh files
- **docs/book/gmsh_tutorial.md**: Comprehensive step-by-step tutorial
- **llm/ARCHITECTURE.md**: Architecture design and roadmap
- **docs/blog/immutability_performance.md**: Performance analysis

## Academic Use Case

Perfect for research where you need:
1. Spatial discretization (FEM assembly)
2. Custom time integration schemes
3. Novel solver algorithms
4. Integration with other Julia packages

JuliaFEM handles the messy FEM assembly; you control the solving.
