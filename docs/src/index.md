# JuliaFEM.jl

JuliaFEM.jl is an open-source finite element method (FEM) framework written in Julia, designed for reliable, scalable, and distributed finite element analysis.

## Overview

JuliaFEM.jl provides a modern, type-safe API for finite element analysis with support for:

- **Multiple element types**: Triangles, quadrilaterals, tetrahedra, hexahedra, and more
- **Various physics**: Linear and nonlinear elasticity, heat transfer, and more
- **GPU acceleration**: CUDA support for high-performance computing
- **Distributed computing**: Multi-GPU and MPI support for large-scale problems

## Key Features

- **Type-safe element system**: Elements are parameterized by topology, basis functions, and DOF specifications
- **Modern physics API**: Clean separation between mesh, material, field, and formulation
- **Flexible material models**: Linear elastic, Neo-Hookean, perfect plasticity, and more
- **Efficient assembly**: COO and CSC sparse matrix formats with GPU support

## Installation

Install JuliaFEM.jl using Julia's package manager:

```julia
using Pkg
Pkg.add("JuliaFEM")
```

## Quick Start

```julia
using JuliaFEM

# Create a mesh
mesh = create_unit_cube_mesh(Hex8, 10, 10, 10)

# Define a physics problem
physics = Physics(
    name = "elasticity",
    mesh = mesh,
    element_set = :all,
    field = Displacement{3}(),
    formulation = ContinuumFormulation{FullThreeD}(),
    material = LinearElastic(; E=200e9, ν=0.3)
)

# Assemble and solve
assemble!(physics)
solve!(physics)
```

## Documentation

- [API Reference](@ref) - Complete API documentation

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/JuliaFEM/JuliaFEM.jl) for guidelines.

## License

JuliaFEM.jl is licensed under the MIT License.
