# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
DOF System - Type-Level Field Specifications

Main entry point for the DOF system.

# Philosophy (November 2025)

Multi-field is fundamental! Single-field is just a special case with one key.

Field specifications are `DOFSet` types (currently implemented as NamedTuples), used purely at the type level:

```julia
# Clean syntax using abstract DOF{T,E} types:
(T = DOF{Float64, Vertex}, u = DOF{Vec{3,Float64}, Vertex})

# Using DOFSet macro (preferred):
S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}

# Used in Element type parameter:
Element{Tetrahedron{4}, Lagrange{1}, S}
```

Note: `@NamedTuple` also works (since `DOFSet = NamedTuple`), but `@DOFSet` is preferred for future compatibility.

# Module Structure
- `api.jl`: DOF{T,E} abstract type + dof_size() utility
- `fields.jl`: Field specification system (@DOFSet, ndofs, etc.)
"""

# DOF{T,E} abstract type (type-level specification only)
include("api.jl")

# Field specification system (@DOFSet macro, ndofs, etc.)
include("fields.jl")

# Note: dof_connectivity.jl is included later in JuliaFEM.jl after Element is defined
# Note: dof_handler.jl is included later in JuliaFEM.jl after Mesh is defined
