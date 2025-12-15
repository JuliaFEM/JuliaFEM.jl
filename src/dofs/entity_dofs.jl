# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Entity-Based DOF Implementations

This file provides convenient type aliases and helper functions for common
DOF patterns using the entity-based DOF system.

The core type `DOF{T, E<:TopologicalEntity}` is defined in api.jl.
This file adds convenience constructors and commonly used patterns.
"""

# ============================================================================
# Convenience Type Aliases
# ============================================================================

"""
    ScalarDOF{E<:TopologicalEntity}

Type alias for scalar DOFs at entities.

# Examples
```julia
ScalarDOF{Vertex}      # Scalar at vertices (any topology)
ScalarDOF{Edge}        # Scalar on edges (any topology)
ScalarDOF{Face}        # Scalar on faces (any topology)
```

Equivalent to `DOF{Vec{1}, E}` (scalar as 1-component vector).
"""
const ScalarDOF{E} = DOF{Vec{1}, E} where {E<:TopologicalEntity}

"""
    VectorDOF{D, E<:TopologicalEntity}

Type alias for D-dimensional vector DOFs at entities.

# Examples
```julia
VectorDOF{2, Vertex}   # 2D displacement at vertices
VectorDOF{3, Vertex}   # 3D displacement at vertices
VectorDOF{3, Edge}     # 3D field on edges
```

Equivalent to `DOF{Vec{D}, E}`.
"""
const VectorDOF{D, E} = DOF{Vec{D}, E} where {D, E<:TopologicalEntity}

"""
    TensorDOF{D, E<:TopologicalEntity}

Type alias for D×D tensor DOFs at entities.

# Examples
```julia
TensorDOF{2, Vertex}   # 2D stress/strain at vertices
TensorDOF{3, Vertex}   # 3D stress/strain at vertices
```

Equivalent to `DOF{Tensor{2,D}, E}`.
"""
const TensorDOF{D, E} = DOF{Tensor{2,D,Float64}, E} where {D, E<:TopologicalEntity}

# ============================================================================
# Common DOF Patterns
# ============================================================================

# Standard Lagrange element DOF types for common problems
const LagrangeHeat1D = DOF{Vec{1}, Vertex}
const LagrangeHeat2D = DOF{Vec{1}, Vertex}
const LagrangeHeat3D = DOF{Vec{1}, Vertex}

const LagrangeElasticity1D = DOF{Vec{1}, Vertex}
const LagrangeElasticity2D = DOF{Vec{2}, Vertex}
const LagrangeElasticity3D = DOF{Vec{3}, Vertex}

# Nédélec (H(curl)) element DOF types for electromagnetic problems
const NedelecScalar1D = DOF{Vec{1}, Edge}
const NedelecScalar2D = DOF{Vec{1}, Edge}
const NedelecScalar3D = DOF{Vec{1}, Edge}
const NedelecVector3D = DOF{Vec{3}, Edge}

# Raviart-Thomas (H(div)) element DOF types for fluid flow problems
const RaviartThomasScalar2D = DOF{Vec{1}, Face}
const RaviartThomasScalar3D = DOF{Vec{1}, Face}
const RaviartThomasVector3D = DOF{Vec{3}, Face}

# Discontinuous Galerkin (DG) element DOF types
const DGScalar = DOF{Vec{1}, Cell}
const DGVector2D = DOF{Vec{2}, Cell}
const DGVector3D = DOF{Vec{3}, Cell}

# ============================================================================
# Helper Functions
# ============================================================================

"""
    make_vertex_dof(::Type{T}) where T

Create DOF type for quantity T at vertices.

Topology context comes from Element.

# Examples
```julia
make_vertex_dof(Vec{1})    # DOF{Vec{1}, Vertex}
make_vertex_dof(Vec{3})    # DOF{Vec{3}, Vertex}
```
"""
make_vertex_dof(::Type{T}) where T = DOF{T, Vertex}

"""
    make_edge_dof(::Type{T}) where T

Create DOF type for quantity T on edges.

Topology context comes from Element.

# Examples
```julia
make_edge_dof(Vec{1})      # DOF{Vec{1}, Edge}
make_edge_dof(Vec{3})      # DOF{Vec{3}, Edge}
```
"""
make_edge_dof(::Type{T}) where T = DOF{T, Edge}

"""
    make_face_dof(::Type{T}) where T

Create DOF type for quantity T on faces.

Topology context comes from Element.

# Examples
```julia
make_face_dof(Vec{1})      # DOF{Vec{1}, Face}
make_face_dof(Vec{3})      # DOF{Vec{3}, Face}
```
"""
make_face_dof(::Type{T}) where T = DOF{T, Face}

"""
    make_cell_dof(::Type{T}) where T

Create DOF type for quantity T in cell interior.

Topology context comes from Element.

# Examples
```julia
make_cell_dof(Vec{1})      # DOF{Vec{1}, Cell}
make_cell_dof(Vec{3})      # DOF{Vec{3}, Cell}
```
"""
make_cell_dof(::Type{T}) where T = DOF{T, Cell}

# ============================================================================
# Exports
# ============================================================================

export ScalarDOF, VectorDOF, TensorDOF,
       LagrangeHeat1D, LagrangeHeat2D, LagrangeHeat3D,
       LagrangeElasticity1D, LagrangeElasticity2D, LagrangeElasticity3D,
       NedelecScalar1D, NedelecScalar2D, NedelecScalar3D, NedelecVector3D,
       RaviartThomasScalar2D, RaviartThomasScalar3D, RaviartThomasVector3D,
       DGScalar, DGVector2D, DGVector3D,
       make_vertex_dof, make_edge_dof, make_face_dof, make_cell_dof
