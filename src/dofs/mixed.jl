# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Mixed DOF Types - Entity-Based

Implementation of mixed/coupled DOF systems where multiple DOF types
coexist on the same element.

# Core Type

- `MixedDOF{DOFs}`: Composition of multiple entity-based DOF types

# Use Cases

**Stokes (velocity + pressure):**
```julia
MixedDOF{(
    DOF{Vec{2}, Vertex{Triangle{3}}},    # Velocity at vertices
    DOF{Float64, Vertex{Triangle{3}}}    # Pressure at vertices
)}
```

**3-Field Formulation (u + p + J):**
```julia
MixedDOF{(
    DOF{Vec{3}, Vertex{Tetrahedron{4}}},    # Displacement at vertices
    DOF{Float64, Vertex{Tetrahedron{4}}},   # Pressure at vertices
    DOF{Float64, Edge{Tetrahedron{4}}}      # Volumetric DOF on edges
)}
```

**Mixed Interpolation (Taylor-Hood):**
```julia
MixedDOF{(
    DOF{Vec{2}, Vertex{Triangle{6}}},    # Quadratic velocity
    DOF{Float64, Vertex{Triangle{3}}}    # Linear pressure (different topology!)
)}
```

# DOF Ordering

Mixed DOFs are concatenated in order:
```julia
MixedDOF{(DOF{Vec{2}, Vertex{Triangle{3}}}, DOF{Float64, Vertex{Triangle{3}}})}
# For Triangle{3}:
# Field 1: [u1x, u1y, u2x, u2y, u3x, u3y]  (6 DOFs from 2×3 vertices)
# Field 2: [p1, p2, p3]                     (3 DOFs from 1×3 vertices)
# Total:   [u1x, u1y, u2x, u2y, u3x, u3y, p1, p2, p3]  (9 DOFs)
```

"""

# MixedDOF is already defined in api.jl, this file adds helper functions

# ============================================================================
# Helper Functions
# ============================================================================

"""
Check if DOF type is mixed.
"""
is_mixed(::Type{<:MixedDOF}) = true
is_mixed(::Type{<:DOF}) = false

"""
Get the tuple of constituent DOF types.
"""
function get_dof_types(::Type{MixedDOF{DOFs}}) where {DOFs}
    return DOFs.parameters
end

"""
Get number of fields in mixed system.
"""
function num_fields(::Type{MixedDOF{DOFs}}) where {DOFs}
    return length(DOFs.parameters)
end

"""
Get DOF type for specific field (1-indexed).
"""
function get_field_dof_type(::Type{MixedDOF{DOFs}}, field_idx::Int) where {DOFs}
    return DOFs.parameters[field_idx]
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, ::Type{MixedDOF{DOFs}}) where {DOFs}
    dof_types = [string(T) for T in DOFs.parameters]
    print(io, "MixedDOF{(", join(dof_types, ", "), ")}")
end

