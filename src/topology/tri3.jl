# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Tri3 <: AbstractTopology

Three-node triangular element in 2D.

# Reference Element
```
     η
     ^
     |
  (0,1)
     |  \\
     |    \\
     |      \\
     +---------> ξ
  (0,0)    (1,0)
```

# Node Ordering
Nodes are numbered counter-clockwise starting from origin:
1. (0, 0) - Origin
2. (1, 0) - Along ξ-axis
3. (0, 1) - Along η-axis

# Properties
- Nodes: 3
- Dimension: 2
- Edges: 3
- Faces: 1 (the element itself)

# Typical Usage
```julia
julia> topology = Tri3()
julia> nnodes(topology)
3
julia> dim(topology)
2
```

See also: [`AbstractTopology`](@ref), [`Tri6`](@ref), [`Quad4`](@ref)
"""
struct Tri3 <: AbstractTopology end

nnodes(::Tri3) = 3
dim(::Tri3) = 2

function reference_coordinates(::Tri3)
    return (
        (0.0, 0.0),  # Node 1
        (1.0, 0.0),  # Node 2
        (0.0, 1.0),  # Node 3
    )
end

function edges(::Tri3)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right
        (3, 1),  # Edge 3: Left
    )
end

# For 2D elements, faces are the element itself
faces(::Tri3) = ((1, 2, 3),)
