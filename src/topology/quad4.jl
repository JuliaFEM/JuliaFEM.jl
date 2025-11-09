# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Quad4 <: AbstractTopology

Four-node quadrilateral element in 2D.

# Reference Element
```
     η
     ^
     |
  4  |  3
     +-----+
     |     |
     |  +  | --> ξ
     |     |
     +-----+
  1        2
```

# Node Ordering
Nodes are numbered counter-clockwise starting from (-1, -1):
1. (-1, -1) - Bottom-left
2. ( 1, -1) - Bottom-right
3. ( 1,  1) - Top-right
4. (-1,  1) - Top-left

# Properties
- Nodes: 4
- Dimension: 2
- Edges: 4
- Faces: 1 (the element itself)

# Typical Usage
```julia
julia> topology = Quad4()
julia> nnodes(topology)
4
julia> dim(topology)
2
```

See also: [`AbstractTopology`](@ref), [`Quad8`](@ref), [`Tri3`](@ref)
"""
struct Quad4 <: AbstractTopology end

nnodes(::Quad4) = 4
dim(::Quad4) = 2

function reference_coordinates(::Quad4)
    return (
        (-1.0, -1.0),  # Node 1
        (1.0, -1.0),  # Node 2
        (1.0, 1.0),  # Node 3
        (-1.0, 1.0),  # Node 4
    )
end

function edges(::Quad4)
    return (
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Right
        (3, 4),  # Edge 3: Top
        (4, 1),  # Edge 4: Left
    )
end

# For 2D elements, faces are the element itself
faces(::Quad4) = ((1, 2, 3, 4),)
