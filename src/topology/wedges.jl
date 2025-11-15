# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Wedge{N} <: AbstractTopology{N}

Wedge (prism) topology with N nodes.

# Node Count Variants
- `Wedge{6}` (alias `Wedge6`): Linear wedge (P1)
- `Wedge{15}` (alias `Wedge15`): Quadratic wedge (P2)
"""
struct Wedge{N} <: AbstractTopology{N} end

const Wedge6 = Wedge{6}
const Wedge15 = Wedge{15}

nnodes(::Wedge{N}) where {N} = N
dim(::Wedge{N}) where {N} = 3

function reference_coordinates(::Wedge{6})
    return (
        (0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 1.0, -1.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)
    )
end

function edges(::Wedge{N}) where {N}
    return (
        (1, 2), (2, 3), (3, 1),  # Bottom triangle
        (4, 5), (5, 6), (6, 4),  # Top triangle
        (1, 4), (2, 5), (3, 6)   # Vertical edges
    )
end

function faces(::Wedge{N}) where {N}
    return (
        (1, 3, 2),       # Bottom triangle
        (4, 5, 6),       # Top triangle
        (1, 2, 5, 4),    # Quad face 1
        (2, 3, 6, 5),    # Quad face 2
        (3, 1, 4, 6)     # Quad face 3
    )
end

export Wedge6, Wedge15
