# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Pyramid{N} <: AbstractTopology{N}

Pyramidal topology with N nodes.

# Node Count Variants
- `Pyramid{5}` (alias `Pyr5`): Linear pyramid (P1)
- `Pyramid{13}` (alias `Pyr13`): Quadratic pyramid (P2)
"""
struct Pyramid{N} <: AbstractTopology{N} end

const Pyr5 = Pyramid{5}

nnodes(::Pyramid{N}) where {N} = N
dim(::Pyramid{N}) where {N} = 3

function reference_coordinates(::Pyramid{5})
    return (
        (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0)
    )
end

function edges(::Pyramid{N}) where {N}
    return (
        (1, 2), (2, 3), (3, 4), (4, 1),  # Base edges
        (1, 5), (2, 5), (3, 5), (4, 5)   # Edges to apex
    )
end

function faces(::Pyramid{N}) where {N}
    return (
        (1, 4, 3, 2),  # Quad base
        (1, 2, 5),     # Triangle face 1
        (2, 3, 5),     # Triangle face 2
        (3, 4, 5),     # Triangle face 3
        (4, 1, 5)      # Triangle face 4
    )
end

export Pyr5
