# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Hexahedron{N} <: AbstractTopology{N}

Hexahedral (brick) topology with N nodes.

# Node Count Variants
- `Hexahedron{8}` (alias `Hex8`): Linear hexahedron (P1 Lagrange)
- `Hexahedron{20}` (alias `Hex20`): Quadratic serendipity hexahedron (P2, no center nodes)
- `Hexahedron{27}` (alias `Hex27`): Quadratic full hexahedron (P2, includes center nodes)
"""
struct Hexahedron{N} <: AbstractTopology{N} end

const Hex8 = Hexahedron{8}
const Hex20 = Hexahedron{20}
const Hex27 = Hexahedron{27}

nnodes(::Hexahedron{N}) where {N} = N
dim(::Hexahedron{N}) where {N} = 3

function reference_coordinates(::Hexahedron{8})
    return (
        (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)
    )
end

function edges(::Hexahedron{N}) where {N}
    return (
        (1, 2), (2, 3), (3, 4), (4, 1),
        (5, 6), (6, 7), (7, 8), (8, 5),
        (1, 5), (2, 6), (3, 7), (4, 8)
    )
end

function faces(::Hexahedron{N}) where {N}
    return (
        (1, 4, 3, 2), (5, 6, 7, 8),
        (1, 2, 6, 5), (2, 3, 7, 6),
        (3, 4, 8, 7), (4, 1, 5, 8)
    )
end

export Hex8, Hex20, Hex27
