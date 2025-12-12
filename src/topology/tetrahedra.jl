# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Tetrahedron{N} <: AbstractTopology{N}

Tetrahedral topology with N nodes.

# Node Count Variants
- `Tetrahedron{4}` (alias `Tet4`): Linear tetrahedron (P1 Lagrange)
- `Tetrahedron{10}` (alias `Tet10`): Quadratic tetrahedron (P2 Lagrange)
"""
struct Tetrahedron{N} <: AbstractTopology{N} end

const Tet4 = Tetrahedron{4}
const Tet10 = Tetrahedron{10}

nnodes(::Tetrahedron{N}) where {N} = N
dim(::Tetrahedron{N}) where {N} = 3

function reference_coordinates(::Tetrahedron{4})
    return SVector(Vec{3,Float64}.((
        (0.0, 0.0, 0.0),  # N1: Corner at origin
        (1.0, 0.0, 0.0),  # N2: Corner along ξ
        (0.0, 1.0, 0.0),  # N3: Corner along η
        (0.0, 0.0, 1.0)   # N4: Corner along ζ
    )))
end

function reference_coordinates(::Tetrahedron{10})
    return SVector(Vec{3,Float64}.((
        (0.0, 0.0, 0.0),  # N1: Corner
        (1.0, 0.0, 0.0),  # N2: Corner
        (0.0, 1.0, 0.0),  # N3: Corner
        (0.0, 0.0, 1.0),  # N4: Corner
        (0.5, 0.0, 0.0),  # N5: Midpoint edge 1-2
        (0.5, 0.5, 0.0),  # N6: Midpoint edge 2-3
        (0.0, 0.5, 0.0),  # N7: Midpoint edge 3-1
        (0.0, 0.0, 0.5),  # N8: Midpoint edge 1-4
        (0.5, 0.0, 0.5),  # N9: Midpoint edge 2-4
        (0.0, 0.5, 0.5)   # N10: Midpoint edge 3-4
    )))
end

function edges(::T) where {T<:Tetrahedron}
    return SVector(Edge.((
        (1, 2),
        (2, 3),
        (3, 1),
        (1, 4),
        (2, 4),
        (3, 4)
    )))
end

function faces(::T) where {T<:Tetrahedron}
    return SVector(Face.((
        (1, 3, 2),
        (1, 2, 4),
        (2, 3, 4),
        (3, 1, 4)
    )))
end

function vertices(::T) where {T<:Tetrahedron}
    return SVector(Vertex(), Vertex(), Vertex(), Vertex())
end

function cells(::T) where {T<:Tetrahedron}
    return SVector(Cell())
end

nvertices(::Tetrahedron) = 4
nedges(::Tetrahedron) = 6
nfaces(::Tetrahedron) = 4

export Tet4, Tet10
