# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Triangle{N} <: AbstractTopology{N}

Triangular topology with N nodes.

# Node Count Variants
- `Triangle{3}` (alias `Tri3`): Linear triangle (P1 Lagrange)
- `Triangle{6}` (alias `Tri6`): Quadratic triangle (P2 Lagrange)
- `Triangle{7}` (alias `Tri7`): Quadratic triangle with centroid (P2)
- `Triangle{10}` (alias `Tri10`): Cubic triangle (P3 Lagrange)
"""
struct Triangle{N} <: AbstractTopology{N} end

const Tri3 = Triangle{3}
const Tri6 = Triangle{6}
const Tri7 = Triangle{7}
const Tri10 = Triangle{10}

nnodes(::Triangle{N}) where {N} = N
dim(::Triangle{N}) where {N} = 2

function reference_coordinates(::Triangle{3})
    return SVector(Vec{2,Float64}.((
        (0.0, 0.0),  # N1: Corner at origin
        (1.0, 0.0),  # N2: Corner along ξ
        (0.0, 1.0)   # N3: Corner along η
    )))
end

function reference_coordinates(::Triangle{6})
    return SVector(Vec{2,Float64}.((
        (0.0, 0.0),  # N1: Corner
        (1.0, 0.0),  # N2: Corner
        (0.0, 1.0),  # N3: Corner
        (0.5, 0.0),  # N4: Midpoint of edge 1-2
        (0.5, 0.5),  # N5: Midpoint of edge 2-3
        (0.0, 0.5)   # N6: Midpoint of edge 3-1
    )))
end

function reference_coordinates(::Triangle{7})
    return SVector(Vec{2,Float64}.((
        (0.0, 0.0),          # N1: Corner
        (1.0, 0.0),          # N2: Corner
        (0.0, 1.0),          # N3: Corner
        (0.5, 0.0),          # N4: Midpoint edge 1-2
        (0.5, 0.5),          # N5: Midpoint edge 2-3
        (0.0, 0.5),          # N6: Midpoint edge 3-1
        (1.0 / 3.0, 1.0 / 3.0)   # N7: Face centroid
    )))
end

function reference_coordinates(::Triangle{10})
    return SVector(Vec{2,Float64}.((
        (0.0, 0.0),          # N1: Corner
        (1.0, 0.0),          # N2: Corner
        (0.0, 1.0),          # N3: Corner
        (1.0 / 3.0, 0.0),      # N4: Edge 1-2, 1/3
        (2.0 / 3.0, 0.0),      # N5: Edge 1-2, 2/3
        (2.0 / 3.0, 1.0 / 3.0),  # N6: Edge 2-3, 1/3
        (1.0 / 3.0, 2.0 / 3.0),  # N7: Edge 2-3, 2/3
        (0.0, 2.0 / 3.0),      # N8: Edge 3-1, 1/3
        (0.0, 1.0 / 3.0),      # N9: Edge 3-1, 2/3
        (1.0 / 3.0, 1.0 / 3.0)   # N10: Face centroid
    )))
end

function edges(::T) where {T<:Triangle}
    return SVector(Edge.((
        (1, 2),  # Edge 1: Bottom
        (2, 3),  # Edge 2: Diagonal
        (3, 1)   # Edge 3: Left
    )))
end

function faces(::T) where {T<:Triangle}
    return SVector(Face((1, 2, 3)))
end

function vertices(::T) where {T<:Triangle}
    return SVector(Vertex(), Vertex(), Vertex())
end

function cells(::T) where {T<:Triangle}
    return SVector(Cell())
end

nvertices(::Triangle) = 3
nedges(::Triangle) = 3
nfaces(::Triangle) = 1

export Tri3, Tri6, Tri7, Tri10
