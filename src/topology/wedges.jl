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
    return SVector(Vec{3,Float64}.((
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0)
    )))
end

function reference_coordinates(::Wedge{15})
    return SVector(Vec{3,Float64}.((
        # 6 corner nodes
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        # 9 edge midpoint nodes
        (0.5, 0.0, -1.0),  # Bottom triangle edges
        (0.5, 0.5, -1.0),
        (0.0, 0.5, -1.0),
        (0.5, 0.0, 1.0),   # Top triangle edges
        (0.5, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, 0.0, 0.0),   # Vertical edges
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0)
    )))
end

function edges(::T) where {T<:Wedge}
    return SVector(Edge{T}.((
        (1, 2), (2, 3), (3, 1),  # Bottom triangle
        (4, 5), (5, 6), (6, 4),  # Top triangle
        (1, 4), (2, 5), (3, 6)   # Vertical edges
    )))
end

function faces(::T) where {T<:Wedge}
    return SVector(Face{T}.((
        (1, 3, 2),       # Bottom triangle
        (4, 5, 6),       # Top triangle
        (1, 2, 5, 4),    # Quad face 1
        (2, 3, 6, 5),    # Quad face 2
        (3, 1, 4, 6)     # Quad face 3
    )))
end

function vertices(::T) where {T<:Wedge}
    return SVector(
        Vertex{T}(), Vertex{T}(), Vertex{T}(),
        Vertex{T}(), Vertex{T}(), Vertex{T}()
    )
end

function cells(::T) where {T<:Wedge}
    return SVector(Cell{T}())
end

nvertices(::Wedge) = 6
nedges(::Wedge) = 9
nfaces(::Wedge) = 5

export Wedge6, Wedge15
