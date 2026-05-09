# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    Segment{N} <: AbstractTopology{N}

1D segment (line) topology with N nodes.

# Node Count Variants
- `Segment{2}` (alias `Seg2`): Linear segment (P1)
- `Segment{3}` (alias `Seg3`): Quadratic segment (P2, includes midpoint)
"""
struct Segment{N} <: AbstractTopology{N} end

const Seg2 = Segment{2}
const Seg3 = Segment{3}

dim(::Segment{N}) where {N} = 1

function reference_coordinates(::Segment{2})
    return SVector(Vec{1,Float64}.((
        (-1.0,),
        (1.0,)
    )))
end

function reference_coordinates(::Segment{3})
    return SVector(Vec{1,Float64}.((
        (-1.0,),
        (1.0,),
        (0.0,)
    )))
end

function edges(::T) where {T<:Segment}
    return SVector(Edge((1, 2)))
end

function faces(::T) where {T<:Segment}
    return SVector(Vertex(), Vertex())  # Endpoints are "faces" in 1D
end

function vertices(::T) where {T<:Segment}
    return SVector(Vertex(), Vertex())
end

function cells(::T) where {T<:Segment}
    return SVector(Cell())
end

nvertices(::Segment) = 2
nedges(::Segment) = 1
nfaces(::Segment) = 2  # Two endpoints

