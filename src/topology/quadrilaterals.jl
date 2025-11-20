# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
    Quadrilateral{N} <: AbstractTopology{N}

Quadrilateral topology with N nodes.

# Node Count Variants
- `Quadrilateral{4}` (alias `Quad4`): Bilinear quadrilateral (Q1)
- `Quadrilateral{8}` (alias `Quad8`): Quadratic serendipity (Q2, no center node)
- `Quadrilateral{9}` (alias `Quad9`): Quadratic full (Q2, includes center node)
"""
struct Quadrilateral{N} <: AbstractTopology{N} end

const Quad4 = Quadrilateral{4}
const Quad8 = Quadrilateral{8}
const Quad9 = Quadrilateral{9}

nnodes(::Quadrilateral{N}) where {N} = N
dim(::Quadrilateral{N}) where {N} = 2

function reference_coordinates(::Quadrilateral{4})
    return SVector(
        Vec{2,Float64}((-1.0, -1.0)),
        Vec{2,Float64}((1.0, -1.0)),
        Vec{2,Float64}((1.0, 1.0)),
        Vec{2,Float64}((-1.0, 1.0))
    )
end

function reference_coordinates(::Quadrilateral{8})
    return SVector(
        Vec{2,Float64}((-1.0, -1.0)),  # N1: Corner
        Vec{2,Float64}((1.0, -1.0)),   # N2: Corner
        Vec{2,Float64}((1.0, 1.0)),    # N3: Corner
        Vec{2,Float64}((-1.0, 1.0)),   # N4: Corner
        Vec{2,Float64}((0.0, -1.0)),   # N5: Edge midpoint
        Vec{2,Float64}((1.0, 0.0)),    # N6: Edge midpoint
        Vec{2,Float64}((0.0, 1.0)),    # N7: Edge midpoint
        Vec{2,Float64}((-1.0, 0.0))    # N8: Edge midpoint
    )
end

function reference_coordinates(::Quadrilateral{9})
    return SVector(
        Vec{2,Float64}((-1.0, -1.0)),  # N1: Corner
        Vec{2,Float64}((1.0, -1.0)),   # N2: Corner
        Vec{2,Float64}((1.0, 1.0)),    # N3: Corner
        Vec{2,Float64}((-1.0, 1.0)),   # N4: Corner
        Vec{2,Float64}((0.0, -1.0)),   # N5: Edge midpoint
        Vec{2,Float64}((1.0, 0.0)),    # N6: Edge midpoint
        Vec{2,Float64}((0.0, 1.0)),    # N7: Edge midpoint
        Vec{2,Float64}((-1.0, 0.0)),   # N8: Edge midpoint
        Vec{2,Float64}((0.0, 0.0))     # N9: Center node
    )
end

function edges(::Quadrilateral{N}) where {N}
    return ((1, 2), (2, 3), (3, 4), (4, 1))
end

function faces(::Quadrilateral{N}) where {N}
    return ((1, 2, 3, 4),)
end

export Quad4, Quad8, Quad9
