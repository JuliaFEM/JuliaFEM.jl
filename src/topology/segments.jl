# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

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

nnodes(::Segment{N}) where {N} = N
dim(::Segment{N}) where {N} = 1

function reference_coordinates(::Segment{2})
    return ((-1.0,), (1.0,))
end

function reference_coordinates(::Segment{3})
    return ((-1.0,), (1.0,), (0.0,))
end

function edges(::Segment{N}) where {N}
    return ((1, 2),)
end

function faces(::Segment{N}) where {N}
    return ((1,), (2,))  # Endpoints are "faces" in 1D
end

export Seg2, Seg3
