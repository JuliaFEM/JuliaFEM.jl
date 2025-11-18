# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    create_circular_plate_mesh(::Type{Tri3};
                               radius=1.0,
                               nr=4,
                               nθ=24) -> Mesh{Tri3}

Generate a polar triangular mesh for a circular mid-surface.

The mesh consists of concentric rings with constant angular resolution.
The innermost ring is connected to the center node, producing a
fan-like topology that is well-suited for Kirchhoff plate elements such
as DKT.

# Keyword Arguments
- `radius`: Plate radius (default `1.0`).
- `nr`: Number of radial divisions (minimum 1).
- `nθ`: Number of angular sectors per ring (minimum 3).

# Node Sets
- `:all` - All nodes.
- `:center` - The plate center (single node).
- `:outer` - Nodes on the outer rim (r = radius).

# Element Sets
- `:all` - All Tri3 elements.

# Example
```julia
mesh = create_circular_plate_mesh(Tri3; radius=0.5, nr=5, nθ=48)
```
"""
function create_circular_plate_mesh(::Type{Tri3};
    radius::Float64=1.0,
    nr::Int=4,
    nθ::Int=24)

    @assert radius > 0 "radius must be positive"
    @assert nr ≥ 1 "nr (radial divisions) must be ≥ 1"
    @assert nθ ≥ 3 "nθ (angular divisions) must be ≥ 3"

    nodes = Vec{3,Float64}[]
    push!(nodes, Vec(0.0, 0.0, 0.0))  # center node

    rings = Vector{Vector{Int}}(undef, nr)
    for ir in 1:nr
        ring = Vector{Int}(undef, nθ)
        r = radius * ir / nr
        for it in 1:nθ
            θ = 2π * (it - 1) / nθ
            push!(nodes, Vec(r * cos(θ), r * sin(θ), 0.0))
            ring[it] = length(nodes)
        end
        rings[ir] = ring
    end

    connectivity = NTuple{3,UInt32}[]
    if nr ≥ 1
        first_ring = rings[1]
        for k in 1:nθ
            k_next = k == nθ ? 1 : k + 1
            push!(connectivity,
                (UInt32(1), UInt32(first_ring[k]), UInt32(first_ring[k_next])))
        end
    end

    for ir in 2:nr
        inner = rings[ir-1]
        outer = rings[ir]
        for k in 1:nθ
            k_next = k == nθ ? 1 : k + 1
            push!(connectivity,
                (UInt32(inner[k]), UInt32(outer[k]), UInt32(inner[k_next])))
            push!(connectivity,
                (UInt32(outer[k]), UInt32(outer[k_next]), UInt32(inner[k_next])))
        end
    end

    element_sets = Dict{Symbol,Set{UInt32}}(
        :all => Set(UInt32(1):UInt32(length(connectivity)))
    )

    node_sets = Dict{Symbol,Set{UInt32}}()
    node_sets[:all] = Set(UInt32(1):UInt32(length(nodes)))
    node_sets[:center] = Set([UInt32(1)])
    node_sets[:outer] = nr == 0 ? Set([UInt32(1)]) : Set(UInt32.(rings[end]))

    return Mesh{Tri3}(nodes, connectivity, element_sets, node_sets)
end
