# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Geometry cache for zero-allocation assembly.

Defines the mutable `GeometryCache` consumed by every assembler. The cache
is parametric on its backing storage so the same struct works whether it
owns its arrays directly (heap-owned mode used by the element-based
assemblers) or wraps column views into batched SoA storage (DOF-based
assembler).
"""

using Tensors

"""
    GeometryCache{XT,NT,GT,WT} <: AbstractGeometryCache

Per-element geometry workspace. Parametric on the storage backing so the
*same* struct can either own its arrays directly (heap-owned
`Vector`/`Matrix`, the mode used by the element-based assemblers)
or wrap column views into batched SoA storage (the mode used by
the DOF-based assembler, which stores all elements' geometry in
contiguous arrays and hands out `view(...)`-backed `GeometryCache`s).

Both flavours present the same `cache.X[i]`, `cache.N_data[q, k]`,
`cache.∇N_data[q, k]`, `cache.detJ_w[q]` API to kernels.

# Type parameters
- `XT <: AbstractVector{<:Vec{3,<:AbstractFloat}}` — node-coordinate storage
- `NT <: AbstractMatrix{<:AbstractFloat}`          — basis function value storage
  (`[NIP × N]`); enables body forces, mass matrices, surface loads and
  the standard `N`-coupled multi-field kernels (e.g. ε-T thermo-elasticity)
- `GT <: AbstractMatrix{<:Vec{3,<:AbstractFloat}}` — physical gradient storage
  (`[NIP × N]`)
- `WT <: AbstractVector{<:AbstractFloat}`          — `detJ * weight` storage

The constraints are deliberately *loose* — any concrete element-type
combination matching the shape works. In particular both `Float64`
(the default everywhere on the CPU path) and `Float32` (the GPU-storage
mirror used by the KernelAbstractions backend on devices that can't
hold double precision, e.g. Apple GPUs) are admitted by the same struct
and the same downstream microkernel functions.

# Common parameterizations

| backing | `XT` / `NT` / `GT` / `WT` |
| --- | --- |
| heap-owned (default Float64) | `Vector{Vec{3,Float64}}`, `Matrix{Float64}`, `Matrix{Vec{3,Float64}}`, `Vector{Float64}` |
| SoA view (DOF-based, Float64) | `SubArray{Vec{3,Float64}}` ×{X,∇N} + `SubArray{Float64}` ×{N,detJ·w}, all backed by `DOFBasedCOOCache.*_batch` |
| SoA view (Float32 mirror)     | identical shapes with `Float32` element type for GPU-only devices |

The view-backed flavour stores no data of its own; it's a thin handle
constructed once per element at cache build time and reused on every
assembly.
"""
struct GeometryCache{XT<:AbstractVector{<:Vec{3,<:AbstractFloat}},
                     NT<:AbstractMatrix{<:AbstractFloat},
                     GT<:AbstractMatrix{<:Vec{3,<:AbstractFloat}},
                     WT<:AbstractVector{<:AbstractFloat}} <: AbstractGeometryCache
    X::XT                                          # Node coordinates [N]
    N_data::NT                                     # Basis values     [NIP × N]
    ∇N_data::GT                                    # Physical gradients [NIP × N]
    detJ_w::WT                                     # detJ * weight    [NIP]
end

"""
    geometry_eltype(cache::GeometryCache) -> Type{<:AbstractFloat}

Element float type used by this geometry cache (`Float64` for the
default CPU path, `Float32` for the GPU-storage mirror). Read at the
top of every precision-generic microkernel (`evaluate_entry`,
`evaluate_mass_entry`, `compute_diagonal!`, …) so the inner loops use
matching arithmetic precision and accumulate into a `zero(F)`.
"""
@inline geometry_eltype(cache::GeometryCache) = eltype(cache.detJ_w)

"""
    reset!(cache::GeometryCache)

Reset geometry cache to zero values.

# Side Effects
Mutates all arrays in cache to zero.
"""
function reset!(cache::GeometryCache)
    # `fill!` works on both `Vector`/`Matrix` and view-backed
    # `SubArray`s, so the same body covers both parameterizations.
    # The float type comes from the storage's `eltype`, so this works
    # equally for the Float64 default and any Float32 mirror.
    F = geometry_eltype(cache)
    fill!(cache.X, zero(Vec{3,F}))
    fill!(cache.N_data, zero(F))
    fill!(cache.∇N_data, zero(Vec{3,F}))
    fill!(cache.detJ_w, zero(F))
    return nothing
end

# ============================================================================
# CONSTRUCTORS
# ============================================================================

"""
    create_geometry_cache(N::Int, NIP::Int) -> GeometryCache

Create pre-allocated geometry workspace (mutable, Vector-based).

# Arguments
- `N`: Number of nodes in element
- `NIP`: Number of integration points

# Returns
- `GeometryCache` with pre-allocated Vector-based arrays
"""
function create_geometry_cache(N::Int, NIP::Int)
    X = [zero(Vec{3,Float64}) for _ in 1:N]
    N_data = zeros(Float64, NIP, N)
    ∇N_data = Matrix{Vec{3,Float64}}(undef, NIP, N)
    fill!(∇N_data, zero(Vec{3,Float64}))
    detJ_w = zeros(NIP)
    return GeometryCache(X, N_data, ∇N_data, detJ_w)
end

