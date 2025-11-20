# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Geometry cache implementations for zero-allocation assembly.

Contains mutable (GeometryCache) and immutable (ImmutableGeometryCache) variants.
"""

using Tensors

"""
    GeometryCache

Workspace for element geometry (coordinates, gradients, Jacobians, weights).

Contains pre-allocated arrays that are mutated per element during assembly.

# Fields
- `X::Vector{Vec{3,Float64}}`: Node coordinates [N]
- `∇N_data::Matrix{Vec{3,Float64}}`: Physical gradients [NIP × N]
- `detJ_w::Vector{Float64}`: detJ * weight [NIP]

# Zero-Allocation Usage
Arrays are mutated in-place during `prepare_element!` - no heap allocation.

# Design Note
Type parameters removed to avoid 80 bytes allocation in parametric function signatures.
Sizes N and NIP can be queried: `N = length(cache.X)`, `NIP = length(cache.detJ_w)`.
Uses Matrix{Vec} instead of Vector{Vector{Vec}} for better memory layout.
"""
struct GeometryCache <: AbstractGeometryCache
    X::Vector{Vec{3,Float64}}                      # Node coordinates [N]
    ∇N_data::Matrix{Vec{3,Float64}}                # Physical gradients [NIP × N]
    detJ_w::Vector{Float64}                        # detJ * weight [NIP]
end

"""
    ImmutableGeometryCache{N,NIP}

Immutable geometry cache using NTuple for zero-allocation access.

Unlike `GeometryCache`, this version:
- Uses `NTuple` instead of `Vector` (stack-allocated, no heap access)
- Is immutable (must create new instance per element)
- Has **zero allocations** during cache access
- Enables full compiler optimization (sizes known at compile time)

# Type Parameters
- `N`: Number of nodes per element (compile-time constant)
- `NIP`: Number of integration points (compile-time constant)

# Fields
- `X::NTuple{N, Vec{3,Float64}}`: Node coordinates [N]
- `∇N_data::NTuple{NIP, NTuple{N, Vec{3,Float64}}}`: Physical gradients [NIP][N]
- `detJ_w::NTuple{NIP, Float64}`: detJ * weight [NIP]

# Zero-Allocation Access

```julia
# Indexing is zero-allocation:
grad_k = cache.∇N_data[q][k]  # 0 bytes!
weight = cache.detJ_w[q]       # 0 bytes!
```

# Performance Tradeoff

**Pros:**
- Zero allocations during access (vs ~10KB per element for GeometryCache)
- Full compile-time optimization
- Stack-allocated (no GC pressure)

**Cons:**
- Immutable (must create new instance per element)
- Slightly larger code size (tuples unroll in codegen)
- Creation cost moved from update to construction

# Usage

```julia
# Create new cache per element (replaces update! pattern):
geometry_cache = create_geometry_cache(
    ImmutableGeometryCache,
    element_cache, kernel, elem_id, mesh
)

# Then use normally in compute_block!:
K_kl = compute_block!(geometry_cache, material_cache, k, l)
```
"""
struct ImmutableGeometryCache{N,NIP} <: AbstractGeometryCache
    X::NTuple{N,Vec{3,Float64}}
    ∇N_data::NTuple{NIP,NTuple{N,Vec{3,Float64}}}
    detJ_w::NTuple{NIP,Float64}
end

"""
    reset!(cache::GeometryCache)

Reset geometry cache to zero values.

# Side Effects
Mutates all arrays in cache to zero.
"""
function reset!(cache::GeometryCache)
    fill!(cache.X, zero(Vec{3,Float64}))
    fill!(cache.∇N_data, zero(Vec{3,Float64}))
    fill!(cache.detJ_w, 0.0)
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
    ∇N_data = Matrix{Vec{3,Float64}}(undef, NIP, N)
    fill!(∇N_data, zero(Vec{3,Float64}))
    detJ_w = zeros(NIP)
    return GeometryCache(X, ∇N_data, detJ_w)
end

"""
    create_geometry_cache(
        ::Type{ImmutableGeometryCache},
        element_cache::ElementCache,
        kernel::AbstractKernel,
        elem_id::Int,
        mesh::AbstractMesh
    ) -> ImmutableGeometryCache{N,NIP}

Create immutable geometry cache with computed values (zero-allocation constructor).

Unlike mutable `GeometryCache`, this computes all geometry data immediately
and returns an immutable, stack-allocated cache.

# Arguments
- `ImmutableGeometryCache`: Type parameter (dispatch)
- `element_cache`: Element workspace (contains topology, basis, integration points)
- `kernel`: Domain kernel
- `elem_id`: Element index in mesh
- `mesh`: Finite element mesh

# Returns
- `ImmutableGeometryCache{N,NIP}` with all geometry precomputed

# Example

```julia
# Replaces update_geometry_cache! pattern:
# OLD: update_geometry_cache!(geometry_cache, ...)
# NEW: geometry_cache = create_geometry_cache(ImmutableGeometryCache, ...)

geometry_cache = create_geometry_cache(
    ImmutableGeometryCache,
    element_cache, kernel, elem_id, mesh
)

# Use in compute_block (no allocations!):
K_kl = compute_block(geometry_cache, material_cache, k, l)
```
"""
function create_geometry_cache(
    ::Type{ImmutableGeometryCache},
    element_cache::ElementCache{T,B,IPS},
    kernel::AbstractKernel,
    elem_id::Int,
    mesh::AbstractMesh
) where {T,B,IPS}
    # Get element info
    topology = element_cache.topology
    basis = element_cache.basis
    ips = element_cache.ips

    N = nnodes(topology)
    NIP = length(ips)

    # Get element nodes and coordinates
    nodes = mesh.connectivity[elem_id]
    X_tuple = ntuple(i -> mesh.nodes[nodes[i]], N)

    # Compute gradients and weights at all integration points
    ∇N_data_tuple = ntuple(NIP) do q
        ip = ips[q]
        ξ = ip.ξ
        w = ip.weight

        # Compute Jacobian and physical gradients
        J = zero(Tensor{2,3,Float64})
        ∇N_ref = get_basis_derivatives(topology, basis, ξ)

        for i in 1:N
            J += X_tuple[i] ⊗ ∇N_ref[i]
        end

        detJ = det(J)
        J_inv = inv(J)

        # Transform to physical gradients
        ∇N_phys = ntuple(N) do i
            J_inv ⋅ ∇N_ref[i]
        end

        ∇N_phys
    end

    # Compute detJ * weight
    detJ_w_tuple = ntuple(NIP) do q
        ip = ips[q]
        ξ = ip.ξ
        w = ip.weight

        # Recompute Jacobian (could optimize by storing from above)
        J = zero(Tensor{2,3,Float64})
        ∇N_ref = get_basis_derivatives(topology, basis, ξ)

        for i in 1:N
            J += X_tuple[i] ⊗ ∇N_ref[i]
        end

        detJ = det(J)
        detJ * w
    end

    return ImmutableGeometryCache{N,NIP}(X_tuple, ∇N_data_tuple, detJ_w_tuple)
end
