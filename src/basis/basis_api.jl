# Basis Function API - New Design (Nov 2025)
#
# This file implements the new basis function API based on comprehensive
# benchmarking results (see docs/book/adr-003-basis-function-api.md).
#
# Key design decisions:
# 1. Topology passed separately: get_basis_functions(topology, basis, xi)
# 2. Return tuples (zero allocation, type-stable)
# 3. Use simple runtime indexing for single access (fastest!)
# 4. Clear naming: get_basis_functions, get_basis_derivatives
#
# Performance: 6.5 ns for Tet10 derivatives, zero allocations

"""
    get_basis_functions(topology::AbstractTopology, basis::AbstractBasis, xi::Vec)

Evaluate all basis functions at parametric point `xi`.

Returns `NTuple{N, Float64}` where N is the number of nodes/DOFs for the given
topology and basis combination.

# Arguments
- `topology`: Element topology (e.g., `Tetrahedron()`, `Triangle()`)
- `basis`: Interpolation scheme (e.g., `Lagrange{1}()`, `Lagrange{2}()`)
- `xi`: Parametric coordinates as `Vec{D, T}` where D is spatial dimension

# Returns
- `NTuple{N, Float64}`: All N basis function values

# Examples

```julia
# Tet10: 10-node quadratic tetrahedron
topology = Tetrahedron()
basis = Lagrange{2}()
xi = Vec(0.25, 0.25, 0.2)

N_all = get_basis_functions(topology, basis, xi)
# Returns: (N1, N2, ..., N10) as NTuple{10, Float64}

# Access single basis function (simple runtime indexing)
N_5 = N_all[5]

# Verify partition of unity
@assert abs(sum(N_all) - 1.0) < 1e-10
```

# Performance
- Tet10 (10 nodes): ~3.6 ns, zero allocations
- Triangle P1 (3 nodes): ~2.5 ns, zero allocations

See also: [`get_basis_derivatives`](@ref), [`get_basis_function`](@ref)
"""
function get_basis_functions end

"""
    get_basis_derivatives(topology::AbstractTopology, basis::AbstractBasis, xi::Vec)

Evaluate all basis function derivatives at parametric point `xi`.

Returns `NTuple{N, Vec{D, Float64}}` where:
- N is the number of nodes/DOFs
- D is the spatial dimension

Each derivative is ∇N_i = (∂N_i/∂ξ₁, ∂N_i/∂ξ₂, ..., ∂N_i/∂ξ_D)

# Arguments
- `topology`: Element topology (e.g., `Tetrahedron()`, `Triangle()`)
- `basis`: Interpolation scheme (e.g., `Lagrange{1}()`, `Lagrange{2}()`)
- `xi`: Parametric coordinates as `Vec{D, T}` where D is spatial dimension

# Returns
- `NTuple{N, Vec{D, Float64}}`: All N basis function gradients

# Examples

```julia
# Tet10: 10-node quadratic tetrahedron
topology = Tetrahedron()
basis = Lagrange{2}()
xi = Vec(0.25, 0.25, 0.2)

dN_all = get_basis_derivatives(topology, basis, xi)
# Returns: (∇N1, ∇N2, ..., ∇N10) as NTuple{10, Vec{3, Float64}}

# Access single derivative (simple runtime indexing)
dN_5 = dN_all[5]  # Vec{3, Float64} gradient for node 5

# Use in assembly (typical pattern)
for i in 1:10
    for j in 1:10
        dNi = dN_all[i]
        dNj = dN_all[j]
        K_local[i,j] += dot(dNi, dNj) * detJ  # Simplified
    end
end
```

# Performance
- Tet10 (10 nodes, 3D): ~6.5 ns, zero allocations ← HOT PATH!
- Triangle P2 (6 nodes, 2D): ~4.0 ns, zero allocations

This is the critical path for stiffness matrix assembly!

See also: [`get_basis_functions`](@ref), [`get_basis_derivative`](@ref)
"""
function get_basis_derivatives end

"""
    get_basis_function(topology, basis, xi, i::Int)

Convenience function to get a single basis function value.

Equivalent to `get_basis_functions(topology, basis, xi)[i]`.

# Note
Simple runtime tuple indexing is fastest (~1 ns overhead). No need for
Val dispatch or @generated functions (benchmarks showed those are 25-300× slower!).

# Examples

```julia
N_5 = get_basis_function(Tetrahedron(), Lagrange{2}(), xi, 5)
# Equivalent to:
N_5 = get_basis_functions(Tetrahedron(), Lagrange{2}(), xi)[5]
```
"""
@inline function get_basis_function(topology::AbstractTopology,
    basis::AbstractBasis,
    xi::Vec,
    i::Int)
    return get_basis_functions(topology, basis, xi)[i]
end

"""
    get_basis_derivative(topology, basis, xi, i::Int)

Convenience function to get a single basis function derivative.

Equivalent to `get_basis_derivatives(topology, basis, xi)[i]`.

# Note
Simple runtime tuple indexing is fastest (~1 ns overhead).

# Examples

```julia
dN_5 = get_basis_derivative(Tetrahedron(), Lagrange{2}(), xi, 5)
# Equivalent to:
dN_5 = get_basis_derivatives(Tetrahedron(), Lagrange{2}(), xi)[5]
```
"""
@inline function get_basis_derivative(topology::AbstractTopology,
    basis::AbstractBasis,
    xi::Vec,
    i::Int)
    return get_basis_derivatives(topology, basis, xi)[i]
end

# ============================================================================
# Implementation for Lagrange Basis
# ============================================================================
# 
# These implementations are auto-generated for all topology/degree combinations.
# See src/basis/lagrange_generator.jl for the code generation.
#
# Pattern for each topology:
#
# @inline function get_basis_functions(::Topology, ::Lagrange{P}, xi::Vec{D,T}) where T
#     # Compute basis functions
#     return (N1, N2, ..., Nn)  # NTuple{N, Float64}
# end
#
# @inline function get_basis_derivatives(::Topology, ::Lagrange{P}, xi::Vec{D,T}) where T
#     # Compute derivatives
#     return (dN1, dN2, ..., dNn)  # NTuple{N, Vec{D, Float64}}
# end

# Include auto-generated implementations
# TODO: Update lagrange_generator.jl to generate get_basis_* functions
# For now, implementations will be added to lagrange_generated.jl

# ============================================================================
# Backward Compatibility Bridge (Temporary)
# ============================================================================
#
# These functions bridge to the new API from old API.
# Will be deprecated after full migration.

"""
    eval_basis!(basis_type, xi) (DEPRECATED)

**DEPRECATED**: Use `get_basis_functions(topology, basis, xi)` instead.

This function is provided for backward compatibility during migration.
"""
function eval_basis! end

"""
    eval_dbasis!(basis_type, xi) (DEPRECATED)

**DEPRECATED**: Use `get_basis_derivatives(topology, basis, xi)` instead.

This function is provided for backward compatibility during migration.
"""
function eval_dbasis! end

# TODO: Add deprecation warnings after new API is implemented
# @deprecate eval_basis!(args...) get_basis_functions(args...)
# @deprecate eval_dbasis!(args...) get_basis_derivatives(args...)
