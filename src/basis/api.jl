# Basis API - Definitions and Interfaces
# This file combines the abstract basis type and the basis evaluation API.

using Tensors
using LinearAlgebra

# Re-export Vec for convenience (from Tensors.jl)
export Vec

"""
    AbstractBasisDescription

Abstract description of how to construct or generate a basis for a given topology.
Different description types can encode symbolic (Vandermonde) generation, external
rules, or manually provided functions.
"""
abstract type AbstractBasisDescription end

"""
    AbstractBasis

Abstract base type for all finite element basis families.

Basis types describe interpolation schemes (H¹ nodal, H(curl), H(div), plate/shell, etc.).
Topology is passed separately to evaluation routines—basis never owns geometry.
"""
abstract type AbstractBasis end

"""
    Lagrange{P} <: AbstractBasis

Standard nodal Lagrange basis of polynomial order `P`.
Topology is supplied separately to evaluation routines.
"""
struct Lagrange{P} <: AbstractBasis end

"""
    Serendipity{P} <: AbstractBasis

Serendipity basis family (reduced tensor product) of order `P` for quads/hexes.
Topology is supplied separately to evaluation routines.
"""
struct Serendipity{P} <: AbstractBasis end

"""
    VandermondeBasisDescription{Topo,Basis} <: AbstractBasisDescription

Description for bases generated from a polynomial ansatz via a Vandermonde system.
Stores the topology, basis family/order, and the polynomial terms used to build
shape functions.
"""
struct VandermondeBasisDescription{Topo<:AbstractTopology,Basis<:AbstractBasis} <: AbstractBasisDescription
    name::String
    description::String
    topology::Type{Topo}
    family::Type{Basis}
    ansatz::Tuple
    function VandermondeBasisDescription(; name, description, topology::Type{Topo}, family::Type{Basis}, ansatz::Tuple) where {Topo<:AbstractTopology,Basis<:AbstractBasis}
        new{Topo,Basis}(name, description, topology, family, ansatz)
    end
end

basis_family(desc::VandermondeBasisDescription) = desc.family
basis_topology(desc::VandermondeBasisDescription) = desc.topology
basis_order(::VandermondeBasisDescription{<:AbstractTopology,<:Lagrange{P}}) where {P} = P
basis_order(::VandermondeBasisDescription{<:AbstractTopology,<:Serendipity{P}}) where {P} = P
basis_order(desc::VandermondeBasisDescription) = error("basis order not defined for $(desc.family)")

reference_coordinates(desc::VandermondeBasisDescription) = reference_coordinates(desc.topology())

# ---------------------------------------------------------------------------
# Basis evaluation API (new design: topology passed separately)
# ---------------------------------------------------------------------------

"""
    get_basis_functions(topology::AbstractTopology, basis::AbstractBasis, xi::Vec)

Evaluate all basis functions at parametric point `xi`.

Returns `NTuple{N, Float64}` where `N` is the number of basis functions for the
given topology–basis combination. Implementations live in generated code
(e.g., lagrange_generated.jl) or custom basis modules.
"""
function get_basis_functions end

"""
    get_basis_derivatives(topology::AbstractTopology, basis::AbstractBasis, xi::Vec)

Evaluate all basis function derivatives at parametric point `xi`.

Returns `NTuple{N, Vec{D, Float64}}` where `D` is the parametric dimension.
Implementations live in generated code (e.g., lagrange_generated.jl) or
custom basis modules.
"""
function get_basis_derivatives end

"""
    get_basis_function(topology, basis, xi, i::Int)

Convenience accessor for a single basis function value.
Equivalent to `get_basis_functions(topology, basis, xi)[i]`.
"""
@inline function get_basis_function(topology::AbstractTopology,
    basis::AbstractBasis,
    xi::Vec,
    i::Int)
    return get_basis_functions(topology, basis, xi)[i]
end

"""
    get_basis_derivative(topology, basis, xi, i::Int)

Convenience accessor for a single basis function derivative.
Equivalent to `get_basis_derivatives(topology, basis, xi)[i]`.
"""
@inline function get_basis_derivative(topology::AbstractTopology,
    basis::AbstractBasis,
    xi::Vec,
    i::Int)
    return get_basis_derivatives(topology, basis, xi)[i]
end

# ---------------------------------------------------------------------------
# Deprecated bridge (old API names)
# ---------------------------------------------------------------------------

"""
    eval_basis!(basis_type, xi) (DEPRECATED)

Use `get_basis_functions(topology, basis, xi)` instead.
Provided temporarily for migration.
"""
function eval_basis! end

"""
    eval_dbasis!(basis_type, xi) (DEPRECATED)

Use `get_basis_derivatives(topology, basis, xi)` instead.
Provided temporarily for migration.
"""
function eval_dbasis! end

# ---------------------------------------------------------------------------
# Degrees of freedom utility
# ---------------------------------------------------------------------------

"""
    ndofs(basis::AbstractBasis)
    ndofs(::Type{<:AbstractBasis})

Total degrees of freedom for this basis. For standard nodal bases, this is
usually equal to the number of basis functions; specialized bases (e.g., plates)
can override to return multiple DOFs per node.
"""
function ndofs end

# Default: basis implementations should override; no assumption about nnodes here.
ndofs(::AbstractBasis) = error("ndofs not implemented for this basis")
ndofs(::Type{<:AbstractBasis}) = error("ndofs not implemented for this basis type")

export AbstractBasis, Lagrange, Serendipity
export ndofs
export get_basis_functions, get_basis_derivatives, get_basis_function, get_basis_derivative
export eval_basis!, eval_dbasis!
