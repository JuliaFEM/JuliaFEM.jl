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

"""
    nbasis(topology::AbstractTopology, basis::AbstractBasis) -> Int

Return the number of basis functions for the given basis on the given topology.

**Zero-cost:** This function compiles to a constant integer for concrete types.
The return value is compile-time known, enabling type-stable allocations and
constant propagation throughout assembly code.

# Arguments
- `topology::AbstractTopology`: Element topology instance (e.g., `Triangle{3}()`, `Tetrahedron{10}()`)
- `basis::AbstractBasis`: Basis instance (e.g., `Lagrange{1}()`, `Lagrange{2}()`)

# Returns
- `Int`: Number of basis functions (compile-time constant for generated bases)

# Examples
```julia
# Standard nodal bases
nbasis(Triangle{3}(), Lagrange{1}())       # 3 (linear triangle)
nbasis(Triangle{6}(), Lagrange{2}())       # 6 (quadratic triangle)
nbasis(Tetrahedron{10}(), Lagrange{2}())   # 10 (quadratic tetrahedron)

# Exotic elements (nodes ≠ basis functions)
nbasis(Triangle{3}(), DKT())               # 9 (3 nodes but 9 DOFs!)

# Serendipity families
nbasis(Quadrilateral{8}(), Serendipity{2}())  # 8 (reduced quadratic quad)
```

# Implementation Notes

For generated bases (Lagrange, Serendipity), `nbasis` functions are automatically
generated by `basis_generator.jl` alongside the basis function implementations.
This ensures consistency: the generator knows exactly how many basis functions
it produced, so `nbasis` always returns the correct value.

For custom/special bases (DKT, hierarchical, etc.), implement manually:
```julia
@inline nbasis(::Triangle{3}, ::DKT) = 9
```

For truly dynamic bases (adaptive refinement), use instance-based dispatch:
```julia
@inline nbasis(::K, basis::HierarchicalBasis) where {K<:AbstractTopology} = basis.active_modes
```

# Compile-Time Verification

You can verify zero-cost compilation:
```julia
@code_llvm nbasis(Triangle{3}(), Lagrange{1}())
# Should show: ret i64 3 (direct constant return)

@code_native nbasis(Triangle{3}(), Lagrange{1}())
# Should show: mov rax, 3; ret (single instruction)
```

# Usage in Validation

The primary use case is validating the Ciarlet triplet (K, P, Σ):
```julia
function validate_dof_consistency(dof, K, P)
    n_dofs = ndofs(dof, K)
    n_basis = nbasis(K(), P())  # ← Zero-cost!
    @assert n_dofs == n_basis "Inconsistent element: \$n_basis basis ≠ \$n_dofs DOFs"
end
```

See also: [`get_basis_functions`](@ref), [`validate_dof_consistency`](@ref)
"""
function nbasis end

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

export AbstractBasis, Lagrange, Serendipity
export get_basis_functions, get_basis_derivatives, get_basis_function, get_basis_derivative
export nbasis
export eval_basis!, eval_dbasis!
