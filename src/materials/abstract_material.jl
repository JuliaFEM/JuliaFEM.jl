"""
Abstract type hierarchy for material models in JuliaFEM.

This module defines the base abstract type `AbstractMaterial` and the standard
interface that all material models must implement.

# Type Hierarchy

```
AbstractMaterial
├── AbstractElasticMaterial (stateless materials)
│   ├── LinearElastic
│   └── NeoHookean
└── AbstractPlasticMaterial (stateful materials)
    ├── PerfectPlasticity
    └── FiniteStrainPlasticity
```

# Interface Requirements

All concrete material types must implement:

```julia
compute_stress(
    material::AbstractMaterial,
    ε::SymmetricTensor{2,3,T},
    state_old,
    Δt::Float64
) -> (σ::SymmetricTensor{2,3,T}, 𝔻::SymmetricTensor{4,3,T}, state_new)
```

Where:
- `material` - Material model instance
- `ε` - Strain tensor (small strain or Green-Lagrange for finite strain)
- `state_old` - Material state from previous timestep (`nothing` for stateless)
- `Δt` - Time increment [s]
- `σ` - Stress tensor (Cauchy or 2nd Piola-Kirchhoff)
- `𝔻` - Tangent modulus (∂σ/∂ε)
- `state_new` - Updated material state (`nothing` for stateless)

# Material State Convention

**For stateless materials (elastic):**
- `state_old = nothing`
- `state_new = nothing`
- No history dependence

**For stateful materials (plastic, damage, etc.):**
- `state_old::MaterialState` - Frozen during Newton iterations
- `state_new::MaterialState` - Computed but NOT stored until convergence
- State updated only after successful time step

# Design Principles

1. **Uniform API** - Same function signature for all materials
2. **Type stability** - Concrete return types (no Union types)
3. **Zero allocation** - Stack-allocated tensors (Tensors.jl)
4. **Composability** - Materials work with any element type
5. **GPU-ready** - All operations are POD (plain old data)

# Example Usage

```julia
using Tensors

# Create material
steel = LinearElastic(E=200e9, ν=0.3)

# Define strain
ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))

# Compute stress
σ, 𝔻, _ = compute_stress(steel, ε, nothing, 0.0)
```
"""

using Tensors

"""
    AbstractMaterial

Base abstract type for all material models.

All concrete material types must inherit from this type and implement
the `compute_stress` interface.

# Required Interface

```julia
compute_stress(
    material::AbstractMaterial,
    ε::SymmetricTensor{2,3,T},
    state_old,
    Δt::Float64
) where T -> (σ, 𝔻, state_new)
```

# See Also
- [`AbstractElasticMaterial`](@ref) - Base type for stateless materials
- [`AbstractPlasticMaterial`](@ref) - Base type for stateful materials
- [`compute_stress`](@ref) - Standard interface function
"""
abstract type AbstractMaterial end

"""
    AbstractElasticMaterial <: AbstractMaterial

Base abstract type for stateless elastic materials.

Elastic materials have no internal state variables and stress depends
only on current strain. Examples: LinearElastic, NeoHookean.

For these materials:
- `state_old = nothing`
- `state_new = nothing`
- `compute_stress` is a pure function of strain

# Subtypes
- `LinearElastic` - Linear isotropic elasticity (Hooke's law)
- `NeoHookean` - Hyperelastic material (finite strain)
"""
abstract type AbstractElasticMaterial <: AbstractMaterial end

"""
    AbstractPlasticMaterial <: AbstractMaterial

Base abstract type for stateful plastic materials.

Plastic materials have internal state variables (e.g., plastic strain)
that evolve during loading. Examples: PerfectPlasticity, FiniteStrainPlasticity.

For these materials:
- `state_old::PlasticityState` - State at beginning of time step
- `state_new::PlasticityState` - State after loading (committed only on convergence)
- History-dependent behavior (path-dependent)

# Subtypes
- `PerfectPlasticity` - Von Mises plasticity without hardening
- `FiniteStrainPlasticity` - Multiplicative plasticity (finite strain)
"""
abstract type AbstractPlasticMaterial <: AbstractMaterial end

"""
    compute_stress(material, ε, state_old, Δt) -> (σ, 𝔻, state_new)

Standard interface for computing stress and tangent modulus.

This function must be implemented by all concrete material types.

# Arguments
- `material::AbstractMaterial` - Material model instance
- `ε::SymmetricTensor{2,3,T}` - Strain tensor
  - Small strain: ε = ½(∇u + ∇uᵀ)
  - Large strain: E = ½(FᵀF - I) (Green-Lagrange)
- `state_old` - Material state from previous timestep
  - `nothing` for stateless materials (elastic)
  - `MaterialState` for stateful materials (plastic, damage, etc.)
- `Δt::Float64` - Time increment [s]

# Returns
- `σ::SymmetricTensor{2,3,T}` - Stress tensor
  - Small strain: Cauchy stress
  - Large strain: 2nd Piola-Kirchhoff stress
- `𝔻::SymmetricTensor{4,3,T}` - Material tangent modulus (∂σ/∂ε)
- `state_new` - Updated material state
  - `nothing` for stateless materials
  - `MaterialState` for stateful materials

# Type Constraints
- Input strain `ε` and output stress `σ` have same element type `T`
- Tangent `𝔻` is 4th-order symmetric tensor
- Return type is concrete (no `Any` or `Union`)

# Performance Requirements
- Zero allocations (stack-only computation)
- Type-stable (concrete return types)
- SIMD-friendly (Tensors.jl operations)

# Examples

## Stateless Material (Linear Elastic)
```julia
steel = LinearElastic(E=200e9, ν=0.3)
ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
σ, 𝔻, state_new = compute_stress(steel, ε, nothing, 0.0)
@assert state_new === nothing  # Stateless
```

## Stateful Material (Perfect Plasticity)
```julia
steel = PerfectPlasticity(E=200e9, ν=0.3, σ_y=250e6)
state₀ = initial_state(steel)
ε = SymmetricTensor{2,3}((0.002, 0.0, 0.0, 0.0, 0.0, 0.0))
σ, 𝔻, state₁ = compute_stress(steel, ε, state₀, 1.0)
@assert state₁ !== state₀  # State evolved
```

# Implementation Notes

## Newton Iteration Compatibility
Material models must be compatible with Newton-Raphson iteration:
- `state_old` is frozen during all Newton iterations
- `state_new` is computed but NOT stored until convergence
- If Newton fails, material state remains unchanged

## Thread Safety
Implementations should be thread-safe:
- No shared mutable state
- Pure functions (for stateless materials)
- State updates are copy-on-write (for stateful materials)

## GPU Compatibility
For GPU compatibility:
- Use only Tensors.jl operations (no dynamic allocations)
- Avoid function pointers or closures in hot paths
- All types should be POD (plain old data)

# See Also
- [`AbstractMaterial`](@ref) - Base abstract type
- [`AbstractElasticMaterial`](@ref) - Stateless materials
- [`AbstractPlasticMaterial`](@ref) - Stateful materials
"""
function compute_stress end

# Note: Concrete implementations are in separate files:
# - src/materials/linear_elastic.jl
# - src/materials/neo_hookean.jl
# - src/materials/perfect_plasticity.jl
# - src/materials/finite_strain_plasticity.jl
