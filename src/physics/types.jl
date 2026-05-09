# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Physics category types for material trait dispatch.

Defines abstract physics categories used for:
- Material trait functions (required_field_type)
- Material cache generation
- Physics-based dispatch

These are lower-level type tags, not complete FEM problems.
"""

"""
    Elasticity{Dim} <: AbstractPhysics

Elasticity physics in Dim dimensions (2D or 3D).

Used for trait-based dispatch to determine required fields:
```julia
required_field_type(Elasticity{3}())  # → Displacement{3}
```
"""
struct Elasticity{Dim} <: AbstractPhysics end

"""
    Thermal{Dim} <: AbstractPhysics

Thermal physics (heat transfer) in `Dim` dimensions (2 or 3).

Used for trait-based dispatch:
```julia
required_field_type(Thermal{2}())  # → Temperature (2D plane heat)
required_field_type(Thermal{3}())  # → Temperature (3D heat)
```
"""
struct Thermal{Dim} <: AbstractPhysics end

# ============================================================================
# TRAIT FUNCTIONS
# ============================================================================

"""
    required_field_type(physics::AbstractPhysics)

Returns the field type required by the given physics.

Used for compile-time inference of material cache structure and
field requirements.

# Examples
```julia
required_field_type(Elasticity{3}())  # → Displacement{3}
required_field_type(Thermal{2}())     # → Temperature (2D)
required_field_type(Thermal{3}())     # → Temperature (3D)
```

Materials can declare their physics requirements:
```julia
struct ThermoElasticMaterial <: AbstractMaterial
    # ...
end

# Material supports both elasticity and thermal physics
supported_physics(::ThermoElasticMaterial) = (Elasticity{3}(), Thermal{3}())

# System maps this to required field types:
# Elasticity{3} → Displacement{3}
# Thermal → Temperature
```
"""
function required_field_type end

required_field_type(::Elasticity{Dim}) where Dim = Displacement{Dim}
required_field_type(::Thermal{Dim}) where Dim = Temperature
# required_field_type(::Fluid{Dim}) where Dim = Velocity{Dim}  # Add when Velocity field exists
