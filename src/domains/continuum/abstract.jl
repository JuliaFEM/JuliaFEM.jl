# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Abstract types for continuum mechanics domain.

This file contains ONLY abstract type definitions for the continuum domain.
Concrete types are in types.jl, implementations are in theory-specific files.
"""

# ============================================================================
# FORMULATION ABSTRACTS
# ============================================================================

"""
    AbstractFormulation

Abstract type for discretization formulations.

DESIGN PHILOSOPHY: Formulations are DOMAIN-AGNOSTIC dimensionality concepts.

# Key Distinction: Formulation vs Theory

Formulation (domain-agnostic):
- Describes DIMENSIONALITY and geometric simplifications
- Used by multiple physics domains
- Examples: FullThreeD, Axisymmetric
- Can be reused across continuum, heat, acoustics, etc.

Theory (domain-specific):
- Describes PHYSICS assumptions (stress/strain, kinematics)
- Only meaningful for one physics domain
- Examples: PlaneStress (continuum), Kirchhoff (plates)

# Why Separate Them?

Problem: Heat transfer needs FullThreeD and Axisymmetric, just like continuum!
If FullThreeD is defined in domains/continuum/, heat can't use it without duplication.

Solution: Formulations are dimensionality (shared), theories are physics (domain-specific).

# Examples

```julia
# Domain-agnostic formulations
FullThreeD()              # Used by: continuum, heat, poisson, acoustics
Axisymmetric()            # Used by: continuum, heat, etc.

# Domain-specific theories (in domains/*/types.jl)
PlaneStress               # domains/continuum/types.jl
PlaneStrain               # domains/continuum/types.jl
```

# See Also
- Concrete formulation types: `continuum/types.jl` (ContinuumFormulation)
"""
abstract type AbstractFormulation end

"""
    AbstractContinuumTheory

Abstract type for continuum mechanics theories.

Domain-specific physics assumptions for solid mechanics.

# Theories

PlaneStress (σ_xx, σ_yy, σ_xy, σ_zz = 0):
- Thin plates and membranes (thickness << length/width)
- Out-of-plane stress σ_zz = 0
- Examples: Sheet metal, aircraft skin, thin-walled structures

PlaneStrain (ε_xx, ε_yy, ε_xy, ε_zz = 0):
- Thick sections with no variation in z-direction
- Out-of-plane strain ε_zz = 0
- Examples: Dams, tunnels, retaining walls, long cylinders

FullThreeD:
- No simplifications, all six stress/strain components
- Most accurate but most expensive

Axisymmetric:
- Geometry and loading symmetric about z-axis
- No circumferential variations (∂/∂θ = 0)
- Examples: Pressure vessels, pipes, rotating disks

# Mathematical Details

Plane Stress (thin plate):
- Stress state: σ_zz = σ_xz = σ_yz = 0
- Strain: ε_zz ≠ 0 (computed from σ_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix

Plane Strain (thick section):
- Strain state: ε_zz = γ_xz = γ_yz = 0
- Stress: σ_zz ≠ 0 (computed from ε_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix (different from plane stress!)

# See Also
- Concrete theories: `continuum/types.jl` (FullThreeD, PlaneStress, PlaneStrain, Axisymmetric)
"""
abstract type AbstractContinuumTheory end
