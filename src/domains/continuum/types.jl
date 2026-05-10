# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Concrete types for continuum mechanics formulations and theories.

Abstract types are in `abstract.jl`; concrete theory structs and
`ContinuumFormulation` live in this file.

Must be included after `abstract.jl`.
"""

# ============================================================================
# CONCRETE THEORY TYPES
# ============================================================================

"""
    ThreeDimensional <: AbstractContinuumTheory

Bulk three-dimensional model: no in-plane or axisymmetric reduction. All
independent tensor components are retained at the quadrature point (six for
symmetric mechanical stress / strain; three for isotropic flux, etc.).
Domain-agnostic tag shared by `ContinuumKernel`, `HeatKernel`, Darcy-style
kernels, and other `ContinuumFormulation{…}` drivers on 3D meshes.
"""
struct ThreeDimensional <: AbstractContinuumTheory end

"""
    PlaneStress <: AbstractContinuumTheory

2D plane stress assumption (out-of-plane stress = 0). Applicable to thin
plates and membranes where thickness ≪ in-plane dimensions.
"""
struct PlaneStress <: AbstractContinuumTheory end

"""
    PlaneStrain <: AbstractContinuumTheory

2D plane strain assumption (out-of-plane strain = 0). Applicable to thick
sections with no variation in z-direction.
"""
struct PlaneStrain <: AbstractContinuumTheory end

"""
    Axisymmetric <: AbstractContinuumTheory

Axisymmetric analysis (rotation around z-axis). Geometry and loading are
symmetric about the z-axis with no circumferential variation. The 2D mesh
in (r, z) represents the full 3D geometry. Domain-agnostic.
"""
struct Axisymmetric <: AbstractContinuumTheory end

# ============================================================================
# CONCRETE FORMULATION TYPES
# ============================================================================

"""
    ContinuumFormulation{Theory<:AbstractContinuumTheory} <: AbstractFormulation

Standard continuum mechanics formulation, parameterised by theory variant.
Used as a type tag inside `ContinuumKernel{Theory, Material, Field}`.

# Examples

```julia
ContinuumFormulation{ThreeDimensional}()
ContinuumFormulation{PlaneStress}()
ContinuumFormulation{PlaneStrain}()
ContinuumFormulation{Axisymmetric}()
```

The actual assembly contract is defined by `AbstractKernel` (see
`src/assemblers/abstract.jl` for the kernel defaults and
`src/assemblers/microkernel.jl` for the DOF-based microkernel trait);
concrete continuum kernels live in `src/domains/continuum/kernel.jl`.
"""
struct ContinuumFormulation{Theory<:AbstractContinuumTheory} <: AbstractFormulation end
