# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Formulation API definitions.

This file defines formulation abstractions - the mathematical discretization strategies
for different types of FEM problems.

Must be included after fields/api.jl (formulations work with fields).
"""

# ============================================================================
# FORMULATION INTERFACE
# ============================================================================

"""
    AbstractFormulation

Abstract type for discretization formulations.

**DESIGN PHILOSOPHY: Formulations are DOMAIN-AGNOSTIC dimensionality concepts.**

# Key Distinction: Formulation vs Theory

**Formulation** (this file):
- Describes DIMENSIONALITY and geometric simplifications
- Domain-agnostic: Used by multiple physics domains
- Examples: FullThreeD, TwoDimensional{T}, Axisymmetric
- Located in: `src/formulations/api.jl`

**Theory** (domain-specific):
- Describes PHYSICS assumptions (stress/strain, kinematics)
- Domain-specific: Only meaningful for one physics domain
- Examples: PlaneStress (continuum), Kirchhoff (plates), Timoshenko (beams)
- Located in: `src/domains/*/theories.jl`

# Why Separate Them?

**Problem**: Heat transfer needs FullThreeD and Axisymmetric, just like continuum mechanics!
If FullThreeD is defined in `domains/continuum/`, heat can't use it without duplication.

**Solution**: Formulations are dimensionality (shared), theories are physics (domain-specific).

```julia
# Domain-agnostic formulations (this file)
FullThreeD()              # Used by: continuum, heat, poisson, acoustics
TwoDimensional{T}()       # Parametric! T is domain theory or Nothing
Axisymmetric()            # Used by: continuum, heat, etc.

# Domain-specific theories (in domains/*/theories.jl)
PlaneStress               # domains/continuum/theories.jl
PlaneStrain               # domains/continuum/theories.jl
Kirchhoff                 # domains/plates/theories.jl
Timoshenko                # domains/beams/theories.jl
```

# Parametric Formulations

Use `TwoDimensional{Theory}` for 2D problems:

```julia
# Continuum mechanics with plane stress theory
TwoDimensional{PlaneStress}()

# Heat transfer (no special theory needed)
TwoDimensional{Nothing}()

# Plates with Kirchhoff theory
TwoDimensional{Kirchhoff}()
```

# Examples

```julia
# 3D solid mechanics (continuum)
physics = Physics(
    formulation = FullThreeD(),           # Domain-agnostic!
    field = Displacement{3}(),
    mesh = mesh,
    material = steel
)

# 3D heat transfer (reuses SAME formulation!)
physics_heat = Physics(
    formulation = FullThreeD(),           # SAME as continuum!
    field = Temperature(),
    mesh = mesh,
    material = steel
)

# 2D plane stress (continuum with theory)
physics_2d = Physics(
    formulation = TwoDimensional{PlaneStress}(),  # Formulation + theory
    field = Displacement{2}(),
    mesh = mesh_2d,
    material = aluminum
)

# Axisymmetric heat (reuses SAME formulation as continuum!)
physics_axisym = Physics(
    formulation = Axisymmetric(),         # Domain-agnostic!
    field = Temperature(),
    mesh = mesh_2d,
    material = steel
)
```

# Assembly Dispatch

Assembly methods dispatch on formulation × field × domain:

```julia
# 3D continuum mechanics
function assemble!(physics::Physics{FullThreeD, Displacement{3}, M, Mat}, ...)
    # Standard 3D displacement-based assembly
    # Implementation in src/assembly/continuum_3d.jl
end

# 3D heat transfer (SAME formulation, different field!)
function assemble!(physics::Physics{FullThreeD, Temperature, M, Mat}, ...)
    # Thermal assembly (scalar field)
    # Implementation in src/assembly/heat.jl
end

# 2D plane stress
function assemble!(physics::Physics{TwoDimensional{PlaneStress}, Displacement{2}, M, Mat}, ...)
    # 2D assembly with plane stress assumptions
    # Implementation in src/assembly/continuum_2d.jl
end
```

# See Also
- Field types: src/fields/api.jl (Displacement, Temperature)
- Physics coupling: src/physics/api.jl (AbstractPhysics)
- Domain theories: src/domains/continuum/theories.jl, src/domains/plates/theories.jl
- Assembly implementations: src/assembly/continuum_3d.jl, src/assembly/heat.jl
- Architectural rationale: docs/src/developer/FORMULATIONS_AND_SOLVERS.md
"""
abstract type AbstractFormulation end

# ============================================================================
# CONTINUUM FORMULATION (Standard FEM)
# ============================================================================

"""
    AbstractContinuumTheory

**LEGACY**: Theory types currently in formulations/ for backward compatibility.

**FUTURE ARCHITECTURE**: These should move to `src/domains/continuum/theories.jl`
to properly separate domain-agnostic formulations from domain-specific theories.

# Current Theories (Will Move to domains/continuum/theories.jl)
- `PlaneStress` - 2D plane stress (σ_zz = 0, thin plates)
- `PlaneStrain` - 2D plane strain (ε_zz = 0, thick plates)

# Domain-Agnostic Formulations (Stay Here)
- `FullThreeD` - Full 3D analysis (used by continuum AND heat!)
- `Axisymmetric` - Axisymmetric analysis (used by continuum AND heat!)

# Theory Selection Guidelines

**PlaneStress (σ_xx, σ_yy, σ_xy, σ_zz = 0):**
- Thin plates and membranes (thickness << length/width)
- Out-of-plane stress σ_zz = 0
- Examples: Sheet metal, aircraft skin, thin-walled structures
- **Domain**: Continuum mechanics only

**PlaneStrain (ε_xx, ε_yy, ε_xy, ε_zz = 0):**
- Thick sections with no variation in z-direction
- Out-of-plane strain ε_zz = 0
- Examples: Dams, tunnels, retaining walls, long cylinders
- **Domain**: Continuum mechanics only

# Mathematical Details

**Plane Stress (thin plate):**
- Stress state: σ_zz = σ_xz = σ_yz = 0
- Strain: ε_zz ≠ 0 (computed from σ_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix

**Plane Strain (thick section):**
- Strain state: ε_zz = γ_xz = γ_yz = 0
- Stress: σ_zz ≠ 0 (computed from ε_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix (different from plane stress!)

# See Also
- Future location: `src/domains/continuum/theories.jl`
- Architectural rationale: `docs/src/developer/FORMULATIONS_AND_SOLVERS.md`
"""
abstract type AbstractContinuumTheory end

"""
    FullThreeD <: AbstractContinuumTheory

Full 3D analysis with no simplifications.

**DOMAIN-AGNOSTIC**: Can be used by ANY physics domain!

# Usage Across Domains

```julia
# Continuum mechanics (solid mechanics)
physics_solid = Physics(
    formulation = FullThreeD(),
    field = Displacement{3}(),
    ...
)

# Heat transfer (SAME formulation!)
physics_heat = Physics(
    formulation = FullThreeD(),
    field = Temperature(),
    ...
)

# Poisson equation (SAME formulation!)
physics_poisson = Physics(
    formulation = FullThreeD(),
    field = Potential(),
    ...
)
```

# Details
- All six stress/flux components in continuum context
- No geometric simplifications
- Most accurate but most expensive
"""
struct FullThreeD <: AbstractContinuumTheory end

"""
    PlaneStress <: AbstractContinuumTheory

2D plane stress assumption (σ_zz = 0).

Applicable to thin plates and membranes where thickness << in-plane dimensions.
"""
struct PlaneStress <: AbstractContinuumTheory end

"""
    PlaneStrain <: AbstractContinuumTheory

2D plane strain assumption (ε_zz = 0).

Applicable to thick sections with no variation in z-direction.
"""
struct PlaneStrain <: AbstractContinuumTheory end

"""
    Axisymmetric <: AbstractContinuumTheory

Axisymmetric analysis (rotation around z-axis).

**DOMAIN-AGNOSTIC**: Can be used by ANY physics domain with axial symmetry!

# Usage Across Domains

```julia
# Continuum mechanics (pressure vessel)
physics_vessel = Physics(
    formulation = Axisymmetric(),
    field = Displacement{2}(),  # (r, z) displacements
    ...
)

# Heat transfer in cylinder (SAME formulation!)
physics_heat = Physics(
    formulation = Axisymmetric(),
    field = Temperature(),  # T(r, z)
    ...
)
```

# Details
- Geometry and loading symmetric about z-axis
- No circumferential variations (∂/∂θ = 0)
- 2D mesh in (r, z) plane represents 3D geometry
- Examples: Pressure vessels, pipes, rotating disks, cylinders
"""
struct Axisymmetric <: AbstractContinuumTheory end

"""
    ContinuumFormulation{Theory} <: AbstractFormulation

Standard continuum mechanics formulation with theory variant.

This is the fundamental FEM formulation for solid mechanics, heat transfer,
and other continuum physics problems.

# Type Parameter
- `Theory <: AbstractContinuumTheory` - Dimensionality/simplification theory

# Examples

```julia
# 3D elasticity
physics = Physics(
    formulation = ContinuumFormulation{FullThreeD}(),
    field = Displacement{3}(),
    mesh = mesh,
    material = steel
)

# 2D plane stress (thin plate)
physics_2d = Physics(
    formulation = ContinuumFormulation{PlaneStress}(),
    field = Displacement{2}(),
    mesh = mesh_2d,
    material = aluminum
)

# 2D plane strain (thick section)
physics_2d = Physics(
    formulation = ContinuumFormulation{PlaneStrain}(),
    field = Displacement{2}(),
    mesh = mesh_2d,
    material = concrete
)

# Axisymmetric (cylinder)
physics_axisym = Physics(
    formulation = ContinuumFormulation{Axisymmetric}(),
    field = Displacement{2}(),  # (r, z) displacements
    mesh = mesh_2d,
    material = steel
)
```

# Assembly Dispatch

Assembly methods specialize on theory × field combinations:

```julia
# 3D solid mechanics
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
    # Standard 3D displacement-based assembly
    # Full 6×6 strain-displacement matrix (Bε)
    # 6×6 constitutive matrix (Dε)
end

# 2D plane stress
function assemble!(physics::Physics{ContinuumFormulation{PlaneStress}, Displacement{2}, M, Mat})
    # 2D assembly with plane stress assumptions
    # 3×3 reduced strain-displacement matrix
    # 3×3 plane stress constitutive matrix
end

# Heat transfer (same formulation, different field!)
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Temperature, M, Mat})
    # Thermal assembly (scalar field)
    # Thermal conductivity matrix
end
```

# Implementation Location

Concrete assembly implementations are in:
- `src/assembly/continuum_3d.jl` - 3D continuum mechanics
- `src/assembly/continuum_2d.jl` - 2D plane stress/strain
- `src/assembly/axisymmetric.jl` - Axisymmetric problems

# See Also
- [`AbstractContinuumTheory`](@ref) - Theory variants
- Field types: src/fields/api.jl (Displacement, Temperature)
- Physics coupling: src/physics/api.jl (AbstractPhysics)
- Assembly: src/assembly/continuum_*.jl
"""
struct ContinuumFormulation{Theory<:AbstractContinuumTheory} <: AbstractFormulation end
