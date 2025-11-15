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

Formulation defines HOW we discretize the governing equations. Different formulations
exist for different physics domains:

- **Continuum formulations** (this file) - Standard FEM for solid/fluid mechanics
- **Beam formulations** (src/beams/api.jl) - 1D structural elements
- **Shell formulations** (src/shells/api.jl) - 2D structural elements  
- **Truss formulations** (src/trusses/api.jl) - 1D axial elements

# Type Hierarchy
- `ContinuumFormulation{Theory}` - Standard continuum FEM (here)
- `BeamFormulation{Theory}` - Beam elements (src/beams/api.jl)
- `ShellFormulation{Theory}` - Shell elements (src/shells/api.jl)
- `TrussFormulation{Theory}` - Truss elements (src/trusses/api.jl)

# Design Philosophy

**Formulation + Field = Dispatch pattern**

The combination of formulation and field type determines:
- Assembly method dispatch
- Element stiffness computation
- Stress/strain tensor dimensions
- DOF coupling patterns

# Examples

```julia
# 3D solid mechanics
physics = Physics(
    formulation = ContinuumFormulation{FullThreeD}(),
    field = Displacement{3}(),
    mesh = mesh,
    material = steel
)

# 2D plane stress
physics_2d = Physics(
    formulation = ContinuumFormulation{PlaneStress}(),
    field = Displacement{2}(),
    mesh = mesh_2d,
    material = aluminum
)

# Beam structure
physics_beam = Physics(
    formulation = BeamFormulation{Timoshenko}(),
    field = DisplacementRotation{3}(),
    mesh = beam_mesh,
    material = steel
)
```

# Assembly Dispatch

Specialized assembly methods dispatch on formulation × field:

```julia
# 3D continuum mechanics
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
    # Standard 3D displacement-based assembly
    # Implementation in src/assembly/continuum_3d.jl
end

# 2D plane stress
function assemble!(physics::Physics{ContinuumFormulation{PlaneStress}, Displacement{2}, M, Mat})
    # 2D assembly with plane stress assumptions
    # Implementation in src/assembly/continuum_2d.jl
end

# Beam elements
function assemble!(physics::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat})
    # Beam-specific assembly (6 DOFs per node)
    # Implementation in src/assembly/beams.jl
end
```

# See Also
- Field types: src/fields/api.jl (Displacement, Temperature, DisplacementRotation)
- Physics coupling: src/physics/api.jl (AbstractPhysics)
- Domain-specific formulations: src/beams/api.jl, src/shells/api.jl, src/trusses/api.jl
- Assembly implementations: src/assembly/continuum_3d.jl, src/assembly/beams.jl, etc.
"""
abstract type AbstractFormulation end

# ============================================================================
# CONTINUUM FORMULATION (Standard FEM)
# ============================================================================

"""
    AbstractContinuumTheory

Theory variants for continuum formulation.

Controls dimensionality reduction and stress/strain assumptions for continuum
mechanics problems.

# Concrete Theories
- `FullThreeD` - Full 3D analysis (no simplifications)
- `PlaneStress` - 2D plane stress (σ_zz = 0, thin plates)
- `PlaneStrain` - 2D plane strain (ε_zz = 0, thick plates)
- `Axisymmetric` - Axisymmetric analysis (rotation around z-axis)

# Theory Selection Guidelines

**FullThreeD (σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz):**
- General 3D solid mechanics
- No simplifying assumptions
- Most accurate but most expensive

**PlaneStress (σ_xx, σ_yy, σ_xy, σ_zz = 0):**
- Thin plates and membranes (thickness << length/width)
- Out-of-plane stress σ_zz = 0
- Examples: Sheet metal, aircraft skin, thin-walled structures

**PlaneStrain (ε_xx, ε_yy, ε_xy, ε_zz = 0):**
- Thick sections with no variation in z-direction
- Out-of-plane strain ε_zz = 0
- Examples: Dams, tunnels, retaining walls, long cylinders

**Axisymmetric (σ_rr, σ_θθ, σ_zz, σ_rz):**
- Geometry and loading symmetric about z-axis
- No circumferential variations
- Examples: Pressure vessels, pipes, rotating disks

# Usage

```julia
# Full 3D solid mechanics
formulation = ContinuumFormulation{FullThreeD}()

# 2D plane stress (thin plate)
formulation = ContinuumFormulation{PlaneStress}()

# 2D plane strain (thick section)
formulation = ContinuumFormulation{PlaneStrain}()

# Axisymmetric (cylinder, sphere)
formulation = ContinuumFormulation{Axisymmetric}()
```

# Mathematical Details

**Plane Stress (thin plate):**
- Stress state: σ_zz = σ_xz = σ_yz = 0
- Strain: ε_zz ≠ 0 (computed from σ_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix

**Plane Strain (thick section):**
- Strain state: ε_zz = γ_xz = γ_yz = 0
- Stress: σ_zz ≠ 0 (computed from ε_zz = 0 condition)
- Constitutive: 3×3 reduced stiffness matrix (different from plane stress!)

**Axisymmetric:**
- Cylindrical coordinates (r, θ, z)
- No ∂/∂θ terms (axial symmetry)
- 4 stress components: σ_rr, σ_θθ, σ_zz, σ_rz
- Hoop stress σ_θθ from radial displacement

# See Also
- [`ContinuumFormulation`](@ref) - Formulation struct using these theories
"""
abstract type AbstractContinuumTheory end

"""
    FullThreeD <: AbstractContinuumTheory

Full 3D analysis with no simplifications.

All six stress components: σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz
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

Geometry and loading symmetric about z-axis with no circumferential variations.
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
