# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Concrete types for continuum mechanics formulations and theories.

Abstract types are in abstract.jl, implementations are in formulations.jl.

Must be included after abstract.jl.
"""

# ============================================================================
# CONCRETE THEORY TYPES
# ============================================================================

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

2D plane stress assumption (out-of-plane stress = 0).

Applicable to thin plates and membranes where thickness << in-plane dimensions.
"""
struct PlaneStress <: AbstractContinuumTheory end

"""
    PlaneStrain <: AbstractContinuumTheory

2D plane strain assumption (out-of-plane strain = 0).

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

# ============================================================================
# CONCRETE FORMULATION TYPES
# ============================================================================

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
