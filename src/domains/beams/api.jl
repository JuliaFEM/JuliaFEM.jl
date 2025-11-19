# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Beam formulation API definitions.

This file defines beam-specific abstract types and formulation theories.
Must be included after core api.jl.
"""

# ============================================================================
# BEAM FORMULATION THEORIES
# ============================================================================

"""
    AbstractBeamTheory

Abstract type for beam theory variants.

Beam theories differ in how they model shear deformation and cross-section kinematics.

# Concrete Theories
- `EulerBernoulli`: Classical beam theory (no shear deformation)
- `Timoshenko`: Includes shear deformation (thick beams)

# See Also
- [`BeamFormulation`](@ref)
"""
abstract type AbstractBeamTheory end

"""
    EulerBernoulli <: AbstractBeamTheory

Euler-Bernoulli beam theory (classical, no shear deformation).

Assumptions:
- Plane sections remain plane and perpendicular to neutral axis
- No transverse shear deformation
- Valid for slender beams (L/h > 10)

# Usage
```julia
formulation = BeamFormulation{EulerBernoulli}()
physics = Physics(
    formulation=formulation,
    field=DisplacementRotation{3}(),
    mesh=beam_mesh,
    material=steel
)
```
"""
struct EulerBernoulli <: AbstractBeamTheory end

"""
    Timoshenko <: AbstractBeamTheory

Timoshenko beam theory (includes shear deformation).

Assumptions:
- Plane sections remain plane but NOT perpendicular to neutral axis
- Transverse shear deformation included
- Valid for thick beams and higher frequencies

# Usage
```julia
formulation = BeamFormulation{Timoshenko}()
physics = Physics(
    formulation=formulation,
    field=DisplacementRotation{3}(),
    mesh=beam_mesh,
    material=steel
)
```
"""
struct Timoshenko <: AbstractBeamTheory end

"""
    BeamFormulation{Theory<:AbstractBeamTheory} <: AbstractFormulation

Beam element formulation with theory variant.

# Type Parameter
- `Theory`: Beam theory type (EulerBernoulli or Timoshenko)

# Examples
```julia
# Slender beam (classical theory)
BeamFormulation{EulerBernoulli}()

# Thick beam (includes shear)
BeamFormulation{Timoshenko}()
```

# Fields per Node
Typically 6 DOFs in 3D: (ux, uy, uz, θx, θy, θz)
Use with `DisplacementRotation{3}` field type.
"""
struct BeamFormulation{Theory<:AbstractBeamTheory} <: AbstractFormulation end
