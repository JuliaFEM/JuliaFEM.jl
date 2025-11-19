# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Shell formulation API definitions.

This file defines shell-specific abstract types and formulation theories.
Must be included after core api.jl.
"""

# ============================================================================
# SHELL FORMULATION THEORIES
# ============================================================================

"""
    AbstractShellTheory

Abstract type for shell theory variants.

Shell theories differ in how they model transverse shear deformation and thickness effects.

# Concrete Theories
- `ReissnerMindlin`: Thick shells (includes transverse shear)
- `KirchhoffLove`: Thin shells (no transverse shear)

# See Also
- [`ShellFormulation`](@ref)
"""
abstract type AbstractShellTheory end

"""
    ReissnerMindlin <: AbstractShellTheory

Reissner-Mindlin shell theory (thick shells, includes shear).

Assumptions:
- Normals to mid-surface remain straight but NOT perpendicular
- Transverse shear deformation included
- Valid for thick shells (h/L > 1/20)
- 5 DOFs per node: 3 displacements + 2 rotations

# Usage
```julia
formulation = ShellFormulation{ReissnerMindlin}()
physics = Physics(
    formulation=formulation,
    field=DisplacementRotation{3}(),
    mesh=shell_mesh,
    material=steel
)
```
"""
struct ReissnerMindlin <: AbstractShellTheory end

"""
    KirchhoffLove <: AbstractShellTheory

Kirchhoff-Love shell theory (thin shells, no shear).

Assumptions:
- Normals to mid-surface remain straight and perpendicular
- No transverse shear deformation
- Valid for thin shells (h/L < 1/20)
- 3 DOFs per node: 3 displacements (rotations computed from displacements)

# Usage
```julia
formulation = ShellFormulation{KirchhoffLove}()
physics = Physics(
    formulation=formulation,
    field=Displacement{3}(),  # Only displacements, rotations implicit
    mesh=shell_mesh,
    material=aluminum
)
```
"""
struct KirchhoffLove <: AbstractShellTheory end

"""
    ShellFormulation{Theory<:AbstractShellTheory} <: AbstractFormulation

Shell element formulation with theory variant.

# Type Parameter
- `Theory`: Shell theory type (ReissnerMindlin or KirchhoffLove)

# Examples
```julia
# Thick shell (includes shear)
ShellFormulation{ReissnerMindlin}()

# Thin shell (classical theory)
ShellFormulation{KirchhoffLove}()
```

# Fields per Node
- Reissner-Mindlin: 5 DOFs (ux, uy, uz, θx, θy)
- Kirchhoff-Love: 3 DOFs (ux, uy, uz) - rotations implicit
"""
struct ShellFormulation{Theory<:AbstractShellTheory} <: AbstractFormulation end
