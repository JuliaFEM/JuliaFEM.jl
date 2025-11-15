# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Truss formulation API definitions.

This file defines truss-specific abstract types and formulation theories.
Must be included after core api.jl.
"""

# ============================================================================
# TRUSS FORMULATION THEORIES
# ============================================================================

"""
    AbstractTrussTheory

Abstract type for truss theory variants.

Truss elements carry only axial forces (tension/compression), no bending.

# Concrete Theories
- `SimpleTruss`: Standard 2-node truss (axial force only)

# Future Extensions
- `CableTruss`: Cable elements (tension only, no compression)
- `PretensionedTruss`: Trusses with initial stress

# See Also
- [`TrussFormulation`](@ref)
"""
abstract type AbstractTrussTheory end

"""
    SimpleTruss <: AbstractTrussTheory

Simple truss theory (axial force only).

Assumptions:
- Only axial forces (tension/compression)
- No bending moments
- Pin-jointed connections
- 3 DOFs per node in 3D: (ux, uy, uz)
- 2 DOFs per node in 2D: (ux, uy)

# Usage
```julia
formulation = TrussFormulation{SimpleTruss}()
physics = Physics(
    formulation=formulation,
    field=Displacement{3}(),  # 3D truss
    mesh=truss_mesh,
    material=steel
)
```
"""
struct SimpleTruss <: AbstractTrussTheory end

"""
    TrussFormulation{Theory<:AbstractTrussTheory} <: AbstractFormulation

Truss element formulation with theory variant.

# Type Parameter
- `Theory`: Truss theory type (SimpleTruss, CableTruss, etc.)

# Examples
```julia
# Standard truss
TrussFormulation{SimpleTruss}()
```

# Fields per Node
- 3D: 3 DOFs (ux, uy, uz)
- 2D: 2 DOFs (ux, uy)

Use with `Displacement{Dim}` field type.
"""
struct TrussFormulation{Theory<:AbstractTrussTheory} <: AbstractFormulation end
