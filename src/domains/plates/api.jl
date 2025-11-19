# Plate Elements API
#
# This file defines the interface for plate bending elements following
# the modern JuliaFEM architecture with zero-allocation design.
#
# Reference: Batoz, J.-L., Bathe, K.-J., & Ho, L.-W. (1980).
# "A study of three-node triangular plate bending elements."
# International Journal for Numerical Methods in Engineering, 15(12), 1771-1812.

# Dependencies
using Tensors

# This file is part of JuliaFEM, but can be tested standalone
# When integrated: using ..JuliaFEM
# When standalone: Define AbstractMaterial, AbstractPlateFormulation, etc. locally

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractPlateFormulation

Abstract supertype for all plate bending formulations.

Plate elements typically have 3 DOFs per node:
- w: transverse displacement (out-of-plane)
- θx: rotation about x-axis
- θy: rotation about y-axis

Implementations include:
- DKT (Discrete Kirchhoff Triangle): Kirchhoff theory, C0 continuous
- Mindlin plates: Mindlin-Reissner theory, includes shear deformation
"""
abstract type AbstractPlateFormulation <: AbstractFormulation end

"""
    DKTFormulation <: AbstractPlateFormulation

Discrete Kirchhoff Triangle plate bending element.

Based on Kirchhoff plate theory (thin plates, neglects shear deformation).
Uses a discrete approach to enforce Kirchhoff constraints at selected points.

**Theory:**
- 3-node triangular element (Tri3 topology)
- 3 DOFs per node: w, θx, θy (9 total)
- Kirchhoff hypothesis: γxz = γyz = 0 (no shear deformation)
- C0 continuous (displacements), but enforces C1 behavior discretely
- Accurate for thin plates (thickness/span < 1/20)

**Material Properties Required:**
- Young's modulus (E) - from AbstractMaterial
- Poisson's ratio (ν) - from AbstractMaterial
- Thickness (t) - stored in formulation

**References:**
- Batoz et al. (1980) - Original DKT formulation
- Lucena Neto et al. (2017) - Geometric stiffness matrix
"""
struct DKTFormulation <: AbstractPlateFormulation
    thickness::Float64
    geometric_stiffness::Bool
end

"""
    DKTFormulation(; thickness, geometric_stiffness=true)

Create DKT formulation with specified plate thickness.

# Arguments
- `thickness::Float64`: Plate thickness [m]
- `geometric_stiffness::Bool`: Enable geometric stiffness for buckling analysis (default: true)

# Example
```julia
dkt = DKTFormulation(thickness=0.01)  # 10mm plate
```
"""
DKTFormulation(; thickness::Real, geometric_stiffness::Bool=true) =
    DKTFormulation(Float64(thickness), geometric_stiffness)

"""
    get_thickness(formulation::DKTFormulation) -> Float64

Get plate thickness from DKT formulation.
"""
get_thickness(f::DKTFormulation) = f.thickness

# ============================================================================
# Material Properties for Plates
# ============================================================================

"""
    constitutive_matrix_plate(material::AbstractMaterial, thickness::Float64)

Compute bending stiffness matrix D for Kirchhoff plate theory.

Extracts Young's modulus E and Poisson's ratio ν from the material,
then computes the 3×3 bending stiffness matrix.

Returns 3×3 matrix relating curvatures to moments:
```
[Mx]   [D11 D12  0 ] [κx ]
[My] = [D21 D22  0 ] [κy ]
[Mxy]  [ 0   0  D33] [κxy]
```

where D₀ = (E*t³)/(12*(1-ν²))

# Arguments
- `material::AbstractMaterial`: Material with E and ν properties
- `thickness::Float64`: Plate thickness [m]

# Returns
- `Tensor{2,3,Float64}`: 3×3 bending stiffness matrix [Pa·m³]

# Example
```julia
steel = LinearElastic(E=210e9, ν=0.3)
t = 0.01  # 10mm
D = constitutive_matrix_plate(steel, t)
```
"""
function constitutive_matrix_plate(material::AbstractMaterial, thickness::Float64)
    E = material.E
    ν = material.ν
    D0 = E * thickness^3 / (12 * (1 - ν^2))
    return Tensor{2,3}((
        D0, D0 * ν, 0.0,
        D0 * ν, D0, 0.0,
        0.0, 0.0, D0 * (1 - ν) / 2
    ))
end

# ============================================================================
# Displacement Field
# ============================================================================

"""
    PlateDisplacement <: AbstractField

Displacement field for plate bending: w, θx, θy at each node.

For a 3-node triangle:
- 9 total DOFs: (w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3)

Convention:
- w: positive upward (out-of-plane)
- θx: rotation about x-axis (right-hand rule)
- θy: rotation about y-axis (right-hand rule)
"""
struct PlateDisplacement <: AbstractField end

# ============================================================================
# API Functions (to be implemented by formulations)
# ============================================================================

"""
    assemble!(physics::Physics{F, PlateDisplacement, M, Mat}) where {F<:AbstractPlateFormulation, M<:AbstractMesh, Mat<:AbstractMaterial}

Assemble plate element stiffness matrices into global system.

This is the main assembly function following the Physics interface pattern.
Each plate formulation must implement this method.

The material (Mat) should provide:
- `E::Float64`: Young's modulus (via `material.E`)
- `ν::Float64`: Poisson's ratio (via `material.ν`)

The formulation (F) provides:
- `thickness::Float64`: Plate thickness

# Example
```julia
steel = LinearElastic(E=210e9, ν=0.3)
dkt = DKTFormulation(thickness=0.01)
physics = Physics(dkt, PlateDisplacement(), mesh, steel)
assemble!(physics)
```
"""
function assemble! end

# Export types and functions
export AbstractPlateFormulation, DKTFormulation
export PlateDisplacement
export constitutive_matrix_plate
export get_thickness
