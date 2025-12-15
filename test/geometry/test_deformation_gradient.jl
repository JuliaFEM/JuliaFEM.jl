# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
# Deformation Gradient Tests (test/geometry/)

## What
Tests computation of the deformation gradient **F = ∂x/∂X = I + ∂u/∂X**, which maps
from reference (undeformed) to current (deformed) configuration in finite strain mechanics.

## Why
The deformation gradient is **FUNDAMENTAL** to finite strain analysis:
- **Green-Lagrange strain**: E = ½(F^T F - I)
- **Cauchy-Green deformation tensor**: C = F^T F
- **Volume ratio**: J = det(F) (incompressibility requires J = 1)
- **Polar decomposition**: F = RU (rotation × stretch)
- **Material frame**: All stress/strain in reference configuration

Physical requirements:
- **det(F) > 0**: No material inversion (orientation preserved)
- **det(F) ≈ 1**: Nearly incompressible materials (rubber, metal plasticity)
- **F = I**: Undeformed configuration (u = 0)

This test validates:
- **Identity case**: u = 0 → F = I, det(F) = 1
- **Pure translation**: ∇u = 0 → F = I (rigid body motion)
- **Pure stretch**: Diagonal F with stretch ratios
- **Simple shear**: Off-diagonal F terms
- **Small vs finite strain**: Difference in formulations
- **Physical constraints**: det(F) > 0 always
- **Zero allocations**: Hot path allocates nothing

## How
**Test Cases:**

**1. Identity (u = 0)**:
- Unit cube Hex8 element, zero displacement
- Expected: F = I, det(F) = 1.0
- Both finite strain and small strain give same result

**2. Pure Translation**:
- Uniform displacement u = (0.5, 0.5, 0.5) at all nodes
- Expected: ∇u = 0 → F = I + 0 = I
- Validates that rigid body motion doesn't deform material

**3. Pure Stretch (x-direction)**:
- Displacement u_x = 0.1·X (10% engineering strain)
- Expected: F = diag(1.1, 1.0, 1.0), det(F) = 1.1
- Validates uniaxial extension

**4. Simple Shear**:
- Displacement u_x = 0.1·y (shear deformation)
- Expected: F_{12} = 0.1, det(F) = 1.0 (volume preserving)
- Validates shear kinematics

**5. Small vs Finite Strain Difference**:
- 20% stretch: significant displacement gradient
- **Finite strain**: F = I + ∇u (includes gradient)
- **Small strain**: F = I (ignores gradient, approximation)
- Validates that formulations differ for large deformations

**6. Physical Constraint**:
- All physical deformations must have det(F) > 0
- Negative det(F) → inverted element (unphysical)

**7. Tet10 (Quadratic) Elements**:
- Higher-order elements with 10 nodes
- Same tests as Hex8 but with quadratic basis
- Validates that API works for all element types

**8. Zero Allocations**:
- `@allocated compute_deformation_gradient(...)` must return 0
- Critical for performance in assembly loops

## Expected Results
- ✅ **Identity**: F = I, det(F) = 1.0
- ✅ **Translation**: F = I (∇u = 0)
- ✅ **Stretch**: F_{ii} = 1 + ε_{ii}, det(F) = product of stretches
- ✅ **Shear**: Off-diagonal terms non-zero, det(F) = 1.0
- ✅ **Finite ≠ Small**: Different F for large deformations (20%+)
- ✅ **Physical**: det(F) > 0 always
- ✅ **Tet10**: Works with quadratic elements
- ✅ **Zero allocations**: @allocated = 0

## Mathematical Background
**Deformation Gradient (Finite Strain):**
```
F = ∂x/∂X = I + ∂u/∂X
```

Where:
- x = X + u (current position = reference + displacement)
- X = reference coordinates
- u = displacement vector
- ∂u/∂X = displacement gradient

**Small Strain Approximation:**
```
F ≈ I   (ignores ∂u/∂X, valid for ||∇u|| << 1)
```

**Computation:**
```julia
# Displacement gradient
∇u = ∑ᵢ uᵢ ⊗ (∂Nᵢ/∂X)

# Physical derivatives
∂Nᵢ/∂X = J⁻¹ · ∂Nᵢ/∂ξ

# Deformation gradient
F = I + ∇u
```

## Formulation Comparison
**Finite Strain** (use when ||∇u|| > 0.01):
- F = I + ∇u (full nonlinear kinematics)
- E = ½(F^T F - I) (Green-Lagrange strain)
- S = ∂W/∂E (2nd Piola-Kirchhoff stress)
- Required for: Rubber, large rotations, metal forming

**Small Strain** (use when ||∇u|| < 0.01):
- F ≈ I (linear approximation)
- ε = ½(∇u + ∇u^T) (engineering strain)
- σ = C:ε (Cauchy stress)
- Valid for: Linear elasticity, small vibrations

## Architecture Principle
**Tensors.jl for Kinematics**

All kinematic quantities use Tensors.jl:
- Displacement: Vec{3,Float64}
- Gradient: Tensor{2,3} (⊗ outer product)
- Deformation gradient: Tensor{2,3}
- Identity: one(Tensor{2,3})

This provides:
- Natural tensor notation (F[i,j])
- Automatic differentiation ready
- Zero-allocation operations
- GPU compatible

## Usage Pattern
```julia
# In assembly loop:
for ip in integration_points
    # Get basis derivatives
    dN_dξ = get_basis_derivatives(topology, basis, ip.ξ)
    
    # Compute Jacobian
    J = compute_jacobian(X_nodes, dN_dξ)
    
    # Compute deformation gradient
    F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
    
    # Compute strain
    E = 0.5 * (F' ⊡ F - I)
    
    # Material model
    S, 𝔻, state = compute_stress(material, E, state_old, Δt)
    
    # ... assemble
end
```

## Critical Performance Path
Deformation gradient is computed at EVERY integration point, EVERY Newton iteration.
For 100k elements × 8 integration points × 10 Newton iterations = 8M calls!

Zero allocations are MANDATORY!
"""

using Test
using Tensors
using LinearAlgebra

# Use JuliaFEM for basis functions
using JuliaFEM

# Load our new deformation gradient code
include("../src/physics/deformation_gradient.jl")

@testset "Deformation Gradient - Low Level API" begin

    @testset "Identity case (u = 0)" begin
        # Unit cube element, no displacement
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Zero displacement
        u_nodes = tuple([zero(Vec{3,Float64}) for _ in 1:8]...)

        # At element center ξ = (0, 0, 0)
        ξ = Vec(0.0, 0.0, 0.0)

        # Hex8 basis function derivatives at center (new API)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        # Compute Jacobian
        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Finite strain: Should give F = I + 0 = I
        F_finite = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        @test F_finite ≈ one(Tensor{2,3})
        @test det(F_finite) ≈ 1.0

        # Small strain: Should also give F = I
        F_small = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, SmallStrain())
        @test F_small ≈ one(Tensor{2,3})
        @test det(F_small) ≈ 1.0
    end

    @testset "Pure translation" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Uniform translation: u = (0.5, 0.5, 0.5) everywhere
        u_const = Vec(0.5, 0.5, 0.5)
        u_nodes = tuple([u_const for _ in 1:8]...)

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Pure translation ⇒ ∇u = 0 ⇒ F = I
        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        @test F ≈ one(Tensor{2,3})
        @test det(F) ≈ 1.0
    end

    @testset "Pure stretch in x-direction" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Stretch: u_x = 0.1 * X (10% stretch in x)
        u_nodes = (
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.0, 0.0, 0.0),  # u = 0.1 * 0 = 0
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.1, 0.0, 0.0),  # u = 0.1 * 1 = 0.1
            Vec(0.0, 0.0, 0.0)   # u = 0.1 * 0 = 0
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F = [1.1  0  0]
        #               [0    1  0]
        #               [0    0  1]
        @test F[1, 1] ≈ 1.1 atol = 1e-10
        @test F[2, 2] ≈ 1.0 atol = 1e-10
        @test F[3, 3] ≈ 1.0 atol = 1e-10
        @test F[1, 2] ≈ 0.0 atol = 1e-10
        @test F[1, 3] ≈ 0.0 atol = 1e-10
        @test F[2, 3] ≈ 0.0 atol = 1e-10
        @test det(F) ≈ 1.1 atol = 1e-10
    end

    @testset "Simple shear" begin
        # Unit cube
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Shear: u_x = 0.1 * y
        u_nodes = (
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.0, 0.0, 0.0),   # y=0
            Vec(0.1, 0.0, 0.0),   # y=1
            Vec(0.1, 0.0, 0.0)    # y=1
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F = [1    0.1  0]
        #               [0    1    0]
        #               [0    0    1]
        @test F[1, 1] ≈ 1.0 atol = 1e-10
        @test F[1, 2] ≈ 0.1 atol = 1e-10
        @test F[2, 2] ≈ 1.0 atol = 1e-10
        @test F[3, 3] ≈ 1.0 atol = 1e-10
        @test det(F) ≈ 1.0 atol = 1e-10
    end

    @testset "Small vs Finite strain difference" begin
        # Setup with significant displacement gradient
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # 20% stretch in x
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.2, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F_finite = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())
        F_small = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, SmallStrain())

        # Finite strain includes gradient
        @test F_finite[1, 1] ≈ 1.2 atol = 1e-10

        # Small strain ignores gradient
        @test F_small[1, 1] ≈ 1.0 atol = 1e-10

        # They should be different!
        @test !(F_finite ≈ F_small)
    end

    @testset "Physical constraint: det(F) > 0" begin
        # Physical deformation must preserve orientation
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        # Small positive stretch
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test det(F) > 0  # Physical requirement
    end
end

@testset "Deformation Gradient - Tet10 Element" begin

    @testset "Tet10: Identity case" begin
        # Regular tetrahedron nodes (4 corners + 6 edge midpoints)
        X_nodes = (
            Vec(0.0, 0.0, 0.0),              # 1: corner
            Vec(1.0, 0.0, 0.0),              # 2: corner
            Vec(0.0, 1.0, 0.0),              # 3: corner
            Vec(0.0, 0.0, 1.0),              # 4: corner
            Vec(0.5, 0.0, 0.0),              # 5: edge 1-2
            Vec(0.5, 0.5, 0.0),              # 6: edge 2-3
            Vec(0.0, 0.5, 0.0),              # 7: edge 3-1
            Vec(0.0, 0.0, 0.5),              # 8: edge 1-4
            Vec(0.5, 0.0, 0.5),              # 9: edge 2-4
            Vec(0.0, 0.5, 0.5)               # 10: edge 3-4
        )

        # Zero displacement
        u_nodes = tuple([zero(Vec{3,Float64}) for _ in 1:10]...)

        # At element centroid ξ = (1/4, 1/4, 1/4)
        ξ = Vec(0.25, 0.25, 0.25)

        # Tet10 basis function derivatives (new API)
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,2}(), ξ)

        # Compute Jacobian
        J = zero(Tensor{2,3,Float64,9})
        for i in 1:10
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test F ≈ one(Tensor{2,3}) atol = 1e-10
        @test det(F) ≈ 1.0 atol = 1e-10
    end

    @testset "Tet10: Uniform stretch" begin
        # Regular tetrahedron
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(0.5, 0.0, 0.0),
            Vec(0.5, 0.5, 0.0),
            Vec(0.0, 0.5, 0.0),
            Vec(0.0, 0.0, 0.5),
            Vec(0.5, 0.0, 0.5),
            Vec(0.0, 0.5, 0.5)
        )

        # Isotropic expansion: u = 0.1 * X
        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.1, 0.0),
            Vec(0.0, 0.0, 0.1),
            Vec(0.05, 0.0, 0.0),
            Vec(0.05, 0.05, 0.0),
            Vec(0.0, 0.05, 0.0),
            Vec(0.0, 0.0, 0.05),
            Vec(0.05, 0.0, 0.05),
            Vec(0.0, 0.05, 0.05)
        )

        ξ = Vec(0.25, 0.25, 0.25)
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,2}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:10
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Expected: F ≈ 1.1 * I
        @test F[1, 1] ≈ 1.1 atol = 1e-10
        @test F[2, 2] ≈ 1.1 atol = 1e-10
        @test F[3, 3] ≈ 1.1 atol = 1e-10
        @test abs(F[1, 2]) < 1e-10
        @test abs(F[1, 3]) < 1e-10
        @test abs(F[2, 3]) < 1e-10
        @test det(F) ≈ 1.1^3 atol = 1e-10
    end
end

@testset "Deformation Gradient - Zero Allocation" begin

    @testset "Verify zero allocations" begin
        # Setup
        X_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(1.0, 0.0, 0.0),
            Vec(1.0, 1.0, 0.0),
            Vec(0.0, 1.0, 0.0),
            Vec(0.0, 0.0, 1.0),
            Vec(1.0, 0.0, 1.0),
            Vec(1.0, 1.0, 1.0),
            Vec(0.0, 1.0, 1.0)
        )

        u_nodes = (
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.1, 0.0, 0.0),
            Vec(0.0, 0.0, 0.0)
        )

        ξ = Vec(0.0, 0.0, 0.0)
        dN_dξ = get_basis_derivatives(Hexahedron(), Lagrange{Hexahedron,1}(), ξ)

        J = zero(Tensor{2,3,Float64,9})
        for i in 1:8
            J += X_nodes[i] ⊗ dN_dξ[i]
        end

        # Warm up (compile)
        F = compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        # Measure allocations
        allocs = @allocated compute_deformation_gradient(X_nodes, u_nodes, dN_dξ, J, FiniteStrain())

        @test allocs == 0  # Zero allocations!
    end
end
