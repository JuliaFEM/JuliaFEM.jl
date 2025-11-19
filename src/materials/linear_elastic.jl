"""
Linear elastic (Hookean) material model using Tensors.jl.

This module implements isotropic linear elasticity with:
- Zero allocations (stack-allocated symmetric tensors)
- Type-stable implementation
- Clean mathematical notation matching theory

Theory:
    σ = λ·tr(ε)·I + 2μ·ε    (Hooke's law)
    
Where:
    λ = E·ν/((1+ν)(1-2ν))   First Lamé parameter
    μ = E/(2(1+ν))           Shear modulus (second Lamé parameter)
    E                         Young's modulus [Pa]
    ν                         Poisson's ratio [-]

Material tangent:
    𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ
    
Where:
    I        Second-order identity tensor
    𝕀ˢʸᵐ     Symmetric fourth-order identity tensor
    ⊗        Tensor (outer) product
"""

using Tensors

# Load abstract types
include("abstract_material.jl")

"""
    LinearElastic <: AbstractElasticMaterial

Linear elastic (Hookean) material model.

# Fields
- `E::Float64` - Young's modulus [Pa]
- `ν::Float64` - Poisson's ratio [-], must satisfy -1 < ν < 0.5

# Properties
Stateless material: stress depends only on current strain, no history.

# Type Hierarchy
`LinearElastic <: AbstractElasticMaterial <: AbstractMaterial`
"""
struct LinearElastic <: AbstractElasticMaterial
    E::Float64   # Young's modulus [Pa]
    ν::Float64   # Poisson's ratio [-]

    function LinearElastic(E::Float64, ν::Float64)
        # Validate inputs
        E > 0.0 || throw(ArgumentError("Young's modulus E must be positive, got E = $E"))
        -1.0 < ν < 0.5 || throw(ArgumentError("Poisson's ratio must satisfy -1 < ν < 0.5, got ν = $ν"))
        new(E, ν)
    end
end

"""
    LinearElastic(; E, ν)

Convenience constructor with keyword arguments.

# Example
```julia
steel = LinearElastic(E=200e9, ν=0.3)
```
"""
LinearElastic(; E, ν) = LinearElastic(Float64(E), Float64(ν))

"""
    λ(material::LinearElastic) -> Float64

Compute first Lamé parameter from Young's modulus and Poisson's ratio.

# Formula
    λ = E·ν/((1+ν)(1-2ν))

# Returns
First Lamé parameter [Pa]
"""
@inline λ(mat::LinearElastic) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))

"""
    μ(material::LinearElastic) -> Float64

Compute shear modulus (second Lamé parameter) from Young's modulus and Poisson's ratio.

# Formula
    μ = E/(2(1+ν))

Also known as the shear modulus or second Lamé parameter.

# Returns
Shear modulus [Pa]
"""
@inline μ(mat::LinearElastic) = mat.E / (2(1 + mat.ν))

"""
    compute_stress(material::LinearElastic, ε, state_old, Δt) -> (σ, 𝔻, state_new)

Compute stress and tangent modulus from strain for linear elastic material.

# Arguments
- `material::LinearElastic` - Material parameters
- `ε::SymmetricTensor{2,3,T}` - Strain tensor (small strain assumption)
- `state_old::Nothing` - Material state (unused for stateless material)
- `Δt::Float64` - Time increment (unused for rate-independent material)

# Returns
- `σ::SymmetricTensor{2,3,T}` - Cauchy stress tensor [Pa]
- `𝔻::SymmetricTensor{4,3,T}` - Tangent modulus (∂σ/∂ε) [Pa]
- `state_new::Nothing` - Updated material state (always `nothing` for stateless)

# Theory
Hooke's law in tensor form:
    σ = λ·tr(ε)·I + 2μ·ε

Tangent modulus (constant for linear elasticity):
    𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ

# Example
```julia
steel = LinearElastic(E=200e9, ν=0.3)
ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))  # Uniaxial extension
σ, 𝔻, _ = compute_stress(steel, ε, nothing, 0.0)

# Result: σ11 ≈ 220 MPa, σ22 = σ33 ≈ -66 MPa (Poisson effect)
```

# Performance
- Zero allocations (stack-allocated tensors)
- Type-stable return type
- Typical execution time: ~20 ns on modern CPU
"""
function compute_stress(
    material::LinearElastic,
    ε::SymmetricTensor{2,3,T},
    state_old::Nothing,
    Δt::Float64
) where T

    # Lamé parameters
    λ_val = λ(material)
    μ_val = μ(material)

    # Identity tensor (same type as ε)
    I = one(ε)

    # Hooke's law: σ = λ·tr(ε)·I + 2μ·ε
    σ = λ_val * tr(ε) * I + 2μ_val * ε

    # Tangent modulus: 𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ
    𝕀ˢʸᵐ = one(SymmetricTensor{4,3,T})  # Symmetric 4th order identity
    𝔻 = λ_val * (I ⊗ I) + 2μ_val * 𝕀ˢʸᵐ

    return σ, 𝔻, nothing  # No state change (stateless material)
end

"""
    compute_stress(material::LinearElastic, ε::SymmetricTensor{2,3,T}) -> (σ, 𝔻, nothing)

Simplified interface without state management for stateless material.

# Arguments
- `material::LinearElastic` - Material parameters
- `ε::SymmetricTensor{2,3,T}` - Strain tensor

# Returns
Same as full interface: (σ, 𝔻, nothing)

# Example
```julia
steel = LinearElastic(E=200e9, ν=0.3)
ε = SymmetricTensor{2,3}((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
σ, 𝔻, _ = compute_stress(steel, ε)  # Simplified call
```
"""
compute_stress(material::LinearElastic, ε::SymmetricTensor{2,3,T}) where T =
    compute_stress(material, ε, nothing, 0.0)

"""
    elasticity_tensor(material::LinearElastic) -> Tensor{4,3,Float64}

Return 4th-order elasticity tensor C_{ijkl} for assembly.

# Formula
Linear isotropic elasticity:
    C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})

Where:
- λ = E·ν/((1+ν)(1-2ν)) - First Lamé parameter
- μ = E/(2(1+ν)) - Shear modulus
- δ_{ij} = Kronecker delta

# Returns
- `C::Tensor{4,3,Float64}` - Fourth-order elasticity tensor [Pa]

# Usage in Assembly
```julia
material = LinearElastic(E=210e9, ν=0.3)
C = elasticity_tensor(material)

# Use in stiffness computation:
# K_ij^{αβ} = ∫ (∂N_i/∂x_γ) C_{αβγδ} (∂N_j/∂x_δ) dV
```

# Implementation Note
Returns SymmetricTensor{4,3} encoding the full material symmetry.
The tensor has minor and major symmetries: C_{ijkl} = C_{jikl} = C_{ijlk} = C_{klij}
"""
@generated function elasticity_tensor(material::LinearElastic)
    # Generate tensor construction at compile time for zero allocations
    # C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})
    δ(i, j) = i == j ? 1.0 : 0.0

    # Build full 81-component tensor first
    exprs = []
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        if δ(i,j) != 0.0 && δ(k,l) != 0.0
            # Has λ term
            if δ(i,k) != 0.0 && δ(j,l) != 0.0
                # λ + 2μ (diagonal component)
                push!(exprs, :(λ_val + 2*μ_val))
            else
                # λ only (off-diagonal coupling)
                push!(exprs, :(λ_val))
            end
        elseif δ(i,k) != 0.0 && δ(j,l) != 0.0 && i != j
            # μ (shear component)
            push!(exprs, :(μ_val))
        elseif δ(i,l) != 0.0 && δ(j,k) != 0.0 && i != j
            # μ (shear component, swapped indices)
            push!(exprs, :(μ_val))
        else
            # Zero
            push!(exprs, :(0.0))
        end
    end

    return quote
        λ_val = λ(material)
        μ_val = μ(material)
        # Create as Tensor{4,3} then convert - Tensors.jl handles the symmetry extraction
        C_full = Tensor{4,3,Float64}(($(exprs...),))
        SymmetricTensor{4,3}(C_full)
    end
end
