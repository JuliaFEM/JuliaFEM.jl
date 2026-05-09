"""
Linear elastic (Hookean) material model using Tensors.jl.
"""

using Tensors

"""
    LinearElastic <: AbstractElasticMaterial

Linear elastic (Hookean) material model.

# Fields
- `E::Float64` - Young's modulus [Pa]
- `ν::Float64` - Poisson's ratio [-], must satisfy -1 < ν < 0.5
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
"""
LinearElastic(; E, ν) = LinearElastic(Float64(E), Float64(ν))

material_behavior(::LinearElastic) = StatelessConstantTangent()
supported_physics(::LinearElastic) = (Elasticity{3}(),)
required_state_variables(::LinearElastic) = ()

"""
    λ(material::LinearElastic) -> Float64

Compute first Lamé parameter: λ = E·ν/((1+ν)(1-2ν))
"""
@inline λ(mat::LinearElastic) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))

"""
    μ(material::LinearElastic) -> Float64

Compute shear modulus: μ = E/(2(1+ν))
"""
@inline μ(mat::LinearElastic) = mat.E / (2(1 + mat.ν))

"""
    compute_stress(material::LinearElastic, ε, state_old, Δt) -> (σ, 𝔻, state_new)

Compute stress and tangent modulus from strain for linear elastic material.

Hooke's law: σ = λ·tr(ε)·I + 2μ·ε
Tangent: 𝔻 = λ·I⊗I + 2μ·𝕀ˢʸᵐ
"""
function compute_stress(
    material::LinearElastic,
    ε::SymmetricTensor{2,3,T},
    state_old::Union{Nothing,NamedTuple},
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
    𝕀ˢʸᵐ = one(SymmetricTensor{4,3,T,36})  # Symmetric 4th order identity
    𝔻 = λ_val * (I ⊗ I) + 2μ_val * 𝕀ˢʸᵐ

    return σ, 𝔻, NamedTuple()  # No state change (stateless material)
end

"""
    compute_stress(material::LinearElastic, ε::SymmetricTensor{2,3,T}) -> (σ, 𝔻, nothing)

Simplified interface without state management for stateless material.
"""
compute_stress(material::LinearElastic, ε::SymmetricTensor{2,3,T}) where T =
    compute_stress(material, ε, nothing, 0.0)

"""
    elasticity_tensor(material::LinearElastic) -> Tensor{4,3,Float64}

Return 4th-order elasticity tensor: C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})
"""
@generated function elasticity_tensor(material::LinearElastic)
    # Generate tensor construction at compile time for zero allocations
    # C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})
    δ(i, j) = i == j ? 1.0 : 0.0

    # Build full 81-component tensor first
    exprs = []
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        if δ(i, j) != 0.0 && δ(k, l) != 0.0
            # Has λ term
            if δ(i, k) != 0.0 && δ(j, l) != 0.0
                # λ + 2μ (diagonal component)
                push!(exprs, :(λ_val + 2 * μ_val))
            else
                # λ only (off-diagonal coupling)
                push!(exprs, :(λ_val))
            end
        elseif δ(i, k) != 0.0 && δ(j, l) != 0.0 && i != j
            # μ (shear component)
            push!(exprs, :(μ_val))
        elseif δ(i, l) != 0.0 && δ(j, k) != 0.0 && i != j
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
        C_full = Tensor{4,3,Float64,81}(($(exprs...),))
        SymmetricTensor{4,3,Float64,36}(C_full)
    end
end
