# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Linear elastic response on the mechanical strain `ε − ε_e`, where the eigenstrain
`ε_e` (e.g. drying shrinkage) is stored per integration point and updated outside this
routine—typically after a moisture diffusion solve.
"""

using Tensors

struct LinearElasticWithEigenstrain <: AbstractElasticMaterial
    elastic::LinearElastic
end

LinearElasticWithEigenstrain(; E::Real, ν::Real) = LinearElasticWithEigenstrain(LinearElastic(E = E, ν = ν))

material_behavior(::LinearElasticWithEigenstrain) = StatefulStrainDependent()
supported_physics(::LinearElasticWithEigenstrain) = (Elasticity{3}(),)
required_state_variables(::LinearElasticWithEigenstrain) = (Eigenstrain,)

function compute_stress(
    mat::LinearElasticWithEigenstrain,
    ε::SymmetricTensor{2,3},
    ::Nothing,
    Δt::Float64,
)
    return compute_stress(mat, ε, NamedTuple(), Δt)
end

function compute_stress(
    mat::LinearElasticWithEigenstrain,
    ε::SymmetricTensor{2,3},
    state_old::NamedTuple,
    Δt::Float64,
)
    ε_e = get(state_old, :ε_e, zero(SymmetricTensor{2,3}))
    σ, 𝔻, _ = compute_stress(mat.elastic, ε - ε_e, NamedTuple(), 0.0)
    return σ, 𝔻, (ε_e=ε_e,)
end
