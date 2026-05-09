# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Isotropic moisture diffusivity `D_w` for a scalar humidity/pore-water field
[`MoistureContent`](@ref). The weak operator matches steady heat conduction
(`HeatKernel` with [`Thermal`](@ref) physics).
"""

using Tensors

struct MoistureDiffusivity <: AbstractMaterial
    D_w::Float64

    function MoistureDiffusivity(D_w::Float64)
        D_w > 0 || throw(ArgumentError("Moisture diffusivity D_w must be positive"))
        new(D_w)
    end
end

MoistureDiffusivity(; D_w::Real) = MoistureDiffusivity(Float64(D_w))

material_behavior(::MoistureDiffusivity) = StatelessConstantTangent()
supported_physics(::MoistureDiffusivity) = (Thermal{3}(),)

@inline conductivity_tensor(m::MoistureDiffusivity) = m.D_w * one(SymmetricTensor{2,3,Float64,6})
@inline scalar_diffusion_tensor(m::MoistureDiffusivity) = conductivity_tensor(m)
