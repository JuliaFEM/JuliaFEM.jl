# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Linear isotropic heat-conductivity material model.

Mirror image of `LinearElastic` for the heat-conduction physics. The
constitutive law is Fourier's law

    q = -k · ∇T,        ∂q/∂(∇T) = -k · I

so the per-IP "tangent" we cache is the (constant, isotropic) symmetric
2nd-order tensor `k * I`. Anisotropic / temperature-dependent
conductivities are intentionally out of scope; they reuse the same
microkernel contract by replacing this struct.
"""

using Tensors

"""
    HeatConductivity <: AbstractMaterial

Linear isotropic heat-conductivity material.

# Fields
- `k::Float64` — thermal conductivity [W / (m·K)]; must be positive.

# Example
```julia
copper = HeatConductivity(k = 401.0)
steel  = HeatConductivity(k = 50.2)
```
"""
struct HeatConductivity <: AbstractMaterial
    k::Float64

    function HeatConductivity(k::Float64)
        k > 0.0 || throw(ArgumentError("Thermal conductivity k must be positive, got k = $k"))
        new(k)
    end
end

"""
    HeatConductivity(; k)

Convenience constructor with keyword argument.
"""
HeatConductivity(; k) = HeatConductivity(Float64(k))

# ---------- Trait declarations ----------------------------------------------

material_behavior(::HeatConductivity) = StatelessConstantTangent()
supported_physics(::HeatConductivity) = (Thermal{3}(),)
required_state_variables(::HeatConductivity) = ()

# ---------- Constitutive law ------------------------------------------------

"""
    conductivity_tensor(material::HeatConductivity) -> SymmetricTensor{2,3,Float64,6}

Return the (constant, isotropic) conductivity 2nd-order tensor `k · I`.
This is what each IP sees through the heat microkernel buffer.
"""
@inline function conductivity_tensor(material::HeatConductivity)
    return material.k * one(SymmetricTensor{2,3,Float64,6})
end

"""
    scalar_diffusion_tensor(material::HeatConductivity)

Symmetric positive-definite tensor `k` in the weak form
`∫ ∇v · k · ∇u dV` used by [`HeatKernel`](@ref). Identical to
[`conductivity_tensor`](@ref); the name is shared with
[`HydraulicConductivity`](@ref) for primal flow-potential problems.
"""
@inline scalar_diffusion_tensor(material::HeatConductivity) = conductivity_tensor(material)

"""
    compute_heat_flux(material::HeatConductivity, ∇T, state_old, Δt)
        -> (q, K, state_new)

Heat-conduction analogue of `compute_stress`:

* `material::HeatConductivity` — material model
* `∇T::Vec{3,Float64}`         — temperature gradient at the current IP
* `state_old`                  — previous-step state (`nothing` /
  `NamedTuple()` for stateless conductivity)
* `Δt::Float64`                — time increment

Returns `(q, K, state_new)` where

* `q::Vec{3,Float64}`                          — heat flux `q = -k·∇T`
* `K::SymmetricTensor{2,3,Float64,6}`          — conductivity `k·I`,
  the constant symmetric tangent `∂q/∂(∇T) = -K` (sign convention:
  positive-definite K so the stiffness `Bᵀ K B` is SPD)
* `state_new::NamedTuple`                      — empty NamedTuple
  (stateless material)
"""
function compute_heat_flux(
    material::HeatConductivity,
    ∇T::Vec{3,T},
    state_old::Union{Nothing,NamedTuple},
    Δt::Float64,
) where T
    K = conductivity_tensor(material)
    q = -K ⋅ ∇T
    return q, K, NamedTuple()
end

compute_heat_flux(material::HeatConductivity, ∇T::Vec{3,T}) where T =
    compute_heat_flux(material, ∇T, nothing, 0.0)
