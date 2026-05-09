# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Linear isotropic hydraulic conductivity for the **primal** flow potential
(Darcy) equation in steady state,

    q = -K · ∇p,    ∇ · q = f   ⇒   -∇ · (K · ∇p) = f,

with scalar potential `p` (pressure head) discretized like temperature.
The weak stiffness `∫ ∇N_i · K · ∇N_j dV` is assembled by [`HeatKernel`](@ref)
via [`DarcyPotentialKernel`](@ref); see `src/domains/darcy/potential.jl`.

For mixed RT₀–P₀ Darcy ([`DarcyMixedRT0P0Kernel`](@ref), [`DarcyMixedHex8RT0P0Kernel`](@ref)),
[`HydraulicConductivity`](@ref) supplies isotropic `K` → `inv_K = (1/K) I`; pass a symmetric tensor
directly to those kernels for anisotropic inverse conductivity. This struct still supplies `K`
for the primal potential path via [`DarcyPotentialKernel`](@ref).

# Fields
- `K::Float64` — isotropic hydraulic conductivity [m/s]; must be positive.

# Example
```julia
soil = HydraulicConductivity(K = 1e-4)
kernel = DarcyPotentialKernel(ContinuumFormulation{FullThreeD}(), soil)
```
"""

using Tensors

struct HydraulicConductivity <: AbstractMaterial
    K::Float64

    function HydraulicConductivity(K::Float64)
        K > 0.0 || throw(ArgumentError("Hydraulic conductivity K must be positive, got K = $K"))
        new(K)
    end
end

HydraulicConductivity(; K) = HydraulicConductivity(Float64(K))

material_behavior(::HydraulicConductivity) = StatelessConstantTangent()
supported_physics(::HydraulicConductivity) = (Thermal{3}(),)
required_state_variables(::HydraulicConductivity) = ()

"""
    hydraulic_conductivity_tensor(material::HydraulicConductivity)

Isotropic tensor `K · I` used at quadrature points (same storage pattern as
[`conductivity_tensor`](@ref) for [`HeatConductivity`](@ref)).
"""
@inline function hydraulic_conductivity_tensor(material::HydraulicConductivity)
    return material.K * one(SymmetricTensor{2,3,Float64,6})
end

@inline scalar_diffusion_tensor(material::HydraulicConductivity) =
    hydraulic_conductivity_tensor(material)

"""
    compute_seepage_flux(material::HydraulicConductivity, ∇p, state_old, Δt)
        -> (q, Ktensor, state_new)

Darcy analogue of [`compute_heat_flux`](@ref): `q = -K · ∇p`, tangent `Ktensor`
is `∂q/∂(∇p)` up to the same sign convention as the heat routine.
"""
function compute_seepage_flux(
    material::HydraulicConductivity,
    ∇p::Vec{3,T},
    state_old::Union{Nothing, NamedTuple},
    Δt::Float64,
) where {T}
    Ktensor = hydraulic_conductivity_tensor(material)
    q = -Ktensor ⋅ ∇p
    return q, Ktensor, NamedTuple()
end

compute_seepage_flux(material::HydraulicConductivity, ∇p::Vec{3,T}) where {T} =
    compute_seepage_flux(material, ∇p, nothing, 0.0)
