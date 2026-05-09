# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Primal **Darcy flow potential** on a conforming scalar vertex field.

Steady saturated flow with hydraulic head `p` obeys

``-\\nabla \\cdot (K \\nabla p) = f``

(weak form identical to steady heat conduction with conductivity `K`).
[`DarcyPotentialKernel`](@ref) is a thin constructor around [`HeatKernel`](@ref)
with [`HydraulicConductivity`](@ref) and [`PressurePotential`](@ref).

Mixed lowest-order H(div) flux + cell pressure on Tet4 lives in
[`DarcyMixedRT0P0Kernel`](@ref) (`mixed_rt0.jl`). Transient storage is still future work.
"""

using ..JuliaFEM: ContinuumFormulation, HeatKernel, HydraulicConductivity,
    ElementWiseScalarDiffusion, PressurePotential

"""
    DarcyPotentialKernel(formulation, material[, field])

Return [`HeatKernel`](@ref)`{…, HydraulicConductivity, PressurePotential}` for
steady primal Darcy: same assembly path as thermal diffusion with tensor
[`hydraulic_conductivity_tensor`](@ref).

# Example

```julia
kernel = DarcyPotentialKernel(
    ContinuumFormulation{FullThreeD}(),
    HydraulicConductivity(K = 1e-4),
)
```
"""
function DarcyPotentialKernel(
    formulation::ContinuumFormulation{Theory},
    material::HydraulicConductivity,
    field::PressurePotential = PressurePotential(),
) where {Theory}
    return HeatKernel(formulation, material, field; heat_capacity = 0.0)
end

function DarcyPotentialKernel(
    formulation::ContinuumFormulation{Theory},
    material::ElementWiseScalarDiffusion,
    field::PressurePotential = PressurePotential(),
) where {Theory}
    return HeatKernel(formulation, material, field; heat_capacity = 0.0)
end
