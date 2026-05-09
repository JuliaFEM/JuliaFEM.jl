# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    ElementWiseScalarDiffusion <: AbstractMaterial

Piecewise **constant** isotropic scalar conductivity on each volume element
(index matches `mesh.connectivity` order): at quadrature points of element `e`
the tensor is `λ_by_elem[e] · I`.

Used with [`HeatKernel`](@ref) and either [`Temperature`](@ref) (thermal
conductivity values in ``k``) or [`PressurePotential`](@ref) (hydraulic
conductivity ``K``). For a spatially smooth ``λ(x)``, refine the mesh or use
per-region assembly; truly pointwise ``λ`` at quadrature nodes is not covered
here.

# Example

```julia
# Two bricks along x with different K; λ_by_elem[i] is conductivity of element i.
mat = ElementWiseScalarDiffusion([1.0, 4.0])
kernel = HeatKernel(ContinuumFormulation{FullThreeD}(), mat, PressurePotential())
```
"""

struct ElementWiseScalarDiffusion <: AbstractMaterial
    λ_by_elem::Vector{Float64}
end

material_behavior(::ElementWiseScalarDiffusion) = StatelessConstantTangent()
supported_physics(::ElementWiseScalarDiffusion) = (Thermal{3}(),)
required_state_variables(::ElementWiseScalarDiffusion) = ()
