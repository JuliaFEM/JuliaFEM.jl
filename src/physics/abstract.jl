# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractPhysics

Marker abstract type for physics category tags.

In the default 0.x build this is used purely for trait-based dispatch on lightweight
type tags such as `Elasticity{Dim}` and `Thermal{Dim}` (see
`src/physics/types.jl`), which materials use to declare what they support
(`supported_physics`, `required_field_type`, `required_field_types`). It is
not the parent type of any concrete `Physics{...}` struct in the active
codebase. The older `Physics{Formulation, Field, Mesh, Material}` struct
lives in `JuliaFEM.Legacy` and is loaded only when the user opts in via
`JULIAFEM_ENABLE_LEGACY=1`.
"""
abstract type AbstractPhysics end
