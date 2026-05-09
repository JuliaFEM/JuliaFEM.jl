# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Continuum kinematics tag for stateful solid materials.

[`update_material_cache!`](@ref) uses [`continuum_kinematics`](@ref) to decide whether
integration-point strain is the symmetric displacement gradient (`SmallStrainKinematics`)
or Green–Lagrange strain from `F` (`GreenLagrangeKinematics`). Hyperelastic models already
use the Green–Lagrange path via [`StatelessStrainDependent`](@ref).

Finite-strain **multiplicative** elastoplasticity is not implemented here; StVK–J₂ uses
the Green–Lagrange measure with a St. Venant–Kirchhoff elastic predictor (see
[`StVenantKirchhoffJ2Plasticity`](@ref)).
"""

abstract type AbstractContinuumKinematics end

"""Symmetric gradient `ε = sym(∇u)` (small-strain assembly)."""
struct SmallStrainKinematics <: AbstractContinuumKinematics end

"""`E = ½(FᵀF − I)` with `F = I + ∇u ⊗ …` (total Lagrangian assembly)."""
struct GreenLagrangeKinematics <: AbstractContinuumKinematics end

"""
    continuum_kinematics(material::AbstractMaterial)

Return [`SmallStrainKinematics`](@ref) by default; materials such as
[`StVenantKirchhoffJ2Plasticity`](@ref) override with [`GreenLagrangeKinematics`](@ref).
"""
continuum_kinematics(::AbstractMaterial) = SmallStrainKinematics()
