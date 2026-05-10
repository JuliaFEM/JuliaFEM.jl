# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Scalar-parameter homogeneity for per-element kernel columns on the KA matvec path.

[`PerElementKernelColumn`](@ref) stores one kernel object per element. The
KernelAbstractions launcher passes a **single prototype** kernel (the first
element's) into device code while per-element data live in `qp_buffers`.
Any extra scalar read from the kernel object (density, heat capacity, Biot
`α`, …) must therefore match across the whole column. Setup-only checks live
here; the hot path uses `kernel_at(col, eid)` on CPU only.
"""

@inline function _assert_homogeneous_ka_column_kernel_scalars!(kernels::Vector{K}) where {K}
    length(kernels) < 2 && return nothing
    @inbounds k1 = kernels[1]
    return _assert_homogeneous_ka_column_kernel_scalars!(k1, kernels)
end

@inline _assert_homogeneous_ka_column_kernel_scalars!(::AbstractKernel, ::Vector) = nothing

function _assert_homogeneous_ka_column_kernel_scalars!(k1::ContinuumKernel, kernels::Vector)
    d0 = k1.density
    for i in 2:length(kernels)
        @inbounds ki = kernels[i]
        ki.density == d0 || throw(ArgumentError(
            "per-element ContinuumKernel: density must match on all elements " *
            "(mass microkernel reads kernel.density); mismatch at element $i",
        ))
    end
    return nothing
end

function _assert_homogeneous_ka_column_kernel_scalars!(k1::HeatKernel, kernels::Vector)
    c0 = k1.heat_capacity
    for i in 2:length(kernels)
        @inbounds ki = kernels[i]
        ki.heat_capacity == c0 || throw(ArgumentError(
            "per-element HeatKernel: heat_capacity must match on all elements; mismatch at element $i",
        ))
    end
    return nothing
end

function _assert_homogeneous_ka_column_kernel_scalars!(k1::ThermoElasticKernel, kernels::Vector)
    β0 = k1.β
    for i in 2:length(kernels)
        @inbounds ki = kernels[i]
        ki.β == β0 || throw(ArgumentError(
            "per-element ThermoElasticKernel: β must match on all elements; mismatch at element $i",
        ))
    end
    return nothing
end

function _assert_homogeneous_ka_column_kernel_scalars!(k1::BiotPoroelasticKernel, kernels::Vector)
    α0 = k1.α
    S0 = k1.storage_S
    ρ0 = k1.density
    for i in 2:length(kernels)
        @inbounds ki = kernels[i]
        ki.α == α0 ||
            throw(ArgumentError("per-element BiotPoroelasticKernel: α mismatch at element $i"))
        ki.storage_S == S0 ||
            throw(ArgumentError("per-element BiotPoroelasticKernel: storage_S mismatch at element $i"))
        ki.density == ρ0 ||
            throw(ArgumentError("per-element BiotPoroelasticKernel: density mismatch at element $i"))
    end
    return nothing
end

function _assert_homogeneous_ka_column_kernel_scalars!(k1::ThermoPoroelasticKernel, kernels::Vector)
    β0 = k1.β
    α0 = k1.α
    S0 = k1.storage_S
    κ0 = k1.kappa_tp
    ζ0 = k1.zeta_tp
    ρcp0 = k1.heat_capacity
    ρs0 = k1.density
    for i in 2:length(kernels)
        @inbounds ki = kernels[i]
        ki.β == β0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: β mismatch at element $i"))
        ki.α == α0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: α mismatch at element $i"))
        ki.storage_S == S0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: storage_S mismatch at element $i"))
        ki.kappa_tp == κ0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: kappa_tp mismatch at element $i"))
        ki.zeta_tp == ζ0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: zeta_tp mismatch at element $i"))
        ki.heat_capacity == ρcp0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: heat_capacity mismatch at element $i"))
        ki.density == ρs0 ||
            throw(ArgumentError("per-element ThermoPoroelasticKernel: density mismatch at element $i"))
    end
    return nothing
end
