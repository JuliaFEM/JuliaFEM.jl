# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
Column of volume kernels for [`DOFBasedCOOCache`](@ref).

Two storage shapes, both type-stable and allocation-free on assembly hot paths:

* [`UniformKernelColumn`](@ref) — one kernel reused for every element (no extra
  storage beyond the struct).
* [`PerElementKernelColumn`](@ref) — one concrete kernel per element, stored in a
  pre-allocated `Vector{K}` built at cache construction time (setup may allocate;
  Pass 1 / Pass 2 only index this vector).

All kernels in a per-element column must share the same `qpoint_buffer_eltype`,
the same `reference_fields` / state NamedTuple types, and the same
`dofs_per_node`. Kernels whose microkernels read additional scalar parameters
from the kernel object (Biot `α`/`storage_S`/`density`, thermo-elastic `β`,
thermo-poroelastic `β`/`α`/`storage_S`/`kappa_tp`/`zeta_tp`/`heat_capacity`,
continuum `density` for mass, heat `heat_capacity`, …) must match those parameters across elements so
that GPU / KA paths that pass a single prototype kernel remain consistent; the
CPU path always uses `kernel_at(col, eid)`. Scalar checks are implemented in
`ka_column_homogeneity.jl` as `_assert_homogeneous_ka_column_kernel_scalars!`. For kernels whose
`evaluate_entry` reads only `qp_buffers` plus those matched scalars
(`ContinuumKernel`, `HeatKernel`, `ThermoElasticKernel`, `BiotPoroelasticKernel`,
`ThermoPoroelasticKernel`), `ka_per_element_kernel_column_supported` allows the
KA matvec path.
=#

"""
    UniformKernelColumn{K<:AbstractKernel}

Store a single instance `kernel::K` used for every volume element. Hot-path
lookup is a direct field read (no indexing).
"""
struct UniformKernelColumn{K<:AbstractKernel}
    kernel::K
end

"""
    PerElementKernelColumn{K<:AbstractKernel}

Store `kernels[eid]` for each volume element id `eid`. The vector is owned for
the lifetime of the cache and filled at construction (no per-assembly
allocation).
"""
struct PerElementKernelColumn{K<:AbstractKernel}
    kernels::Vector{K}
end

@inline kernel_at(col::UniformKernelColumn, ::Int) = col.kernel
@inline kernel_at(col::PerElementKernelColumn, eid::Int) = @inbounds col.kernels[eid]

@inline prototype_kernel(col::UniformKernelColumn) = col.kernel
@inline prototype_kernel(col::PerElementKernelColumn) = @inbounds col.kernels[1]

"""
    assert_homogeneous_dof_based_kernel_column!(kernels::Vector{K}) where {K}

Validate a vector of kernels before wrapping it in [`PerElementKernelColumn`](@ref).
Called only from cache construction (setup tier), not from assembly hot paths.
"""
function assert_homogeneous_dof_based_kernel_column!(kernels::Vector{K}) where {K}
    n = length(kernels)
    n ≥ 1 || throw(ArgumentError("per-element kernels: empty vector"))
    @inbounds k1 = kernels[1]
    if k1 isa HeatKernel && k1.material isa ElementWiseScalarDiffusion
        throw(ArgumentError(
            "per-element kernel column is not supported with ElementWiseScalarDiffusion; " *
            "use a single HeatKernel whose material carries λ_by_elem[elem_id] instead.",
        ))
    end
    fr1, st1 = reference_fields(k1)
    Buf1 = qpoint_buffer_eltype(k1)
    dpn1 = dofs_per_node(k1)
    for i in 2:n
        @inbounds ki = kernels[i]
        dofs_per_node(ki) == dpn1 || throw(ArgumentError(
            "per-element kernels: dofs_per_node mismatch at element $i (" *
            "$(dofs_per_node(ki)) vs $dpn1)",
        ))
        qpoint_buffer_eltype(ki) === Buf1 || throw(ArgumentError(
            "per-element kernels: qpoint_buffer_eltype mismatch at element $i",
        ))
        fri, sti = reference_fields(ki)
        typeof(fri) === typeof(fr1) || throw(ArgumentError(
            "per-element kernels: reference_fields tuple type mismatch at element $i",
        ))
        typeof(sti) === typeof(st1) || throw(ArgumentError(
            "per-element kernels: material state type mismatch at element $i",
        ))
    end
    _assert_homogeneous_ka_column_kernel_scalars!(kernels)
    return nothing
end

"""
    ka_per_element_kernel_column_supported(col::PerElementKernelColumn) -> Bool

`true` when the KA GPU matvec may use a single prototype kernel object (the
first element's kernel) together with per-element `qp_buffers` filled on CPU
Pass 1. Allowed when every element's `evaluate_entry` either reads material
data only from `qp_buffers`, or reads additional scalars that
[`assert_homogeneous_dof_based_kernel_column!`](@ref) already enforces across
the column (`ContinuumKernel`, `HeatKernel`, `ThermoElasticKernel`,
`BiotPoroelasticKernel`, `ThermoPoroelasticKernel`).
"""
@inline function ka_per_element_kernel_column_supported(::PerElementKernelColumn{K}) where {K}
    return (
        K <: ContinuumKernel ||
        K <: HeatKernel ||
        K <: ThermoElasticKernel ||
        K <: BiotPoroelasticKernel ||
        K <: ThermoPoroelasticKernel
    )
end

@inline ka_per_element_kernel_column_supported(::UniformKernelColumn) = true
