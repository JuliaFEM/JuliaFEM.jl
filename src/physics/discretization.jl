# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
    FEDiscretization(elements, handler, mesh, cache, assembler)

Lightweight bundle of the usual DOF-based finite-element assembly inputs:
`elements` from [`create_elements!`](@ref), the [`DOFHandler`](@ref), the
[`AbstractMesh`](@ref), a pre-built [`DOFBasedCOOCache`](@ref) (whose
[`kernel_column`](@ref) holds the volume kernel), and a
[`DOFBasedCOOAssembler`](@ref) tag.

Use [`assemble!`](@ref)`(fe)` then [`linear_system`](@ref)`(fe)` to obtain
`(K, f)` without threading several separate variables through driver code.

This is intentionally minimal: it does not own the mesh or elements, impose
boundary conditions, or choose a solver. The kernel is read from
`fe.cache.kernel_column` only; there is no separate redundant kernel field.
"""
struct FEDiscretization{E,H,M,C,A}
    elements::Vector{E}
    handler::H
    mesh::M
    cache::C
    assembler::A
end

@inline function assemble!(fe::FEDiscretization; kwargs...)
    return assemble!(fe.cache, fe.assembler, fe.mesh; kwargs...)
end

@inline function assemble_internal_force!(f::AbstractVector{Float64}, fe::FEDiscretization; kwargs...)
    return assemble_internal_force!(f, fe.cache, fe.assembler, fe.mesh; kwargs...)
end

@inline function nonlinear_equilibrium_residual!(
    r::AbstractVector{Float64},
    f_ext::AbstractVector{Float64},
    f_work::AbstractVector{Float64},
    fe::FEDiscretization,
    u::AbstractVector{Float64};
    kwargs...,
)
    return nonlinear_equilibrium_residual!(
        r, f_ext, f_work, fe.cache, fe.assembler, fe.mesh, u; kwargs...,
    )
end

@inline function linear_system(fe::FEDiscretization)
    return extract_system(fe.cache)
end

@inline total_dofs(fe::FEDiscretization) = fe.handler.total_dofs
