# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Microkernel interface for DOF-based assembly.

Enables multi-physics coupling via type dispatch, matrix-free solvers,
and zero-allocation assembly.

See `docs/src/developer/microkernel_architecture.md` for complete documentation.
"""

using ..JuliaFEM: AbstractField, AbstractKernel
using ..JuliaFEM: GeometryCache, AssemblyMaterialWorkspace

# ============================================================================
# MICROKERNEL INTERFACE
# ============================================================================

"""
    evaluate(kernel, field_i, field_j, k, l, comp_i, comp_j, material_cache, geometry_cache, q) -> Float64

Compute single scalar contribution to K[i,j] at integration point q.

# Arguments
- `kernel`: Kernel instance (holds material, geometry properties)
- `field_i`, `field_j`: Field types for dispatch (Displacement, Temperature, etc.)
- `k`, `l`: Shape function/node indices
- `comp_i`, `comp_j`: Component indices (1-3 for vectors, 1 for scalars)
- `material_cache`: Precomputed material state (σ, 𝔻, κ, etc.)
- `geometry_cache`: Precomputed geometry (∇N, detJ, etc.)
- `q`: Integration point index

# Returns
Float64 value WITHOUT detJ*w scaling (assembler handles weighting).

Default implementation returns 0.0 (no coupling). Override for specific kernel-field pairs.

See `docs/src/developer/microkernel_architecture.md` for design rationale and examples.
"""
@inline function evaluate(
    kernel::AbstractKernel,
    field_i::AbstractField,
    field_j::AbstractField,
    k::Int,
    l::Int,
    comp_i::Int,
    comp_j::Int,
    material_workspace::AssemblyMaterialWorkspace,
    geometry_cache::GeometryCache,
    q::Int
)
    return 0.0  # Default: no coupling
end

# ============================================================================
# TRAITS FOR KERNEL REQUIREMENTS
# ============================================================================

"""
    requires_basis_values(kernel) -> Bool

Does kernel need basis function VALUES (N)?
Default: false (most kernels only need gradients).
"""
requires_basis_values(::AbstractKernel) = false

"""
    requires_basis_gradients(kernel) -> Bool

Does kernel need basis function GRADIENTS (∇N)?
Default: true (most kernels need gradients).
"""
requires_basis_gradients(::AbstractKernel) = true

"""
    requires_basis_second_derivatives(kernel) -> Bool

Does kernel need basis function SECOND DERIVATIVES (∇²N)?
Default: false (only beam/plate bending needs this).
"""
requires_basis_second_derivatives(::AbstractKernel) = false
