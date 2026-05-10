# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Linear mixed displacement–pressure (u–p) kernel for nearly incompressible
small-strain elasticity.

DOF layout (element template `S`):

  `@DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}`

or any equivalent where field 1 is vertex displacement and field 2 is a
scalar cell DOF (piecewise-constant pressure on each element).

Weak form (steady, no body force in the stiffness blocks):

  ∫ σ(u) : ε(v) dΩ  +  ∫ p · div(v) dΩ  = rhs(v)
  ∫ q · div(u) dΩ  −  κ⁻¹ ∫ p · q dΩ  = 0

with `σ` and the tangent `𝔻` from `compute_stress` at the small-strain
tensor `ε(u)` (same material pipeline as `ContinuumKernel` for the u–u
block). The compressibility term uses `inv_bulk = 1/κ` (κ bulk modulus);
set `inv_bulk = 0` for the incompressible limit (then `K_pp` is zero and
the pressure block is singular up to discrete hydrostatic modes).

Incompressible / Stokes-style workflow:

1. **Pressure gauge** — pin at least one global pressure DOF (or add
   PSPG / grad–div). Use [`default_pressure_gauge_dof`](@ref) for a
   conventional single-DOF pin on element 1.

2. **BCs** — merge displacement Dirichlet indices with the pressure pin
   in one `PenaltyDirichlet` / `EliminatedDirichlet`.

3. **Solvers** — the operator is symmetric indefinite; use direct `\`,
   MINRES, or GMRES, not CG. `MatrixFreeOperator` sets `isposdef == false`
   for `MixedUPKernel` so Krylov stacks do not assume SPD.

4. **Preconditioners** — `JacobiPreconditioner` is only a baseline on
   saddle problems; approximate Schur / block preconditioners are the
   natural follow-up.

This is the first mixed kernel: `evaluate_entry` dispatches on
`(field_idx(layout_i), field_idx(layout_j))` like `ThermoElasticKernel`.
=#

using Tensors

using ..JuliaFEM: AbstractKernel, AbstractFormulation
using ..JuliaFEM: ContinuumFormulation, ThreeDimensional, AbstractContinuumTheory
using ..JuliaFEM: AbstractMaterial, Displacement
using ..JuliaFEM: AssemblyMaterialWorkspace, compute_stress
import ..JuliaFEM: qpoint_buffer_eltype, update_qpoint_buffer!, evaluate_entry,
                   evaluate_mass_entry,
                   reference_fields, get_field, dofs_per_node
using ..JuliaFEM: DOFLayoutEntry, field_idx, entity_local, component, extract_tangent!


"""
    MixedUPKernel{Theory, Mat}

Mixed `u`–`p` kernel: 3D vertex displacement (field 1) + scalar cell pressure
(field 2). See the file-level docstring for the weak form.

# Fields
- `formulation::ContinuumFormulation{Theory}` — geometric driver (`ThreeDimensional`, …)
- `material::Mat` — mechanical material (`LinearElastic`, …)
- `inv_bulk::Float64` — `1/κ` for the `−κ⁻¹ ∫ p q dΩ` term (`0` = incompressible limit)

# Example

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
kernel = MixedUPKernel(
    ContinuumFormulation{ThreeDimensional}(),
    LinearElastic(E = 210e9, ν = 0.3),
    inv_bulk = 1.0 / (210e9 / 3),  # order-of-magnitude compressible term
)
```
"""
struct MixedUPKernel{Theory<:AbstractContinuumTheory, Mat<:AbstractMaterial} <: AbstractKernel
    formulation::ContinuumFormulation{Theory}
    material::Mat
    inv_bulk::Float64
end

function MixedUPKernel(
    formulation::ContinuumFormulation{Theory},
    material::Mat;
    inv_bulk::Float64 = 0.0,
) where {Theory<:AbstractContinuumTheory, Mat<:AbstractMaterial}
    inv_bulk ≥ 0.0 || throw(ArgumentError("inv_bulk must be ≥ 0, got inv_bulk = $inv_bulk"))
    return MixedUPKernel{Theory, Mat}(formulation, material, inv_bulk)
end

# Saddle-point u–p system; matrix-free K is symmetric indefinite.
@inline operator_is_posdef(::MixedUPKernel) = false

@inline dofs_per_node(::MixedUPKernel) = 4

function get_field(::K) where {K<:MixedUPKernel}
    error("$(K) is mixed u–p — use `local_dof_layout(E)` and `elem.dof_indices`; " *
          "do not call `get_field(kernel)`.")
end

@inline qpoint_buffer_eltype(::MixedUPKernel) = SymmetricTensor{4,3,Float64,36}

@inline function reference_fields(kernel::MixedUPKernel)
    ε_ref = zero(SymmetricTensor{2,3,Float64,6})
    σ_ref, 𝔻_ref, _ = compute_stress(kernel.material, ε_ref, NamedTuple(), 0.0)
    return ((σ = σ_ref, 𝔻 = 𝔻_ref), NamedTuple())
end

@inline function update_qpoint_buffer!(
    buffer::AbstractVector{SymmetricTensor{4,3,Float64,36}},
    workspace::AssemblyMaterialWorkspace{FieldType, StateType},
    ::MixedUPKernel,
) where {FieldType, StateType}
    fields = getfield(workspace, 1)
    extract_tangent!(buffer, fields, FieldType)
    return nothing
end

"""
    evaluate_entry(kernel::MixedUPKernel, geometry_cache, 𝔻_vec, layout_i, layout_j, elem_id::Int)

Volume kernel; `elem_id` is unused.

| (field_i, field_j) | block | contribution |
| ------------------ | ----- | ------------ |
| (1, 1)             | K_uu  | `Σ_q B_iα : 𝔻 : B_jβ · detJ_w` (same as `ContinuumKernel`) |
| (1, 2)             | K_up  | `+ Σ_q (∂N_i/∂x_α) · detJ_w` (trial `p` constant on cell) |
| (2, 1)             | K_pu  | `+ Σ_q (∂N_j/∂x_β) · detJ_w` (= `K_up^T` in the global matrix) |
| (2, 2)             | K_pp  | `- inv_bulk · Σ_q detJ_w` (piecewise-constant `p`, `q`) |
"""
@inline function evaluate_entry(
    kernel::MixedUPKernel,
    geometry_cache,
    𝔻_vec::AbstractVector{<:SymmetricTensor{4,3}},
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
    ::Int,
)
    fi = field_idx(layout_i)
    fj = field_idx(layout_j)

    node_i = entity_local(layout_i)
    node_j = entity_local(layout_j)
    comp_i = component(layout_i)
    comp_j = component(layout_j)

    n_ips = length(geometry_cache.detJ_w)
    K_ij = 0.0

    if fi == 1 && fj == 1
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            C = Tensor{4,3}(𝔻_vec[q])
            K_ij += compute_stiffness_value(∇N_i, ∇N_j, C, comp_i, comp_j) * detJw
        end

    elseif fi == 1 && fj == 2
        @inbounds for q in 1:n_ips
            ∇N_i  = geometry_cache.∇N_data[q, node_i]
            detJw = geometry_cache.detJ_w[q]
            K_ij += ∇N_i[comp_i] * detJw
        end

    elseif fi == 2 && fj == 1
        @inbounds for q in 1:n_ips
            ∇N_j  = geometry_cache.∇N_data[q, node_j]
            detJw = geometry_cache.detJ_w[q]
            K_ij += ∇N_j[comp_j] * detJw
        end

    else  # fi == 2 && fj == 2
        ib = kernel.inv_bulk
        if ib == 0.0
            return 0.0
        end
        vol = 0.0
        @inbounds for q in 1:n_ips
            vol += geometry_cache.detJ_w[q]
        end
        K_ij = -ib * vol
    end

    return K_ij
end

@inline evaluate_mass_entry(
    ::MixedUPKernel,
    geometry_cache,
    qp_buffer,
    layout_i::DOFLayoutEntry,
    layout_j::DOFLayoutEntry,
) = 0.0

# ----------------------------------------------------------------------------
# Pressure gauge helper (incompressible / Stokes-style usage)
# ----------------------------------------------------------------------------

"""
    default_pressure_gauge_dof(handler; field_pressure = 2, elem_id = 1) -> Int

Return one global pressure DOF index, intended for pinning with
`PenaltyDirichlet` / `EliminatedDirichlet` when `inv_bulk == 0`.

`handler` is expected to be a `DOFHandler`. The annotation is left
untyped so this helper can sit in the continuum domain layer without
forcing a load-order dependency on `DOFHandler` itself; the only field
read is `handler.field_starts[field_pressure][elem_id]`.

Assumes the second field in the handler is the scalar cell pressure
(true for `DOF{…, Cell}` with one scalar per cell).

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell}}
elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
p_dof = default_pressure_gauge_dof(handler)
bc = PenaltyDirichlet([fixed_u_dofs; p_dof], zeros(length(fixed_u_dofs) + 1))
```
"""
@inline function default_pressure_gauge_dof(
    handler;
    field_pressure::Int = 2,
    elem_id::Int = 1,
)
    return handler.field_starts[field_pressure][elem_id]
end
