# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
Microkernel contract for the DOF-based assembler.

The DOF-based assembler walks one DOF row at a time and asks the kernel
for a single scalar `K[i, j]`. To keep the assembler kernel-agnostic and
zero-allocation, every kernel must opt in by implementing three pieces
plus one Pass~1 material hook:

1. `qpoint_buffer_eltype(kernel)` — what type of value the kernel needs
   stored once per quadrature point per element. For continuum mechanics
   this is the elasticity tensor; for heat conduction it would be the
   conductivity tensor; for beams a stiffness scalar; etc.

2. `update_qpoint_buffer!(buffer, material_workspace, kernel)` — fill
   that buffer once per element from the per-element material workspace.
   This is called in Pass 1 of `assemble!` (element loop) so it must be
   allocation-free. Piecewise-element scalars (e.g. `ElementWiseScalarDiffusion`)
   use an extra `eid` argument only on the internal
   `_dof_based_fill_qpoint_buffer!` dispatch path in `dof_based_coo.jl`.

3. `prepare_dof_based_material_workspace!(kernel, material_workspace,
   geometry_cache, element_cache, eid, configuration, global_material_cache,
   Δt, ::Type{E})` — fill each integration point of `material_workspace`
   for this element before `update_qpoint_buffer!`. The default seeds every
   IP from [`reference_fields`](@ref)`(kernel)`; continuum mechanics overrides
   this for strain- and state-dependent materials (see
   `domains/continuum/dof_based_pass1.jl`).

4. `evaluate_entry(kernel, geometry_cache, qpoint_buffer, layout_i, layout_j, elem_id)`
   — the actual microkernel. Returns the single scalar `K[i, j]` for the
   local DOF pair `(i, j)` described by two `DOFLayoutEntry` values.
   `elem_id` is the volume element index (needed for facet-oriented kernels).
   Called inside Pass 2 of `assemble!`, in a hot loop, so it must also
   be allocation-free.

Together these methods let the DOF-based assembler dispatch on any
`AbstractKernel` without baking in domain-specific Pass~1 logic in the
driver, while keeping the inner loop fully type-stable thanks to the
compile-time `local_dof_layout(E)` table that produces the `DOFLayoutEntry`
arguments.

The KernelAbstractions stiffness / mass matvecs (`dof_based_coo_ka.jl`) call
the same `evaluate_entry` / `evaluate_mass_entry` with one prototype kernel per
launch; per-element variation must live in `qpoint_buffer` columns unless the
volume kernel column is uniform. For `PerElementKernelColumn`, see
`ka_per_element_kernel_column_supported` in `dof_based/kernel_column.jl`.
=#

"""
    qpoint_buffer_eltype(kernel::AbstractKernel) -> Type

Element type of the per-quadrature-point buffer the kernel needs. The
DOF-based assembler allocates `Vector{qpoint_buffer_eltype(kernel)}` of
length `n_ips` per element and fills it once via `update_qpoint_buffer!`
in Pass 1.

A kernel must define this method; there is no default.
"""
function qpoint_buffer_eltype end

"""
    update_qpoint_buffer!(buffer, material_workspace, kernel::AbstractKernel)

Populate the per-quadrature-point buffer for one element from the
per-element material workspace. Called once per element in Pass 1 of the
DOF-based assembler.

Must be allocation-free.
"""
function update_qpoint_buffer! end

"""
    evaluate_entry(
        kernel::AbstractKernel,
        geometry_cache,
        qpoint_buffer,
        layout_i::DOFLayoutEntry,
        layout_j::DOFLayoutEntry,
        elem_id::Int,
    ) -> Float64

Compute the scalar stiffness contribution `K[i, j]` for the local DOF
pair `(i, j)` on a prepared element (`elem_id` indexes `mesh.connectivity`).

Each `DOFLayoutEntry` describes one local DOF as
`(field_idx, entity_local, component)`, produced at compile time by
`local_dof_layout(::Type{Element{K, P, S, N}})`. This contract lets a
kernel inspect both DOFs (e.g. to dispatch on the field pair for
multi-field problems) while keeping the call site uniform.

`geometry_cache` provides `∇N`, `detJ·w`, and node coordinates.
`qpoint_buffer` is the kernel-specific buffer filled by
`update_qpoint_buffer!`.

Must be allocation-free in the inner loop.
"""
function evaluate_entry end

"""
    evaluate_mass_entry(
        kernel::AbstractKernel,
        geometry_cache,
        qpoint_buffer,
        layout_i::DOFLayoutEntry,
        layout_j::DOFLayoutEntry,
    ) -> Float64

Compute the scalar mass-matrix contribution `M[i, j]` for the local
DOF pair `(i, j)`. Same call shape and constraints as `evaluate_entry`,
but for the `(N_i, ρ N_j)` bilinear form instead of `(B_i : C : B_j)`.

The default implementation returns `0.0`, so a kernel that has not opted
into mass-matrix support transparently produces a structural-zero `M`
through `apply_M!` / `assemble_M!`. Continuum and heat kernels override
this; thermo-elastic / new-physics kernels can override or inherit
zero-mass.

Reads basis values from `geometry_cache.N_data` (the SoA batch added in
the geometry-cache refactor) and the per-IP weights from
`geometry_cache.detJ_w`. Material density / heat capacity is carried on
the kernel itself rather than per-IP, so this microkernel does not
look at `qpoint_buffer` for the linear case (the argument is still
present so a future variable-density material drops in without changing
the assembler).

Must be allocation-free in the inner loop.
"""
@inline evaluate_mass_entry(
    ::AbstractKernel,
    geometry_cache,
    qpoint_buffer,
    layout_i,
    layout_j,
) = 0.0

"""
    reference_fields(kernel::AbstractKernel) -> (fields_ref::NamedTuple, empty_state::NamedTuple)

Per-quadrature-point material *reference* values used to seed the
per-element material workspace at the start of every assembly pass.

For *stateless, constant-tangent* materials (linear elasticity, linear
heat conduction) every IP can simply receive `fields_ref` directly,
eliminating any per-IP constitutive call in Pass 1 and keeping the
assembler allocation-free. For materials with state (plasticity etc.)
this same hook can return a sensible "current-step zero" reference and
the per-IP update is done elsewhere.

The returned `NamedTuple`'s field names must match what the kernel's
`update_qpoint_buffer!` reads from the material workspace, e.g.
`(σ, 𝔻)` for `ContinuumKernel`, `(q, k)` for `HeatKernel`.

A kernel must define this method; there is no default — the previous
`(σ=…, 𝔻=…)` hardcode lived inside `DOFBasedCOOCache` and made the
assembler accidentally continuum-only.
"""
function reference_fields end

"""
    prepare_dof_based_material_workspace!(
        kernel,
        material_workspace,
        geometry_cache,
        element_cache,
        eid::Int,
        configuration::Union{Nothing,AbstractVector{Float64}},
        global_material_cache::Union{Nothing,GlobalMaterialCache},
        Δt::Float64,
        ::Type{E},
    ) -> Nothing

DOF-based Pass~1 (element loop): populate every integration point of
`material_workspace` for this element before [`update_qpoint_buffer!`](@ref).

The default implementation copies [`reference_fields`](@ref)`(kernel)` to
all IPs. Kernels that need configuration-dependent or stateful constitutive
updates (e.g. [`ContinuumKernel`](@ref)) should override this method.

Must be allocation-free on the hot path.
"""
function prepare_dof_based_material_workspace! end

@inline function prepare_dof_based_material_workspace!(
    kernel::AbstractKernel,
    material_workspace::AssemblyMaterialWorkspace,
    ::GeometryCache,
    element_cache::ElementCache,
    ::Int,
    ::Union{Nothing,AbstractVector{Float64}},
    ::Union{Nothing,GlobalMaterialCache},
    ::Float64,
    ::Type{<:AbstractElement},
)
    fields_ref_e, empty_state_e = reference_fields(kernel)
    fields_mw = getfield(material_workspace, 1)
    states_mw = getfield(material_workspace, 2)
    ips_ec = getfield(element_cache, :ips)
    nips = length(ips_ec)
    @inbounds for q in 1:nips
        fields_mw[q] = fields_ref_e
        states_mw[q] = empty_state_e
    end
    return nothing
end
