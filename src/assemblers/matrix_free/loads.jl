# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Neumann (load) boundary conditions for the DOF-based assembler.

Counterpart of `dirichlet.jl` for the right-hand-side: declarative
load types that accumulate into `f` (or any user-provided vector) via a
single `apply_load!(f, load, cache, asm, kernel, mesh)` entry point.
The `kernel` argument selects the integration path via method dispatch; it
should describe the same volume physics as `cache.kernel_column` for the model.
Each concrete subtype implements the integration appropriate to it,
reusing the SoA `N_data` / `detJ_w` batches the DOF-based assembler
already builds in `_prepare_caches!`.

Load types provided include:

  * `NodalForce(dofs, values)` — point loads on known global DOF
    indices. Allocation-free and trivially zero-cost; just an indexed
    `f[d] += value` loop.

  * `UniformBodyForce(b)` — constant body force per unit volume over
    the entire mesh. Computes `f[i, α] += ∫_Ω N_i · b_α dV` per element
    using the same batched geometry the matrix-free `apply_M!` /
    `apply_K!` rely on. Vector-valued for elasticity (`Vec{3,Float64}`),
    scalar-valued for heat (`Float64` heat source per unit volume).

  * `SurfaceLoad(faces, traction)` — distributed traction (or heat
    flux) over a list of mesh faces (quads, triangles, or **segments**
    with `NN = 2`). Computes `f[i, α] += ∫_Γ N_i · t_α dS` (or
    `∫ N_i · t ds` on segments) per face using face Gauss rules.
    Vector-valued for elasticity, scalar-valued for heat. The natural
    complement of `UniformBodyForce`: body force lives in the volume
    integral, surface traction lives in the surface integral. Entries
    scatter into **field 1** DOFs via `dof_handler.field_starts[1]`
    (same as `cache.field_starts1`).

  * [`SurfaceScalarFluxOnField`](@ref) — same scalar face quadrature as
    `SurfaceLoad` with `Float64` (or `Vector{Float64}`) flux, but targets an
    arbitrary **vertex field index** (`dof_handler.field_starts[field_index]`).
    Use for thermal or fluid Neumann data on multi-field handlers (e.g.
    [`ThermoPoroelasticKernel`](@ref) with `u` / `T` / `p`).

  * [`UniformMixedDarcySource`](@ref) — adds ``f \\cdot |K_e|`` to each cell
    pressure DOF for mixed RT₀–P₀ Darcy (second equation); works with
    [`AbstractDarcyMixedRT0P0Kernel`](@ref) (Tet4 or Hex8).

  * [`MixedDarcyTet4BoundaryNormalFluxLoad`](@ref) — Tet4: uniform normal flux density
    ``g`` on panels `(elem_id, local_face)` (triangle quadrature).
  * [`MixedDarcyHex8BoundaryNormalFluxLoad`](@ref) — Hex8: same weak form on quad faces
    (four Gauss points per face).

`apply_load!` is *additive* on `f`, so multiple loads compose
naturally:

```julia
apply_load!(f, body,    cache, asm, kernel, mesh)
apply_load!(f, surface, cache, asm, kernel, mesh)
apply_load!(f, point,   cache, asm, kernel, mesh)
```

Combined with `EliminatedDirichlet`'s `apply_constraint!(K, b, c)`
"lift" of the RHS and the matrix-free `op` returned by
`matrix_free_op`, this gives a complete declarative BC story for both
direct and Krylov solves.
"""

abstract type AbstractNeumannLoad end

# ----------------------------------------------------------------------------
# NodalForce
# ----------------------------------------------------------------------------

"""
    NodalForce(dofs, values)

Concentrated point loads on a list of global DOF indices. Stores the
DOF index list and the corresponding force values; `apply_load!` does
`f[d] += value` for each pair.

Useful for prescribed reactions, manually assembled surface tractions,
or for verifying correctness of the body-force / matrix-free path
against a reference assembled solve.
"""
struct NodalForce{IT<:AbstractVector{<:Integer},
                  VT<:AbstractVector{Float64}} <: AbstractNeumannLoad
    dofs::IT
    values::VT

    function NodalForce(dofs::IT, values::VT) where {
            IT<:AbstractVector{<:Integer}, VT<:AbstractVector{Float64}}
        @assert length(dofs) == length(values) (
            "NodalForce: dofs ($(length(dofs))) and values ($(length(values))) " *
            "must have the same length")
        return new{IT, VT}(dofs, values)
    end
end

NodalForce(dofs::AbstractVector{<:Integer},
           values::AbstractVector{<:Real}) =
    NodalForce(dofs, Float64.(values))

"""
    apply_load!(f::AbstractVector{Float64}, load::NodalForce,
                cache::DOFBasedCOOCache, asm, kernel, mesh) -> f

Accumulate point loads into `f`: `f[load.dofs[k]] += load.values[k]`.
Allocation-free; ignores `cache`/`asm`/`kernel`/`mesh` (kept in the
signature for API symmetry with body / surface loads).
"""
@inline function apply_load!(f::AbstractVector{Float64},
                             load::NodalForce,
                             cache::DOFBasedCOOCache,
                             asm::DOFBasedCOOAssembler,
                             kernel::AbstractKernel,
                             mesh::AbstractMesh)
    _depwarn_redundant_kernel_arg!(:apply_load!)
    @inbounds for k in eachindex(load.dofs)
        f[load.dofs[k]] += load.values[k]
    end
    return f
end

# ----------------------------------------------------------------------------
# UniformBodyForce
# ----------------------------------------------------------------------------

"""
    UniformBodyForce(b)

Constant body force per unit volume over the whole mesh.

* For elasticity (`ContinuumKernel`, 3 DOFs/node), `b` is a
  `Vec{3,Float64}` — body force per unit volume, e.g. gravity
  `Vec{3}((0.0, 0.0, -ρ * 9.81))`.
* For heat (`HeatKernel`, 1 DOF/node), `b` is a `Float64` — heat
  source per unit volume `[W/m³]`.

`apply_load!` integrates `f[i, α] += ∫ N_i b_α dV` element by element,
component by component, using the SoA `N_data` and `detJ_w` batches
from the cache. Allocation-free after `_prepare_caches!` is warm.

Variable / spatially-dependent body forces drop in by introducing a new
type and overriding `_body_component(load, X, comp)` — the integration
loop in `apply_load!` is identical.
"""
struct UniformBodyForce{V} <: AbstractNeumannLoad
    b::V
end

# Component accessors — make the same integration loop work for both
# vector (elasticity) and scalar (heat) loads. `comp` is the component
# index from `DOFLayoutEntry` (always 1 for a scalar field).
@inline _body_component(b::Vec{N,Float64}, comp::Int) where {N} = b[comp]
@inline _body_component(b::Real, comp::Int) = Float64(b)

"""
    apply_load!(f::AbstractVector{Float64}, load::UniformBodyForce,
                cache::DOFBasedCOOCache, asm, kernel, mesh) -> f

Element-by-element assembly of `f[i, α] += ∫ N_i b_α dV` for every
local DOF. Reuses the cache's SoA `N_data` and `detJ_w` batches and the
compile-time `local_dof_layout` table, so the inner loop is fully
type-stable and allocation-free after warmup.
"""
function apply_load!(f::AbstractVector{Float64},
                     load::UniformBodyForce,
                     cache::DOFBasedCOOCache{T,B,IPS,E,GC,Buf,FieldType,StateType,KS},
                     asm::DOFBasedCOOAssembler,
                     kernel::AbstractKernel,
                     mesh::AbstractMesh) where {T,B,IPS,E<:AbstractElement,
                                                GC,Buf,FieldType,StateType,KS}
    _depwarn_redundant_kernel_arg!(:apply_load!)
    @assert length(f) == cache.ndofs (
        "apply_load!: f has length $(length(f)); expected $(cache.ndofs)")

    # We reuse the same Pass 1 `_prepare_caches!` as `apply_K!` /
    # `assemble_M!` so the geometry / N_data / detJ_w are populated
    # exactly the same way. Cheap to call repeatedly because Pass 1
    # is a fixed-cost sweep over the elements.
    _prepare_caches!(cache, mesh)

    elements         = cache.elements
    element_caches   = cache.element_caches
    geometry_caches  = cache.geometry_caches

    layout     = local_dof_layout(E)
    ndofs_elem = length(layout)

    @inbounds for elem_idx in 1:length(elements)
        ec = element_caches[elem_idx]
        gc = geometry_caches[elem_idx]
        n_ips     = length(gc.detJ_w)
        dofs_elem = ec.dofs

        @inbounds for li in 1:ndofs_elem
            entry  = layout[li]
            node_i = entity_local(entry)
            comp_i = component(entry)
            bα     = _body_component(load.b, comp_i)
            if bα == 0.0
                continue
            end

            sum_q = 0.0
            @inbounds for q in 1:n_ips
                N_i   = gc.N_data[q, node_i]
                detJw = gc.detJ_w[q]
                sum_q += N_i * detJw
            end

            f[Int(dofs_elem[li])] += bα * sum_q
        end
    end

    return f
end

# ----------------------------------------------------------------------------
# UniformMixedDarcySource — ∫ q f dΩ on P₀ pressure test functions
# ----------------------------------------------------------------------------

@inline function _mixed_darcy_flux_li(layout, iface::Int)
    @inbounds for li in 1:length(layout)
        e = layout[li]
        if Int(field_idx(e)) == 1 && Int(entity_local(e)) == iface && Int(component(e)) == 1
            return li
        end
    end
    return 0
end

"""
    UniformMixedDarcySource(f)

Uniform volumetric source for the **pressure test equation** of mixed RT₀–P₀ Darcy:

``\\int_\\Omega q \\,(\\nabla\\!\\cdot u)\\,\\mathrm{d}\\Omega = \\int_\\Omega q \\,f\\,\\mathrm{d}\\Omega``

with piecewise constant `q` (one DOF per cell). Each cell pressure unknown receives
``f \\cdot |K_e|`` where `|K_e|` is the element volume from ``\\sum_q \\det J \\, w``.

Implemented for [`AbstractDarcyMixedRT0P0Kernel`](@ref) (Tet4 and Hex8 mixed kernels).
The element template must place the scalar cell pressure *after* face flux fields in
[`local_dof_layout`](@ref) (the constructor discovers the pressure local DOF by `field_idx == 2`).

Calls `_prepare_caches!` once (same as [`UniformBodyForce`](@ref)).
"""
struct UniformMixedDarcySource <: AbstractNeumannLoad
    f::Float64
end

function apply_load!(
    fvec::AbstractVector{Float64},
    load::UniformMixedDarcySource,
    cache::DOFBasedCOOCache{T, B, IPS, E, GC, Buf, FT, ST, KS},
    asm::DOFBasedCOOAssembler,
    kernel::AbstractDarcyMixedRT0P0Kernel,
    mesh::AbstractMesh,
) where {T, B, IPS, E <: AbstractElement, GC, Buf, FT, ST, KS}
    @assert length(fvec) == cache.ndofs (
        "apply_load!: f has length $(length(fvec)); expected $(cache.ndofs)")

    _prepare_caches!(cache, mesh)

    layout = local_dof_layout(E)
    p_li = 0
    @inbounds for li in 1:length(layout)
        if Int(field_idx(layout[li])) == 2
            p_li = li
            break
        end
    end
    p_li > 0 || error(
        "UniformMixedDarcySource: no local DOF with field_idx == 2 in Element{$E}; " *
            "use @DOFSet{σ::DOF{RT0FaceFlux, Face}, p::DOF{Float64, Cell}} with σ first.",
    )

    element_caches = cache.element_caches
    geometry_caches = cache.geometry_caches
    nelem = length(element_caches)

    src = load.f
    @inbounds for eid in 1:nelem
        gc = geometry_caches[eid]
        ec = element_caches[eid]
        vol = 0.0
        n_ips = length(gc.detJ_w)
        for q in 1:n_ips
            vol += gc.detJ_w[q]
        end
        gdof = Int(ec.dofs[p_li])
        fvec[gdof] += src * vol
    end

    return fvec
end

# ----------------------------------------------------------------------------
# MixedDarcyTet4BoundaryNormalFluxLoad — ∫ g φ·n dS on RT₀ flux test functions (Tet4)
# ----------------------------------------------------------------------------

# Symmetric order-2 rule on the reference triangle ξ ≥ 0, η ≥ 0, ξ + η ≤ 1 (area 1/2).
const _MIXED_DARCY_TRI3_AB =
    ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
const _MIXED_DARCY_TRI3_W = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)

"""
    MixedDarcyTet4BoundaryNormalFluxLoad(panels, g)

Boundary contribution ``\\int_\\Gamma g\\, \\mathbf{\\phi}_i \\cdot \\mathbf{n}\\,\\mathrm{d}S`` to the
**flux test equation** of mixed RT₀–P₀ Darcy on Tet4, with uniform normal flux density ``g``
(`[m/s]`, outward positive relative to the element). Each entry of `panels` is
`(elem_id, local_face)` with `local_face ∈ 1:4` ([`faces(::Tet4)`](@ref)).

Uses three quadrature points per triangle (exact if ``g \\,\\mathbf{\\phi}\\!\\cdot\\!\\mathbf{n}`` is linear on the face).
Requires [`DarcyMixedRT0P0Kernel`](@ref) and [`Mesh{4, Tet4}`](@ref).
"""
struct MixedDarcyTet4BoundaryNormalFluxLoad <: AbstractNeumannLoad
    panels::Vector{Tuple{Int, Int}}
    g::Float64
end

function apply_load!(
    fvec::AbstractVector{Float64},
    load::MixedDarcyTet4BoundaryNormalFluxLoad,
    cache::DOFBasedCOOCache{T, B, IPS, E, GC, Buf, FT, ST, KS},
    asm::DOFBasedCOOAssembler,
    kernel::DarcyMixedRT0P0Kernel,
    mesh::Mesh{4, Tet4},
) where {T, B, IPS, E <: AbstractElement, GC, Buf, FT, ST, KS}
    @assert length(fvec) == cache.ndofs (
        "apply_load!: f has length $(length(fvec)); expected $(cache.ndofs)")

    _prepare_caches!(cache, mesh)

    layout = local_dof_layout(E)
    element_caches = cache.element_caches
    geometry_caches = cache.geometry_caches
    nelem = length(element_caches)

    @inbounds for pid in eachindex(load.panels)
        eid, lf = load.panels[pid]
        (1 ≤ eid ≤ nelem) || throw(ArgumentError("MixedDarcyTet4BoundaryNormalFluxLoad: elem_id $eid out of range 1:$nelem"))
        (1 ≤ lf ≤ 4) || throw(ArgumentError("MixedDarcyTet4BoundaryNormalFluxLoad: local_face $lf out of range 1:4"))

        gc = geometry_caches[eid]
        ec = element_caches[eid]
        X = gc.X

        fc = faces(Tet4())[lf]
        @inbounds p1 = X[Int(fc.vertices[1])]
        @inbounds p2 = X[Int(fc.vertices[2])]
        @inbounds p3 = X[Int(fc.vertices[3])]

        Vphys = 0.0
        n_ips = length(gc.detJ_w)
        for q in 1:n_ips
            Vphys += gc.detJ_w[q]
        end

        cross_vec = (p2 - p1) × (p3 - p1)
        jac_face = norm(cross_vec)
        jac_face > 0.0 || continue

        orient = Float64(tet_face_outward_sign(X, lf))
        n_unit = orient * cross_vec / jac_face

        for k in 1:3
            ξq, ηq = _MIXED_DARCY_TRI3_AB[k]
            wq = _MIXED_DARCY_TRI3_W[k]
            λ1 = 1.0 - ξq - ηq
            xq = λ1 * p1 + ξq * p2 + ηq * p3
            base = load.g * wq * jac_face
            for ifi in 1:4
                φ = _rt0_phi_tet4(X, Vphys, ifi, xq)
                li = _mixed_darcy_flux_li(layout, ifi)
                li > 0 || error("MixedDarcyTet4BoundaryNormalFluxLoad: no flux DOF for local face $ifi")
                gdof = Int(ec.dofs[li])
                fvec[gdof] += base * dot(φ, n_unit)
            end
        end
    end

    return fvec
end

# ----------------------------------------------------------------------------
# SurfaceLoad — distributed traction / heat flux integrated on faces
# ----------------------------------------------------------------------------

"""
    SurfaceLoad(faces, traction)

Distributed surface load (traction `t`, heat flux `q`, …) integrated as
`∫_Γ N_i · t dS` over a list of mesh faces. The natural complement of
`UniformBodyForce`: body force lives in the volume integral, surface
traction lives in the surface integral.

# Arguments

- `faces::Vector{NTuple{NN,Int}}` — each face is the tuple of *global*
  mesh-node IDs of its corners. `NN = 4` for a quadrilateral face
  (e.g. one face of a Hex8 element); `NN = 3` for a triangular face
  (one face of a Tet4); `NN = 2` for a straight **segment** (two distinct
  corners, order along the edge). For `NN == 2`, vector `traction` is a
  **line load density** `[N/m]` (scalar flux `[W/m]` or `[m²/s]` analogue);
  the weak form is `∫_Γ N_i · τ ds` with `ds` the physical arc length.

- `traction` — one of
    * `Vec{3,Float64}` — uniform vector traction on every face (3D
      elasticity), in units of force per unit area.
    * `Float64`        — uniform scalar flux on every face (heat
      surface flux `[W/m²]`, or normal **Darcy flux** `[m/s]` for primal
      potential — same `∫_Γ N_i q · dS` assembly via [`SurfaceLoad`](@ref)).
    * `Vector{Vec{3,Float64}}` — per-face vector traction.
    * `Vector{Float64}` — per-face scalar flux.

# Quadrature

* Quadrilateral face (`NN == 4`) → 2 × 2 Gauss (4 points), exact for
  bilinear basis on a flat quad with constant traction.
* Triangular face (`NN == 3`) → 1-point centroid (exact for linear
  basis with constant traction; bumped to 3-point if needed by a
  later non-flat / higher-order extension).
* Segment (`NN == 2`) → 2-point Gauss on `[-1, 1]` (exact for linear
  `N` with constant line load).

# Surface Jacobian

For a face `Γ` with corner coordinates `xᵢ`, the surface measure is

    dS = ‖∂x/∂ξ × ∂x/∂η‖ dξ dη

(quad) or `‖e₁ × e₂‖ / 2` (tri, where `eₖ` are the in-plane edge
vectors). Both forms are reduced to a single `_face_metric` call below
so the per-face loop stays type-stable.

# Notes

* `apply_load!` for `SurfaceLoad` does not call `_prepare_caches!`
  — it works directly off `mesh.nodes` for the face geometry and the
  `DOFHandler.field_starts` table for DOF lookup. There is no need
  to materialise face elements, build a face cache, or call any
  volume-element machinery. This keeps the surface path compositional
  (any subset of faces from any mesh works) and disjoint from the
  ContinuumKernel evaluation path.
* The implementation is allocation-free after warmup. The face-corner
  coordinate buffer is stack-allocated (`SVector` for `NN ≤ 8`).
"""
struct SurfaceLoad{NN, V} <: AbstractNeumannLoad
    faces::Vector{NTuple{NN,Int}}
    traction::V

    function SurfaceLoad(faces::Vector{NTuple{NN,Int}}, traction::V) where {NN, V}
        # `Vec{3,Float64}` is itself an `AbstractVector` (it subtypes
        # `StaticArray`), so distinguish *per-face* tractions
        # (heap-stored `Vector{...}`) from a uniform tensor traction
        # by the concrete `Vector` type, not by `AbstractVector`.
        if traction isa Vector{<:Vec{3,Float64}} || traction isa Vector{Float64}
            @assert length(traction) == length(faces) (
                "SurfaceLoad: per-face traction has $(length(traction)) entries " *
                "but $(length(faces)) faces were given")
        end
        if !(NN == 2 || NN == 3 || NN == 4)
            error(
                "SurfaceLoad: only NN=2 (segment), NN=3 (triangular), and NN=4 " *
                "(quadrilateral) faces are supported (got NN=$NN).",
            )
        end
        return new{NN, V}(faces, traction)
    end
end

"""
    SurfaceScalarFluxOnField(faces, traction, field_index)

Scalar surface flux / traction with the same face quadrature as
[`SurfaceLoad`](@ref) for `Float64` or `Vector{Float64}` `traction`, but the
contribution is assembled into `dof_handler.field_starts[field_index]`
(vertex field), not field `1`.

`field_index` must satisfy `1 ≤ field_index ≤ length(dof_handler.field_starts)`.
For a typical [`ThermoPoroelasticKernel`](@ref) `DOFSet` ordered `u`, `T`, `p`,
use `2` for temperature Neumann flux and `3` for prescribed normal Darcy flux
on pore pressure (same units as a scalar `SurfaceLoad` on a single-field mesh).

Allocation-free after warmup (same `_integrate_face!` path as `SurfaceLoad`).
"""
struct SurfaceScalarFluxOnField{NN,V} <: AbstractNeumannLoad
    faces::Vector{NTuple{NN,Int}}
    traction::V
    field_index::Int

    function SurfaceScalarFluxOnField(
        faces::Vector{NTuple{NN,Int}},
        traction::V,
        field_index::Int,
    ) where {NN,V}
        field_index ≥ 1 || throw(
            ArgumentError("SurfaceScalarFluxOnField: field_index must be ≥ 1 (got $field_index)"),
        )
        if traction isa Vector{Float64}
            @assert length(traction) == length(faces) (
                "SurfaceScalarFluxOnField: per-face traction has $(length(traction)) entries " *
                "but $(length(faces)) faces were given",
            )
        elseif traction isa Vector{<:Vec{3,Float64}}
            throw(
                ArgumentError(
                    "SurfaceScalarFluxOnField: use SurfaceLoad for vector traction on field 1",
                ),
            )
        end
        if !(NN == 2 || NN == 3 || NN == 4)
            error(
                "SurfaceScalarFluxOnField: only NN=2 (segment), NN=3 (triangular), " *
                "and NN=4 (quadrilateral) faces are supported (got NN=$NN).",
            )
        end
        return new{NN,V}(faces, traction, field_index)
    end
end

# Per-face traction accessor — dispatches on the traction-storage type
# itself: a single value is uniform across all faces, a `Vector{...}`
# is one entry per face. Both vector- (elasticity) and scalar- (heat)
# valued tractions are supported by the same two methods.
@inline _face_traction(t::Vec{3,Float64}, f::Int) = t
@inline _face_traction(t::Float64,        f::Int) = t
@inline _face_traction(t::Vector{<:Vec{3,Float64}}, f::Int) = @inbounds t[f]
@inline _face_traction(t::Vector{Float64},          f::Int) = @inbounds t[f]

@inline _t_component(t::Vec{3,Float64}, comp::Int) = t[comp]
@inline _t_component(t::Float64,        comp::Int) = t

# Component count of a single-face traction value (not the whole load).
@inline _t_comp_count(::Vec{3,Float64}) = 3
@inline _t_comp_count(::Float64)        = 1

# Face geometry: use [`get_basis_functions`](@ref) / [`get_basis_derivatives`](@ref)
# on `Quad4`, `Tri3`, `Seg2` with [`Lagrange{1}`](@ref) (see `basis/basis_generated.jl`).

# ----------------------------------------------------------------------------
# MixedDarcyHex8BoundaryNormalFluxLoad — ∫ g φ·n dS on RT₀ flux test functions (Hex8)
# ----------------------------------------------------------------------------

@inline function _hex8_face_volume_ref_coords(ξf::Float64, ηf::Float64, lf::Int)
    fc = faces(Hex8())[lf]
    vs = fc.vertices
    ref_c = reference_coordinates(Hex8())
    N = get_basis_functions(Quad4(), Lagrange{1}(), Vec{2}((ξf, ηf)))
    @inbounds return N[1] * ref_c[vs[1]] +
        N[2] * ref_c[vs[2]] +
        N[3] * ref_c[vs[3]] +
        N[4] * ref_c[vs[4]]
end

"""
    MixedDarcyHex8BoundaryNormalFluxLoad(panels, g)

Boundary contribution ``\\int_\\Gamma g\\, \\mathbf{\\phi}_i \\cdot \\mathbf{n}\\,\\mathrm{d}S`` to the
**flux test equation** of mixed RT₀–P₀ Darcy on Hex8, with uniform normal flux density ``g``
(`[m/s]`, outward positive relative to the element). Each entry of `panels` is
`(elem_id, local_face)` with `local_face ∈ 1:6` ([`faces(::Hex8)`](@ref)).

Contravariant Piola push-forward of [`rt0_hex8_reference_basis`](@ref) matches
[`DarcyMixedHex8RT0P0Kernel`](@ref). Uses four Gauss points per quadrilateral face.

Requires [`DarcyMixedHex8RT0P0Kernel`](@ref) and [`Mesh{8, Hex8}`](@ref).

Note: entries scale with ``\\int_{\\partial\\Omega} \\mathbf{\\phi}_i\\!\\cdot\\!\\mathbf{n}\\,\\mathrm{d}S`` (Piola field),
which for a single trilinear brick need not match the lumped ``\\pm 1`` divergence coupling used in
[`DarcyMixedHex8RT0P0Kernel`](@ref) (e.g. ``4`` on one ``[0,1]^3`` element from ``[-1,1]^3`` reference).
"""
struct MixedDarcyHex8BoundaryNormalFluxLoad <: AbstractNeumannLoad
    panels::Vector{Tuple{Int, Int}}
    g::Float64
end

function apply_load!(
    fvec::AbstractVector{Float64},
    load::MixedDarcyHex8BoundaryNormalFluxLoad,
    cache::DOFBasedCOOCache{T, B, IPS, E, GC, Buf, FT, ST, KS},
    asm::DOFBasedCOOAssembler,
    kernel::DarcyMixedHex8RT0P0Kernel,
    mesh::Mesh{8, Hex8},
) where {T, B, IPS, E <: AbstractElement, GC, Buf, FT, ST, KS}
    @assert length(fvec) == cache.ndofs (
        "apply_load!: f has length $(length(fvec)); expected $(cache.ndofs)")

    _prepare_caches!(cache, mesh)

    layout = local_dof_layout(E)
    element_caches = cache.element_caches
    geometry_caches = cache.geometry_caches
    nelem = length(element_caches)

    @inbounds for pid in eachindex(load.panels)
        eid, lf = load.panels[pid]
        (1 ≤ eid ≤ nelem) || throw(ArgumentError("MixedDarcyHex8BoundaryNormalFluxLoad: elem_id $eid out of range 1:$nelem"))
        (1 ≤ lf ≤ 6) || throw(ArgumentError("MixedDarcyHex8BoundaryNormalFluxLoad: local_face $lf out of range 1:6"))

        gc = geometry_caches[eid]
        ec = element_caches[eid]
        X = gc.X
        length(X) == 8 || continue

        fc = faces(Hex8())[lf]
        vs = fc.vertices
        @inbounds p1 = X[Int(vs[1])]
        @inbounds p2 = X[Int(vs[2])]
        @inbounds p3 = X[Int(vs[3])]
        @inbounds p4 = X[Int(vs[4])]

        orient = Float64(hex8_face_outward_sign(X, lf))

        @inbounds for gpt in REF_GAUSS_QUAD_2X2
            ξf = gpt[1]
            ηf = gpt[2]
            wf = gpt[3]
            ξv = _hex8_face_volume_ref_coords(ξf, ηf, lf)
            dN_dξ = get_basis_derivatives(Hex8(), Lagrange{1}(), ξv)
            J = X[1] ⊗ dN_dξ[1]
            for a in 2:8
                J += X[a] ⊗ dN_dξ[a]
            end
            detJ = det(J)
            detJ == 0.0 && continue

            dN_face = get_basis_derivatives(Quad4(), Lagrange{1}(), Vec{2}((ξf, ηf)))
            tξ = dN_face[1][1] * p1 + dN_face[2][1] * p2 + dN_face[3][1] * p3 + dN_face[4][1] * p4
            tη = dN_face[1][2] * p1 + dN_face[2][2] * p2 + dN_face[3][2] * p3 + dN_face[4][2] * p4
            jac_met = norm(tξ × tη)
            jac_met == 0.0 && continue
            # Normal direction must match [`hex8_face_outward_sign`](@ref) (first triangle on
            # `faces(::Hex8)`); ‖tξ×tη‖ is still the correct surface Jacobian for the quad map.
            cross_face = (p2 - p1) × (p3 - p1)
            nf = norm(cross_face)
            nf == 0.0 && continue
            n_unit = orient * cross_face / nf
            dS_w = jac_met * wf

            base = load.g * dS_w
            @inbounds for ifi in 1:6
                ψ = rt0_hex8_reference_basis(ifi, ξv)
                φ = piola_contravariant(J, ψ)
                li = _mixed_darcy_flux_li(layout, ifi)
                li > 0 || error("MixedDarcyHex8BoundaryNormalFluxLoad: no flux DOF for local face $ifi")
                gdof = Int(ec.dofs[li])
                fvec[gdof] += base * dot(φ, n_unit)
            end
        end
    end

    return fvec
end

# ----- core face integration ------------------------------------------------

# Look up the global DOF index for (node_id, component) given the
# *concrete* field-starts vector for field 1. Hoisted outside the hot
# loops so we only pay one type-assertion per `apply_load!` call,
# never per face / per component.
@inline _node_field_dof(field_starts1::Vector{Int}, node_id::Int, comp::Int) =
    @inbounds field_starts1[node_id] + (comp - 1)

# One-face integration: writes f[d] += traction · N_i · dS · w.
# Specialised on NN (2, 3, or 4) so the inner loops fully unroll. Takes the
# pre-resolved `field_starts1::Vector{Int}` so the inner DOF lookup is
# fully concrete-typed (no allocations regardless of the cache's
# `DOFHandler` abstract field).
@inline function _integrate_face!(f::AbstractVector{Float64},
                                  field_starts1::Vector{Int},
                                  nodes_xyz,
                                  face_nodes::NTuple{NN,Int},
                                  t,
                                  ::Val{NN}) where {NN}
    n_comp = _t_comp_count(t)

    # Pull face corner coordinates onto the stack as a small NTuple.
    # Avoids any heap allocation in the hot loop.
    X = ntuple(i -> nodes_xyz[face_nodes[i]], NN)

    if NN == 4
        @inbounds for (ξ, η, w) in REF_GAUSS_QUAD_2X2
            ξv = Vec{2}((ξ, η))
            N = get_basis_functions(Quad4(), Lagrange{1}(), ξv)
            dN = get_basis_derivatives(Quad4(), Lagrange{1}(), ξv)
            tξ = dN[1][1] * X[1] + dN[2][1] * X[2] + dN[3][1] * X[3] + dN[4][1] * X[4]
            tη = dN[1][2] * X[1] + dN[2][2] * X[2] + dN[3][2] * X[3] + dN[4][2] * X[4]
            dS_w = norm(tξ × tη) * w
            for li in 1:NN
                Ni = N[li] * dS_w
                node = face_nodes[li]
                for comp in 1:n_comp
                    d = _node_field_dof(field_starts1, node, comp)
                    f[d] += _t_component(t, comp) * Ni
                end
            end
        end
    elseif NN == 3
        @inbounds for (ξ, η, w) in REF_GAUSS_TRIANGLE_CENTROID
            ξv = Vec{2}((ξ, η))
            N = get_basis_functions(Tri3(), Lagrange{1}(), ξv)
            dN = get_basis_derivatives(Tri3(), Lagrange{1}(), ξv)
            tξ = dN[1][1] * X[1] + dN[2][1] * X[2] + dN[3][1] * X[3]
            tη = dN[1][2] * X[1] + dN[2][2] * X[2] + dN[3][2] * X[3]
            dS_w = norm(tξ × tη) * w
            for li in 1:NN
                Ni = N[li] * dS_w
                node = face_nodes[li]
                for comp in 1:n_comp
                    d = _node_field_dof(field_starts1, node, comp)
                    f[d] += _t_component(t, comp) * Ni
                end
            end
        end
    elseif NN == 2
        @inbounds p1 = X[1]
        @inbounds p2 = X[2]
        L = norm(p2 - p1)
        L == 0.0 && return nothing
        half = L / 2
        @inbounds for (ξ, w) in REF_GAUSS_SEGMENT_ORDER2
            N = get_basis_functions(Seg2(), Lagrange{1}(), Vec{1}((ξ,)))
            dS_w = half * w
            for li in 1:NN
                Ni = N[li] * dS_w
                node = face_nodes[li]
                for comp in 1:n_comp
                    d = _node_field_dof(field_starts1, node, comp)
                    f[d] += _t_component(t, comp) * Ni
                end
            end
        end
    else
        error("_integrate_face!: unsupported face node count NN=$NN (expected 2, 3, or 4)")
    end
    return nothing
end

"""
    apply_load!(f::AbstractVector{Float64}, load::SurfaceLoad,
                cache::DOFBasedCOOCache, asm, kernel, mesh) -> f

Surface-traction / heat-flux integration: `f[i, α] += ∫_Γ N_i · t_α dS`
for every face in `load.faces`. Uses the cache's `dof_connectivity`
parent `DOFHandler` to map `(face_node, component)` to global DOF
indices, and the mesh's nodal coordinates to compute the surface
Jacobian on the fly. Allocation-free after warmup.
"""
function apply_load!(f::AbstractVector{Float64},
                     load::SurfaceLoad{NN,V},
                     cache::DOFBasedCOOCache,
                     asm::DOFBasedCOOAssembler,
                     kernel::AbstractKernel,
                     mesh::AbstractMesh) where {NN,V}
    @assert length(f) == cache.ndofs (
        "apply_load!: f has length $(length(f)); expected $(cache.ndofs)")

    # The cache pre-resolves `dof_handler.field_starts[1]` into a
    # concrete `Vector{Int}` field at construction so the hot loop
    # below stays allocation-free even though `cache.dof_handler` is
    # stored at the abstract `DOFHandler` type.
    field_starts1 = cache.field_starts1
    nodes_xyz = mesh.nodes
    nfaces = length(load.faces)

    @inbounds for fi in 1:nfaces
        face_nodes = load.faces[fi]
        t = _face_traction(load.traction, fi)
        _integrate_face!(f, field_starts1, nodes_xyz, face_nodes, t, Val(NN))
    end

    return f
end

"""
    apply_load!(f, load::SurfaceScalarFluxOnField, cache, asm, kernel, mesh) -> f

Same quadrature as [`SurfaceLoad`](@ref) for scalar `traction`, using
`dof_handler.field_starts[load.field_index]` for global DOF indices.
"""
function apply_load!(
    f::AbstractVector{Float64},
    load::SurfaceScalarFluxOnField{NN,V},
    cache::DOFBasedCOOCache,
    asm::DOFBasedCOOAssembler,
    kernel::AbstractKernel,
    mesh::AbstractMesh,
) where {NN,V}
    _depwarn_redundant_kernel_arg!(:apply_load!)
    @assert length(f) == cache.ndofs (
        "apply_load!: f has length $(length(f)); expected $(cache.ndofs)",
    )
    nfs = length(cache.dof_handler.field_starts)
    (load.field_index ≤ nfs) || throw(
        ArgumentError(
            "SurfaceScalarFluxOnField: field_index=$(load.field_index) exceeds " *
            "handler field count ($nfs)",
        ),
    )
    field_starts = cache.dof_handler.field_starts[load.field_index]::Vector{Int}
    nodes_xyz = mesh.nodes
    nfaces = length(load.faces)

    @inbounds for fi in 1:nfaces
        face_nodes = load.faces[fi]
        t = _face_traction(load.traction, fi)
        _integrate_face!(f, field_starts, nodes_xyz, face_nodes, t, Val(NN))
    end

    return f
end
