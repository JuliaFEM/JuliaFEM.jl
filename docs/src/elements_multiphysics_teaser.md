# [Elements, DOF placement, materials, and multiphysics](@id elements_multiphysics_teaser)

This page is a conceptual teaser for the current (0.x) element pipeline: what the
types mean, how global numbering attaches to an element, how materials
hook in, and how a deliberately mixed DOF specification still follows one
mechanical pattern.

For API listings, see [API Reference](@ref). For maintainer detail and
invariants, see `AGENTS.md` and the `README.md` files under `src/elements/`
and `src/dofs/`.

## Element as a typed template

A finite element is Ciarlet's triple ``(K, P, \Sigma)`` carried almost
entirely in the type system:

| Role | Julia type |
|------|----------------|
| Reference domain ``K`` | Topology type, e.g. `Tetrahedron{4}`, `Hex8` |
| Local space ``P`` | Basis type, e.g. `Lagrange{1}` |
| Degrees of freedom ``\Sigma`` | `S`, a `@DOFSet{...}` (NamedTuple of `DOF{Quantity, Entity}`) |
| Total local DOF count | Type parameter `N`, inferred from `K` and `S` |

The runtime struct is minimal:

```julia
struct Element{K, P, S, N}
    id::UInt
    dof_indices::NTuple{N, UInt64}
end
```

`K` and `P` are not stored; they only refine the type. The only heavy
per-element data is the flat tuple `dof_indices`, one global DOF index per
local DOF, in the canonical order defined below.

## Where each DOF "lives"

You declare fields with `@DOFSet`:

```julia
S = @DOFSet{
    T::DOF{Temperature, Vertex},
    u::DOF{Displacement{3}, Vertex},
    p::DOF{Float64, Cell},   # discontinuous scalar (e.g. pressure bubble)
}
```

Each entry is `DOF{Quantity, Entity}`:

- `Quantity` fixes the number of components per attachment point
  (scalar temperature, 3-vector displacement, …) via `dof_size`.
- `Entity` selects the topological anchor: `Vertex`, `Edge`, `Face`, or
  `Cell`.

The constructor for `Element{K,P,S,N}` checks that `N` equals `ndofs(K, S)`.

`create_elements!(mesh, Element{K,P,S})` builds a `DOFHandler` and fills
each element's `dof_indices` so shared mesh entities reuse the same global
indices. You do not hand-wire indices by hand in normal use.

## Compile-time layout: `local_dof_layout`

The order of entries in `dof_indices` is defined by
`local_dof_layout(::Type{Element{K,P,S,N}})`, a `@generated` function that
returns `NTuple{N, DOFLayoutEntry}`. Each entry stores
`(field_idx, entity_local, component)` for one local DOF index.

Fields are visited in NamedTuple key order. For each field, all vertex
DOFs (if `Entity === Vertex`) are enumerated in local vertex order, then
components; then the next field. A `Cell`-anchored field contributes one
logical "entity" worth of components per element (pressure bubble, …).

Assemblers and matrix-free operators index this tuple instead of doing
runtime `div`/`mod` arithmetic to decode local DOFs.

Current limitation: the generated `local_dof_layout` implements `Vertex`
and `Cell` attachments. `Edge` and `Face` are part of the long-term design
(documented in `src/dofs/fields.jl`) but are not wired through the
generator yet; adding them is a localized extension.

## Materials: traits and kernels

Materials subtype `AbstractMaterial`. Several open functions form a small
trait layer, for example:

- `material_behavior(material)` — constant tangent, strain-dependent,
  stateful plasticity, …
- `supported_physics(material)` — tuple of physics markers such as
  `Elasticity{3}()`, `Thermal{3}()`
- `required_state_variables(material)` — what must live in IP state
  storage

Concrete materials implement these in their own files; see
`src/materials/traits.jl` and `src/materials/api.jl`.

Important separation: the element type `S` describes which global DOFs exist
and how they are ordered. A kernel decides which physics runs at a
quadrature point and which material(s) supply tangents and forces.

Example: `ContinuumKernel` pairs one elastic material with a displacement
field. `ThermoElasticKernel` (under `src/domains/thermo_elastic/`) holds
both a mechanical and a thermal material; `evaluate_entry` dispatches on
the `field_idx` pair from `DOFLayoutEntry` so `K_uu`, `K_TT`, and the
coupling blocks are assembled from one element loop. That is the intended
pattern for multiphysics: one `S`, one element storage model, kernels that
know the physics blocks.

## A deliberately mixed specification

Thermo-mechanics with temperature and displacement on vertices is the
simplest coupled story (four DOFs per node on a tetrahedron). A harsher
stress test for the DOF machinery is vertex displacement plus a
discontinuous cell pressure (Stokes-style `u`–`p`):

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex},
            p::DOF{Float64, Cell}}
```

The test file `test/dofs/test_multifield_dof_system.jl` walks this path
end-to-end: `ndofs` matches `local_dof_layout`, block ordering (all `u`
DOFs, then all `p` DOFs), handler connectivity, and helpers such as
`field_dof_range` / `element_dofs` aligned with the layout tuple.

Adding more named fields is mostly declaring another `DOF{…, …}` line in
`@DOFSet` and providing a kernel that implements the cross-blocks you care
about. The expensive part is physics and testing, not extending the
element struct.

## Summary

- Elements are types plus a flat global index tuple; `S` is the single
  source of truth for DOF count and ordering.
- `create_elements!` maps mesh topology to those indices in a type-stable
  way.
- Materials expose traits; kernels connect materials to the relevant
  `field_idx` blocks via `local_dof_layout`.
- Multifield and mixed `Vertex`/`Cell` anchoring already exercise the
  production DOF path; richer entity attachments follow the same layout
  idea once implemented.
