# src/basis/

Shape-function families and the symbolic generator that produces them.

The module separates two ideas that classical FE texts often bundle:

- topology owns reference geometry, node count and connectivity
  (`src/topology/`),
- basis owns interpolation: how a function on the reference element is
  expressed as a linear combination of shape functions.

`get_basis_functions(::Type{Topo}, ::Type{Basis}, xi)` and
`get_basis_derivatives(...)` are the two entry points used everywhere
else in the package.

## Files

- `api.jl`                  — `AbstractBasis`, `Lagrange{P}`,
  `Serendipity{P}`, the public `get_basis_functions` /
  `get_basis_derivatives` API and the `VandermondeBasisDescription`
  struct used by the generator.
- `basis_descriptions.jl`   — catalogue of `VandermondeBasisDescription`
  entries (one per element type the generator should produce).
- `vandermonde.jl`          — generator-time Vandermonde matrix helper.
  Not included by the runtime package path; loaded only when running
  `basis_generator.jl`.
- `subs.jl`                 — generator-time symbolic substitution and
  simplification helper. Not included by the runtime package path; loaded
  only when running `basis_generator.jl`.
- `basis_generator.jl`      — offline tool that turns the catalogue into
  generated code. Run with
  `julia --project=. src/basis/basis_generator.jl`.
- `basis_generated.jl`      — emitted by the generator; this is the file
  that runtime code actually loads. Do not edit by hand.

Specialised bases that do not fit the Vandermonde pattern live outside
this directory next to the physics that needs them. An older NURBS
prototype was archived to `llm/design/legacy-revival/iga/`; the DKT
plate-bending prototype was retired (see
`llm/sessions/` for the corresponding session log).

## Quick start

```julia
using JuliaFEM
using Tensors

topo  = Tetrahedron{10}
basis = Lagrange{2}
xi    = Vec(0.2, 0.3, 0.1)

N  = get_basis_functions(topo, basis, xi)    # SVector{10, Float64}
dN = get_basis_derivatives(topo, basis, xi)  # SVector{10, Vec{3, Float64}}

u_at_xi = dot(node_values, N)
grad_u  = sum(node_values[i] * dN[i] for i in 1:10)
```

The dispatch is on types, not instances; the API also accepts instances
for convenience and forwards via `typeof`.

## Currently generated catalogue

The generator emits methods for the following element types
(`src/basis/basis_descriptions.jl`):

- 1D:           Seg2, Seg3
- 2D triangles: Tri3, Tri6
- 2D quads:     Quad4, Quad8 (serendipity), Quad9
- 3D tets:      Tet4, Tet10
- 3D hexes:     Hex8, Hex20 (serendipity), Hex27
- 3D pyramid:   Pyr5
- 3D wedges:    Wedge6, Wedge15

## Vandermonde generator

For a chosen reference topology with nodes `x_k` and an ansatz
`{p_j(xi)}`, the generator solves the Vandermonde system

    V * a_i = e_i,    V[k, j] = p_j(x_k)

for each basis function `N_i`, then applies symbolic differentiation and
simplification, and emits a specialised method per `(Topology, Basis)`
pair into `basis_generated.jl`. Each emitted method is a tiny inlined
function that returns `SVector{N, Float64}` or
`SVector{N, Vec{D, Float64}}` and runs in a few nanoseconds with no
allocations.

The whole pipeline runs offline; runtime code never depends on
symbolic-math packages.

## Adding a new generated element

1. Add the element's reference coordinates to the corresponding
   topology under `src/topology/` (skip if the topology already exists).
2. Add a `VandermondeBasisDescription` entry to
   `src/basis/basis_descriptions.jl` listing the new ansatz.
3. Re-run the generator:

   ```bash
   julia --project=. src/basis/basis_generator.jl
   ```

4. Commit both `src/basis/basis_descriptions.jl` and the regenerated
   `src/basis/basis_generated.jl`.

For specialised bases that do not fit the Vandermonde pattern (Nédélec,
plate-bending shape functions, hierarchical bases) the recommended
pattern is a direct implementation that defines its own
`get_basis_functions` / `get_basis_derivatives` methods, alongside the
generator output. There is currently no in-tree example; place such a
basis next to the physics that consumes it (e.g. `src/domains/plates/`)
when one is added.

## Performance notes

The generated code is fully type-stable and zero-allocation. Sample
single-evaluation timings on a recent x86 laptop (Julia 1.10):

- Tri3   basis:  about 1.2 ns
- Tri6   basis:  about 2.1 ns
- Tet4   basis:  about 1.5 ns
- Tet10  basis:  about 3.6 ns
- Hex8   basis:  about 2.8 ns
- Hex27  basis:  about 8.9 ns

Derivative evaluations are roughly 1.5x to 2x more expensive. All
methods return `SVector` instances and are typically inlined by the
caller, so the cost essentially disappears inside an integration loop.

## Related code

- `src/topology/`     — reference shapes and `reference_coordinates`,
  the single source of truth used by the generator.
- `src/quadrature/`   — integration points consumed alongside the basis
  evaluations.
- `src/geometry/`     — Jacobians and physical-space derivatives.
- `test/topology/`    — basis correctness is exercised indirectly via
  the topology tests and `test/elements/test_interpolate_local_fields.jl`.
  A standalone basis suite lives at `llm/design/legacy-tests/basis/`
  pending a rewrite that drops its `include("../../src/...")` lines.
