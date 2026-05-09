# src/quadrature/

Gauss-Legendre quadrature rules for every supported topology. The API
returns static data (`SVector{N, QuadraturePoint{D, Float64}}`) so
integration loops in the assemblers are inlined and allocation-free.

## Files

- `api.jl`                   — public types and interface
  (`AbstractQuadratureRule`, `GaussLegendre{N, V}`,
  `QuadraturePoint{D, T}`, `get_quadrature_points`, `default_quadrature`).
- `quaddata.jl`              — 1D Gauss-Legendre point and weight
  tables.
- `gl_tensor_product.jl`     — segments, quadrilaterals, hexahedra
  (tensor products of `quaddata`).
- `gl_triangles.jl`          — triangle rules.
- `gl_tetrahedra.jl`         — tetrahedron rules.
- `gl_wedges.jl`             — wedge / prism rules.
- `gl_pyramids.jl`           — pyramid rules.
- `gauss.jl`                 — `integration_points(topology_instance)` and
  `npoints(topology_instance)` thin wrappers over `default_quadrature` +
  `get_quadrature_points`, used by kernels that iterate with a concrete
  topology value rather than a type.

## Quick start

```julia
using JuliaFEM
using Tensors

points = get_quadrature_points(Hexahedron, GaussLegendre{2}())
# SVector{8, QuadraturePoint{3, Float64}}

for qp in points
    xi = qp.coords    # Vec{3, Float64}
    w  = qp.weight    # Float64
    # ... evaluate basis at xi and accumulate ...
end
```

## Type parameters

`GaussLegendre{N, V}`:

- `N` is the order in the mathematical sense — points per dimension on
  tensor-product elements, or a rule index on simplices.
- `V` is a variant tag (`:default`, `:B`, …) used where a topology has
  more than one rule of the same order.

`QuadraturePoint{D, T}` carries a `Vec{D, T}` of parametric coordinates
(deliberately a Tensors.jl `Vec`, for compatibility with the basis
functions and Jacobian code) and a scalar weight.

## Default rules

```julia
default_quadrature(1)                 # GaussLegendre{2}()
default_quadrature(Hexahedron, 1)     # GaussLegendre{2}()
```

The convention is `GaussLegendre{basis_order + 1}()`, which integrates
the stiffness contribution of the corresponding Lagrange basis exactly.

## Available rules

Tensor-product elements (segments, quads, hexes) provide
`GaussLegendre{1}` through `GaussLegendre{5}` (1, 2, 3, 4, 5 points per
direction). Triangles provide rules of total degrees 1 through 6 with
some `:B` variants. Tetrahedra provide rules of degrees 1 through 4
(degree 4 keeps all weights positive). Wedges and pyramids each
provide a small handful of rules; see the corresponding `gl_*.jl`
file for the exact list.

## Adding a rule

1. Implement the rule in the appropriate `gl_*.jl` file. Keep the
   method `@inline`, return an `SVector` of `QuadraturePoint`, and use
   `Vec{D}` (not `SVector{D}`) for the coordinates.
2. Add coverage under `test/quadrature/` for point count, weight sum
   and polynomial exactness.
3. If you are extending `default_quadrature`, update the dispatch in
   `api.jl`.

## Related code

- `src/topology/`        — the topologies these rules are attached to.
- `src/basis/`           — basis functions evaluated at the quadrature
  points.
- `src/geometry/`        — Jacobian-weighted integration.
- `src/assemblers/`      — every assembler iterates over these point
  lists.
- `test/quadrature/`     — public test suite.
