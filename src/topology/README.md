# src/topology/

Reference shapes for finite elements: parametric coordinates, edge and
face connectivity, topological entities. Topology owns geometry; basis
functions and quadrature are separate concerns and live in
`src/basis/` and `src/quadrature/`.

## Files

- `api.jl`            — `AbstractTopology{N}`, the topological-entity
  hierarchy (`TopologicalEntity`, `Vertex`, `Edge`, `Face`, `Cell`),
  the entity-counting helpers (`nvertices`, `nedges`, `nfaces`) and the
  generic interface (`nnodes`, `dim`, `reference_coordinates`, `edges`,
  `faces`).
- `topology.jl`       — small compatibility shim included by
  `JuliaFEM.jl`.
- `segments.jl`       — `Segment{N}` plus the `Seg2`, `Seg3` aliases.
- `triangles.jl`      — `Triangle{N}` plus `Tri3`, `Tri6`, `Tri7`,
  `Tri10`.
- `quadrilaterals.jl` — `Quadrilateral{N}` plus `Quad4`, `Quad8`,
  `Quad9`.
- `tetrahedra.jl`     — `Tetrahedron{N}` plus `Tet4`, `Tet10`.
- `hexahedra.jl`      — `Hexahedron{N}` plus `Hex8`, `Hex20`, `Hex27`.
- `pyramids.jl`       — `Pyramid{N}` plus `Pyr5`.
- `wedges.jl`         — `Wedge{N}` plus `Wedge6`, `Wedge15`.

## Topologies provided

The full set of currently exported aliases:

| Family         | Type signature        | Aliases                  |
|----------------|------------------------|---------------------------|
| Segment (1D)   | `Segment{N}`           | `Seg2`, `Seg3`            |
| Triangle (2D)  | `Triangle{N}`          | `Tri3`, `Tri6`, `Tri7`, `Tri10` |
| Quad (2D)      | `Quadrilateral{N}`     | `Quad4`, `Quad8`, `Quad9` |
| Tetrahedron    | `Tetrahedron{N}`       | `Tet4`, `Tet10`           |
| Hexahedron     | `Hexahedron{N}`        | `Hex8`, `Hex20`, `Hex27`  |
| Pyramid        | `Pyramid{N}`           | `Pyr5`                    |
| Wedge / Prism  | `Wedge{N}`             | `Wedge6`, `Wedge15`       |

## Interface

Every topology implements:

```julia
nnodes(::Triangle{6})              # 6
dim(::Triangle{6})                 # 2
reference_coordinates(::Tri6)      # SVector{6, Vec{2, Float64}}
edges(::Tri6)                      # NTuple of (i, j) corner pairs
faces(::Tet4)                      # NTuple of (i, j, k) corner triples
```

The functions return compile-time constants (`nnodes`, `dim`) or static
data (`SVector`, `NTuple`) so they are inlined at the call site and add
no allocations to the assembly hot path.

Reference coordinate conventions:

- Segments        — `xi in [-1, 1]`.
- Triangles       — `(u, v)` with `u, v >= 0` and `u + v <= 1`.
- Quadrilaterals  — `(u, v) in [-1, 1]^2`.
- Tetrahedra      — `(u, v, w) >= 0` with `u + v + w <= 1`.
- Hexahedra       — `(u, v, w) in [-1, 1]^3`.

## Pairing topology and basis

Topology and basis are independent type parameters of the element
template. The basis dispatches on its own type parameter only:

```julia
get_basis_functions(::Type{Triangle{6}}, ::Type{Lagrange{2}}, xi)
```

so the same `Triangle{6}` can be combined with any compatible basis
family (`Lagrange{2}`, `Serendipity{2}`, …) without redefining the
topology.

## Adding a new topology

1. Create the file under `src/topology/` (or extend an existing family
   with a new `N`).
2. Define the `struct` and any aliases.
3. Implement `nnodes`, `dim`, `reference_coordinates`, `edges` and
   `faces`. Use `SVector` / `NTuple` returns; keep them allocation
   free.
4. Add the export to `src/JuliaFEM.jl`.
5. Add unit tests under `test/topology/`.
6. If you also want a Lagrange basis on this topology, add the
   matching `VandermondeBasisDescription` to
   `src/basis/basis_descriptions.jl` and re-run
   `julia --project=. src/basis/basis_generator.jl`.

## Related code

- `src/basis/`        — basis functions and the symbolic generator.
- `src/quadrature/`   — integration points for each topology.
- `src/mesh/`         — `Mesh{N, T<:AbstractTopology{N}}`.
- `src/elements/`     — `Element{K, P, S, N}` template.
- `test/topology/`    — public test suite.
