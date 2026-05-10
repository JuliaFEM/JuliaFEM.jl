# src/mesh/

Mesh data structures, mesh generation utilities and a few graph-level
optimisations used by the rest of the package.

## Files

- `api.jl`            — `AbstractMesh`, `AbstractRefineStrategy` and the
  abstract interface (`nnodes_total`, `nelements`, `get_node`,
  `connectivity_matrix`, `get_node_set`, `get_element_set`, …).
- `mesh.jl`           — concrete `Mesh{N, T<:AbstractTopology{N}}`
  type, constructors with validation, the inverse connectivity
  (`node -> elements`) needed by node-based assembly, helpers for sets
  and surface extraction.
- `structured.jl`     — `create_structured_box_mesh` (`Hex8` or `Quad4`),
  `create_structured_line_mesh` (`Seg2`), `create_unit_cube_mesh`,
  `create_cook_membrane_mesh` (`Quad4`, Cook skew panel),
  `create_cantilever_mesh`, `create_thin_plate_mesh`. Boundary node sets are
  populated automatically (`:xmin`, `:xmax`, …).
- `refine.jl`         — `LongestEdgeBisection` and the `refine` entry
  point. Used for h-convergence studies.

## Type-stable mesh

```julia
Mesh{N, T<:AbstractTopology{N}}
```

`N` is the number of nodes per element and `T` is the element topology.
Concrete instances such as `Mesh{8, Hex8}` carry no abstract fields, so
the assembly hot path can be fully inferred. The mesh stores

- `nodes::Vector{Vec{3, Float64}}` — coordinates (always 3D; 2D problems
  use `z = 0`),
- `connectivity::Vector{NTuple{N, UInt32}}` — fixed-size connectivity,
- `element_sets::Dict{Symbol, Set{UInt32}}`,
- `node_sets::Dict{Symbol, Set{UInt32}}`,
- `inverse_connectivity::Vector{Vector{Tuple{UInt32, UInt8}}}` —
  populated lazily; needed by node-based assembly.

## Common usage

```julia
mesh = create_structured_box_mesh(Hex8;
    xmin = 0.0, xmax = 1.0, nx = 4,
    ymin = 0.0, ymax = 1.0, ny = 4,
    zmin = 0.0, zmax = 1.0, nz = 4,
)

n_nodes = nnodes_total(mesh)
n_elems = nelements(mesh)

X       = get_node(mesh, 42)
fixed   = get_node_set(mesh, :xmin)
body    = get_element_set(mesh, :body)
elem_xs = get_elements_for_node(mesh, node_id)
```

## Refinement

```julia
refined = refine(mesh, LongestEdgeBisection(3))
```

`LongestEdgeBisection(levels)` performs adaptive bisection along the
longest edge of each element, doubling the element count per level.

## I/O

Mesh import from external tools (Gmsh, Netgen, Abaqus, …) is **not** part of
the core load path; use optional extensions or companion packages that build
`Mesh{…}` (see `src/io/README.md`). For **Gmsh**, load `Gmsh.jl` after JuliaFEM
so `JuliaFEMGmshExt` activates, then call [`read_gmsh_msh`](@ref) or
[`mesh_from_current_gmsh_model`](@ref) (`src/mesh/gmsh_stub.jl`). VTK / XDMF
output is not implemented in the new path; legacy results writers under
`src/legacy/` cover older tests.

## Related code

- `src/topology/`         — element topologies referenced by `Mesh{N,T}`.
- `src/dofs/`             — the DOF handler walks the mesh to assign
  global DOF numbers.
- `src/assemblers/`       — element-based and DOF-based assembly both
  consume the mesh and its inverse connectivity.
- `test/`                 — mesh-level coverage rides on the active
  domain test suites (`test/domains/`, `test/assemblers/`); the legacy
  parallel/coloring/RCM tests live at
  `llm/design/legacy-tests/mesh/`.
