# src/elements/

The `Element{K, P, S, N}` template plus the DOF-extraction and
field-interpolation utilities used by the assembly kernels.

## Mathematical background

A finite element is, in Ciarlet's sense, a triple `(K, P, Σ)` where

- `K`  is the reference domain (a topology),
- `P`  is the local approximation space (a basis family), and
- `Σ`  is a set of linear functionals (degrees of freedom).

In the implementation `Σ` is encoded by a field specification `S` that
together with `K` and `P` uniquely determines the functionals for the
standard Lagrange / Serendipity families. `S` is the DOFSet built by the
`@DOFSet` macro in `src/dofs/`.

## The element type

```julia
struct Element{K<:AbstractTopology, P<:AbstractBasis, S<:DOFSet, N}
    id::UInt
    dof_indices::NTuple{N, UInt64}
end
```

- `K` and `P` are types — no runtime fields.
- `S` is a NamedTuple type whose values are `DOF{Quantity, Entity}`
  (see `src/dofs/README.md`).
- `N` is the total number of local DOFs (computed by the constructor).
- `dof_indices` is a flat tuple of global DOF indices in the order
  defined by `local_dof_layout(::Type{Element{K, P, S, N}})`.

Use `create_elements!(mesh, Element{K, P, S})` to build a
`Vector{Element{K, P, S, N}}` together with a `DOFHandler` that already
carries the inverse DOF connectivity.

## Compile-time DOF layout

`local_dof_layout(::Type{Element{K, P, S, N}})` is a `@generated`
function returning `NTuple{N, DOFLayoutEntry}`; each entry exposes
`field_idx`, `entity_local`, `component`. The compiler folds the result
into a constant at the call site, so DOF decoding inside hot loops is a
tuple lookup with no arithmetic.

```julia
S  = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ET = Element{Hex8, Lagrange{1}, S, 24}
local_dof_layout(ET)
```

## Building elements

### Single-field

```julia
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ET = Element{Tetrahedron{4}, Lagrange{1}, S}

elements, handler = create_elements!(mesh, ET)
```

### Multi-field

```julia
S = @DOFSet{T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}}

elements, handler = create_elements!(mesh, Element{Tetrahedron{4}, Lagrange{1}, S})
```

In both cases `dof_indices` is a flat `NTuple` whose ordering is dictated
by `local_dof_layout`.

## DOF extraction from a global vector

Two extraction strategies are provided. Both are zero-allocation and
type-stable.

### Flat extraction

```julia
dofs = extract_element_dofs(elem, u_global)
# (u = (1.0, 2.0, ..., 12.0),)
```

### Structured extraction

Reinterprets the values into the field's quantity type so that they can
be combined with shape-function values directly.

```julia
dofs = extract_element_dofs_structured(elem, u_global)
# (u = (Vec{3}(1,2,3), Vec{3}(4,5,6), ...),)

u_at_xi = N1 * dofs.u[1] + N2 * dofs.u[2] + N3 * dofs.u[3] + N4 * dofs.u[4]
```

## Field-block ranges (multi-field)

Per-field local index ranges are computed at compile time and are useful
for picking out coupling sub-blocks of an element matrix:

```julia
T_range = field_dof_range(elem, :T)   # 1:4 for Tet4 + Vertex
u_range = field_dof_range(elem, :u)   # 5:16
K_Tu    = K_local[T_range, u_range]
```

## Type queries

```julia
topology_type(elem)  # K
basis_type(elem)     # P
dof_type(elem)       # S
n_element_dofs(elem) # N
nnodes(elem)         # nnodes(K)
```

## Files

- `elements.jl`              — element type, constructors, type queries,
  `local_dof_layout`.
- `extract_element_dofs.jl`  — flat and structured DOF extraction.
- `interpolate.jl`           — field interpolation at points
  (`interpolate_field`, `interpolate_fields`,
  `interpolate_field_value`, `interpolate_local_fields`).

## Related code

- `src/dofs/README.md`       — DOFSet, `DOF{Q, E}` and `DOFHandler`.
- `src/topology/`            — `Triangle`, `Tetrahedron`, `Hexahedron`,
  topological entities.
- `src/basis/README.md`      — basis families and interpolation API.
- `src/assemblers/`          — assemblers consume `local_dof_layout`
  and the element DOF tuples.
- `test/elements/`           — public test suite.
