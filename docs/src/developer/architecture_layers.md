# Architecture layers

This page states the intended dependency direction between parts of
JuliaFEM.jl. It is a maintainer contract: it describes how the code is
meant to fit together. Julia does not enforce these edges the way
C++ include paths or separate link libraries do; the main concrete anchor
today is the ordered `include` list in `src/JuliaFEM.jl` (comments
`Level 1` … `Level 8`).

For motivation and a similar three-layer pattern in another project, see
[OpenPFC package architecture](https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/concepts/architecture.md)
(kernel → runtime → frontend, with an include audit on the kernel tree).

## Logical layers

JuliaFEM is organised into four logical layers. Names are chosen to stay
close to common FEM vocabulary and to parallel OpenPFC-style thinking
without implying a one-to-one directory rename.

```mermaid
flowchart TB
  subgraph discretisation [Discretisation and algebra]
    topo[topology]
    quad[quadrature]
    geom[geometry]
    basis[basis]
    sparse[sparse]
  end
  subgraph model [Model and DOF template]
    fields[fields]
    dofs[dofs]
    elems[elements]
    mats[materials]
    mesh_api[mesh/api abstract mesh]
  end
  subgraph contracts [Assembly contracts]
    kint[abstract + microkernel]
  end
  subgraph physics [Physics kernels]
    dom[domains kernels and cache updaters]
  end
  subgraph drivers [Drivers and I/O]
    mesh_impl[mesh/mesh + structured + refine]
    dof_h[dof_handler + dof_connectivity]
    asm[assemblers traversal caches matrix_free]
    io[io e.g. gmsh_reader]
    legacy[legacy optional]
  end
  drivers --> physics
  drivers --> model
  physics --> contracts
  physics --> model
  physics --> discretisation
  model --> discretisation
  contracts --> model
  contracts --> discretisation
```

### Layer A — Discretisation and algebra

Reference cells, integration rules, shape-function machinery, small
geometry helpers, and sparse scratch types used by assembly.

Typical directories: `src/topology/`, `src/quadrature/`, `src/geometry/`,
`src/basis/`, `src/sparse/`.

These should stay free of mesh file formats, solvers, and concrete
`Mesh` implementations beyond what is already in early API stubs.

### Layer B — Model and DOF template

Field quantities, DOF sets, the `Element{K,P,S,N}` template,
compile-time `local_dof_layout`, material models and traits, and abstract
mesh interfaces (`src/mesh/api.jl`).

Typical directories: `src/fields/`, `src/dofs/` (excluding
`dof_handler.jl` until it is pulled in with concrete mesh), `src/elements/`,
`src/materials/`, plus early mesh API tags.

### Layer C — Physics kernels

Discipline-specific `AbstractKernel` implementations, weak-form and
microkernel entry points, and continuum helpers that update per-element
caches (`update_*_cache!` under `src/domains/`).

This layer implements what to integrate. It may depend on layers A
and B and on the contracts in `src/assemblers/abstract.jl` (kernel
defaults) and `src/assemblers/microkernel.jl` (DOF-based microkernel
trait). It should not depend on concrete mesh readers, optional I/O, or
the legacy module.

Typical directories: `src/domains/`, `src/physics/` (type tags and shared
strain helpers used by kernels), and the two contract files above.

### Layer D — Drivers and I/O

Concrete `Mesh` types, `DOFHandler`, assemblers (element-based and
DOF-based), matrix-free operators, mesh generators, and readers.

Typical directories: `src/mesh/mesh.jl`, `src/mesh/structured.jl`,
`src/mesh/refine.jl`, `src/dofs/dof_handler.jl`,
`src/dofs/dof_connectivity.jl`, `src/assemblers/` (except the small
contract files already attributed to layer C), `src/io/`, optional
`src/legacy/`.

KernelAbstractions-based code (for example `dof_based_coo_ka.jl`) belongs
here: it chooses how and where assembly runs, analogous to a
“runtime” port in OpenPFC’s sense, not the physics statement of the weak
form.

## Allowed dependency table

The row may import or call into the column when the cell says yes.

| From / To | A Discretisation | B Model / template | C Physics | D Drivers / I/O |
|------------|:----------------:|:------------------:|:-----------:|:---------------:|
| A          | yes              | no                 | no          | no              |
| B          | yes              | yes                | no          | no              |
| C          | yes              | yes                | yes         | no              |
| D          | yes              | yes                | yes         | yes             |

“Import” here means direct file-level coupling: new `using`/`import`,
`include`, or tight type coupling that would force layer D to compile when
working only on layer A.

`JuliaFEM.Legacy` is optional and must not be required by layers A–C.
When legacy is enabled, it may depend on symbols defined anywhere in the
main module, but new code should not add dependencies from A–C into
`src/legacy/`.

## How this relates to `src/JuliaFEM.jl`

The include manifest orders work so that, in the common case, types and
functions used by layer C already exist when domain files load, while
concrete mesh and DOF-handler code loads afterward. If you add a new file,
place it in the directory that matches its layer, then insert
`include(...)` at the lowest level that still allows the module to
load (preferably next to related files).

Note: `src/assemblers/` holds both early contracts (`abstract.jl`,
`microkernel.jl`) and some concrete drivers (for example
`element_based/element_based_coo.jl`) that load before `src/domains/`
because the module is linear. Layer C code must still depend only on
contracts and shared caches, not on the internals of a specific
assembler implementation. New domain code must not reach into
element-based or DOF-based traversal details.

When domain kernel files use `using ..JuliaFEM: ...`, they pull names from
the partially-built module. That is convenient for dispatch but weak as a
boundary: reviewers should still ask whether a new dependency belongs in
layer C or only in D.

## Enforcement (today and next steps)

Compliance relies on review, on the include order in `JuliaFEM.jl`, and on a
small static audit:

- From the repository root:  
  `julia scripts/check_layer_contract.jl`  
  This fails if inner-layer trees gain forbidden references (for example Gmsh
  I/O or KernelAbstractions under `src/domains/`, or mesh drivers under layer A
  directories). CI runs the same command on every matrix job.

Further steps (not implemented yet):

- Move optional heavy or device-specific stacks behind
  [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-code-in-packages-(Extensions))
  so a minimal install stays small and dependencies stay explicit.

## See also

- `AGENTS.md` in the repository root — invariants (zero-allocation hot
  paths, type stability, element template as source of truth).
- `src/README.md` — directory map.
- [Home](@ref home) — user-facing quick start on this documentation site.
