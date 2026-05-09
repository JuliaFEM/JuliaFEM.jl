# src/

Top-level layout of the JuliaFEM source tree. The package is **0.x**; the
repository is in the middle of a deliberate architectural reset toward a
**stable 1.0**. The directories below list what is currently active and where
the boundary with the legacy code lies.

For the canonical, up-to-date architecture summary read `AGENTS.md` in
the repository root. For per-module details, follow the README files
inside each subdirectory.

## Active modules

| Directory       | Responsibility                                                                                                              |
|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| `topology/`     | Reference shapes and topological entities (`Triangle`, `Tetrahedron`, `Hexahedron`, `Vertex`, `Edge`, `Face`, `Cell`, …).    |
| `basis/`        | Shape-function families (`Lagrange{P}`, `Serendipity{P}`, plate bases) plus the symbolic generator that produces them.       |
| `quadrature/`   | Gauss-Legendre integration points for every supported topology.                                                              |
| `geometry/`     | Jacobians, physical derivatives, strain tensors.                                                                             |
| `mesh/`         | Type-stable `Mesh{N, Topo}`, structured/circular generators, refinement, RCM ordering, gmsh wrapper.                         |
| `dofs/`         | `DOF{Quantity, Entity}`, `@DOFSet`, `DOFHandler`, DOF connectivity and the field/dof-extraction infrastructure.              |
| `elements/`     | `Element{K, P, S, N}` template, compile-time `local_dof_layout`, DOF-extraction and field-interpolation utilities.           |
| `materials/`    | `LinearElastic`, `NeoHookean`, `PerfectPlasticity`, `HeatConductivity` and the trait-based dispatch used by the kernels.     |
| `assemblers/`   | Assembly caches, `ElementBasedAssembler` (COO/CSC) and `DOFBasedCOOAssembler`, the matrix-free `apply_K!`/`apply_M!` path.   |
| `domains/`      | Physics kernels per discipline: `continuum/`, `heat/`, `thermo_elastic/`. See `domains/README.md`; each subdirectory implements an `AbstractKernel`. |
| `physics/`      | `AbstractPhysics` tag (`Elasticity{Dim}`, `Thermal{Dim}`), microkernel contract for the DOF-based path, formulation helpers and strain extraction. |
| `io/`           | Mesh readers (Gmsh by default; Abaqus + Aster live in the optional `Legacy` submodule).                                      |
| `sparse/`       | Lightweight sparse-matrix utilities (DOK / COO).                                                                             |
| `fields/`       | `AbstractField` (`Displacement`, `Temperature`, `DisplacementRotation`) and the `LocalField` interpolation type.             |

## Top-level files

- `JuliaFEM.jl` is the package entry point. It pulls in everything above
  in dependency order and is the source of truth for the export list.

## Legacy

`legacy/` contains the older pre-reset framework code (`Problem` / `Solver` /
`Analysis`, the Dict-based field types `DCTI` / `DVTI` / `DCTV` / `DVTV`,
the `Element(Poi1, ...)` constructor family, the Abaqus mesh-reader
stack, the FEMBase-style `update!` shims, …). It is wrapped in
`module Legacy` and is loaded only when the environment variable
`JULIAFEM_ENABLE_LEGACY=1` is set at module-load time. Names live under
`JuliaFEM.Legacy.<name>` and are not re-exported at the top level.

The legacy gate is evaluated when the precompile cache is built, so
switching modes requires `Pkg.precompile()` after toggling the variable
(or deleting the cache under `~/.julia/compiled/v$VERSION/JuliaFEM/`).

New development should not extend `legacy/`.

## Conventions

- File names are `lowercase_with_underscores.jl`.
- Modules avoid in-source markdown documents; sub-READMEs are the only
  exception. Architectural notes and design proposals belong in `llm/`.
- Hot-path code must remain zero-allocation and type-stable; the
  regression for this lives in
  `test/assemblers/test_dof_based_zero_alloc.jl`.
- Greek letters are not used in code identifiers (`u`, `v`, `w`
  for reference coordinates). They are fine in comments and docstrings.
