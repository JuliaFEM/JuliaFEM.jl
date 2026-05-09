# `src/legacy/` — optional pre-reset API

Everything here loads only when **`JULIAFEM_ENABLE_LEGACY=1`** is set before
`using JuliaFEM`. Symbols live under **`JuliaFEM.Legacy`** and are not
re-exported from `JuliaFEM`.

## What is inside (include order from `Legacy.jl`)

1. **`dcti_dvti_fields.jl`** — Dict-backed field containers (`DCTI`, `DVTI`, …).
2. **`elements_lagrange.jl`** — legacy `Element(Poi1, …)` style constructors.
3. **`linear_system.jl`** — sparse system helpers used by old solvers.
4. **`core_types.jl`** — legacy node / integration-point types.
5. **`assembly_problems.jl`** — `Problem` / field problem hierarchy.
6. **`analysis.jl`** — analysis drivers and writers.
7. **`deprecated_fembase.jl`** — compatibility shims with warnings.
8. **`problems_dirichlet.jl`** — legacy Dirichlet boundary drivers.
9. **`solvers.jl`** — solver stack.
10. **`io/`** — Abaqus keyword register, mesh/model parsers, surface helpers, download shim.
11. **`deprecations.jl`** — small `assemble!` shims and the `Abaqus` helper module.

## Modern replacements (0.x)

| Legacy concept | Current direction |
|----------------|-----------------|
| Dict field bags, `register_fields!` | `@DOFSet`, `DOFHandler`, `create_elements!` |
| `Problem` / `Analysis` drivers | `AbstractKernel` + assemblers + external solvers |
| `Assembly` multi-mesh container | One `Mesh` per region or explicit coupling code |
| Old mesh readers | `read_gmsh_mesh` in active `src/io/` where available; Abaqus/Aster here |

See also package **`docs/src/legacy.md`** and **`juliafem.github.io/docs/migration-and-versioning.md`**.
