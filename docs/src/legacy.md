# [Legacy module](@id legacy_module)

JuliaFEM.jl is **0.x**; the default `using JuliaFEM` surface is the type-stable
pipeline described in `AGENTS.md` (`Element{K,P,S,N}`, `DOFHandler`, domain
kernels, modern assemblers).

Older tutorials, notebooks, and helper scripts target a **previous** API:
`Problem` / `Assembly` / `Solver` / `Analysis`, Dict-based fields, legacy element
constructors, and some mesh readers (for example Abaqus `.inp` and Code Aster
`.med`). That code lives under `src/legacy/` inside `module Legacy`.

## Enabling Legacy

Set the environment variable **`JULIAFEM_ENABLE_LEGACY=1`** before the first
`using JuliaFEM` in a Julia session so the `Legacy` submodule is loaded.
Symbols are namespaced as `JuliaFEM.Legacy.<name>` and are **not** re-exported
from `JuliaFEM`.

## When to use it

- Porting old scripts or reproducing historical results.
- Reading meshes or models that still depend on legacy constructors.

For new work, prefer the current API and the Quarto site map at
`juliafem.github.io/docs/documentation-map.md`.
