# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Physics-level interface stubs.

Older JuliaFEM versions had a `Physics{Formulation, Field, Mesh, Material}` struct
that owned `assemble!` / `solve!` / `add_dirichlet!` / `add_neumann!` as the
top-level user surface. The current architectural reset replaced that high-level surface with
the `AbstractKernel` + assembler + matrix-free hooks design that lives under
`src/assemblers/`. The four function names below are kept here only as
top-level placeholders so that:

  - the `JuliaFEM.Legacy` submodule (when loaded via `JULIAFEM_ENABLE_LEGACY=1`)
    can hang its method definitions on the same generic functions;
  - the assemblers (`src/assemblers/element_based/element_based_coo.jl`,
    `src/assemblers/dof_based/dof_based_coo.jl`) can extend `assemble!` with
    methods that match the current cache-based workflow.

There is no `Physics(...)` constructor in the active build. The current
end-to-end flow is documented in `AGENTS.md` and exercised by
`test/runtests.jl`.
"""

"""
    assemble!(...)

Generic assembly entry point. Concrete methods are added by the element-based
and DOF-based assemblers under `src/assemblers/`.
"""
function assemble! end

"""
    solve!(...)

Reserved name. The default 0.x build does not provide a top-level `solve!` method;
solvers are composed externally via `MatrixFreeOperator` + Krylov methods or
via direct factorisation of the assembled matrix. Methods on this name only
exist inside `JuliaFEM.Legacy`.
"""
function solve! end

"""
    add_dirichlet!(...)

Reserved name. In the default 0.x build, Dirichlet conditions are expressed as
`PenaltyDirichlet` / `EliminatedDirichlet` value objects passed to the
matrix-free operator. Methods on this name only exist inside `JuliaFEM.Legacy`.
"""
function add_dirichlet! end

"""
    add_neumann!(...)

Reserved name. In the default 0.x build, Neumann loads are expressed as
`NodalForce` / `UniformBodyForce` / `SurfaceLoad` value objects applied via
`apply_load!`. Methods on this name only exist inside `JuliaFEM.Legacy`.
"""
function add_neumann! end
