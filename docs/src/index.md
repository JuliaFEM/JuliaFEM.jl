# [JuliaFEM.jl](@id home)

JuliaFEM.jl is an open-source finite element framework written in Julia.
The package is **0.x**; the repository is in the middle of a deliberate
architectural reset toward a **stable 1.0** with a type-stable, zero-allocation,
GPU-friendly assembly pipeline.

**Broader documentation** (book, user/developer guides, examples) lives in the
same repository under **`juliafem.github.io/`** (Quarto). Start from
**`juliafem.github.io/learning_path.md`** and **`juliafem.github.io/docs/documentation-map.md`**
to see how this Documenter site relates to that material.

This page is a short, current-API quick start. The
[API Reference](@ref) lists exported symbols.

For a maintainer-oriented summary of the current architecture and the
non-negotiable invariants, see `AGENTS.md` in the repository root.
For per-module developer notes, see the `README.md` files under
`src/<topic>/`.
For a logical layer diagram and an explicit dependency contract between
those layers, see [Architecture layers](@ref).

## Installation

```julia
using Pkg
Pkg.add("JuliaFEM")
```

## A modern minimal example

The following sets up a unit-cube linear-elasticity problem, builds an
`Element{K, P, S, N}` template with a compile-time DOF layout, assembles
the stiffness via the matrix-free-friendly DOF-based assembler, and
extracts the assembled `K` and right-hand side `f`.

The listing is included verbatim from
`docs/src/snippets/minimal_elasticity_quickstart.jl`. That file is executed in
`Pkg.test()` (`test/docs/runtests.jl`) and in CI (`scripts/verify_docs_quickstart.jl`)
so the example cannot drift from the package.

```@literalinclude
snippets/minimal_elasticity_quickstart.jl
```

## Multi-field elements

`@DOFSet` accepts more than one field, and the rest of the pipeline is
multi-field aware (see `local_dof_layout` and the thermo-elastic kernel
in `src/domains/thermo_elastic/`). A longer walkthrough with a runnable
block lives on [Thermo-elastic walkthrough](@ref thermo_elastic_walkthrough).

```julia
S = @DOFSet{T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}}
```

## Matrix-free path

The same `cache` and `kernel` drive the matrix-free
`apply_K!` / `apply_M!` operators in `src/assemblers/`:

- Dirichlet:                   `PenaltyDirichlet`, `EliminatedDirichlet`
- Linear MPC:                  `LinearMPC`
- Neumann loads:               `NodalForce`, `UniformBodyForce`,
                               `SurfaceLoad`
- Preconditioners:             `JacobiPreconditioner`,
                               `BlockJacobiPreconditioner`,
                               `ICholPreconditioner`
- Generalized eigensolve:      `lowest_eigenpairs`, `solve_eigenproblem`

`matrix_free_op(cache, asm, kernel, mesh; dirichlet, mpc)` returns a
closure that wraps `apply_K!` with constraint hooks; it composes with
any `LinearOperator`/Krylov stack.

## Inspecting the compile-time DOF layout

`local_dof_layout(::Type{Element{K,P,S,N}})` is a `@generated` function
that returns an `NTuple{N, DOFLayoutEntry}` describing
`(field_idx, entity_local, component)` for each local DOF. The compiler
folds it into a constant at the call site, so DOF decoding is a tuple
lookup with no runtime arithmetic.

```julia
S  = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ET = Element{Hex8, Lagrange{1}, S, 24}
local_dof_layout(ET)
```

## Where to look next

- Elements, DOFs, traits, mixed fields:
                        [Elements and multiphysics](@ref elements_multiphysics_teaser)
- Coupled thermo-elasticity (multi-field):
                        [Thermo-elastic walkthrough](@ref thermo_elastic_walkthrough)
- Assembler trade-offs:
                        [Choosing an assembler](@ref assembler_choice)
- Legacy API (optional `Legacy` submodule):
                        [Legacy module](@ref legacy_module)
- API reference:        [API Reference](@ref)
- Repository layout:    [Repository layout](@ref) (see also the pointer
                        `docs/repository_layout.md` on GitHub).
- Changelog (0.x):       `docs/NEWS.md` in the project root.
- Architecture:         `AGENTS.md` in the project root, plus
                        `src/README.md` for a per-module overview.
- Tests as documentation:
                        `test/assemblers/test_dof_based_*.jl`,
                        `test/assemblers/test_eigensolve.jl`,
                        `test/assemblers/test_linear_mpc.jl`,
                        `test/assemblers/test_surface_load.jl`,
                        `test/assemblers/test_ichol_preconditioner.jl`.
