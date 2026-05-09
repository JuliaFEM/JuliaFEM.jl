# JuliaFEM.jl package tests

The active tree mirrors `src/`. `test/runtests.jl` includes one topic bundle per
row in the `TOPICS` array:

- `basis`, `fields`, `physics`, `elements`, `materials`, `topology`
- `domains/continuum`, `domains/heat`, `domains/thermo_elastic`
- `dofs`, `assemblers`, `validation`, `reference`

Each topic directory contains its own `runtests.jl` that aggregates `@testset`
files in that folder.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

GitHub Actions runs the full suite on Julia 1.10 and latest stable, and a
parallel **assemblers-only** job (latest stable, no coverage) via
`test_args` — see `.github/workflows/CI.yml` job `test-assemblers`.

Filter to one or more topics via `test_args` (strings must match `TOPICS`
in `test/runtests.jl`, e.g. `assemblers`, `dofs`, `domains/heat`):

```bash
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["assemblers"])'
```

Same filtering when invoking the runner directly:

```bash
julia --project=. test/runtests.jl assemblers
julia --project=. test/runtests.jl assemblers dofs
```

## Older / Legacy tests

Specs that still target removed APIs or experimental layouts may live under
`llm/design/legacy-tests/` (often gitignored). They are **not** picked up by
`test/runtests.jl`.

## Related

- `test/materials/README.md` — materials-specific layout.
- Website-only notes under `juliafem.github.io/test/` are **not** this suite.
