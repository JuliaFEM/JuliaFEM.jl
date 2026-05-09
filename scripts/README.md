# scripts/

Standalone helper scripts that are not part of the package code and not
exercised by the test suite. They are run manually during development.

## Current scripts

### `coverage.jl`

Runs the test suite under `--code-coverage=user`, summarises the
resulting `.cov` files, and writes `coverage/lcov.info` for editor
gutters and Codecov-style tooling.

```bash
# Full pass (re-runs the test suite, ~50 - 60 s).
julia scripts/coverage.jl

# Re-summarise the .cov files left behind by a previous run, no tests.
julia scripts/coverage.jl --summary-only

# Run the legacy mode in addition (JULIAFEM_ENABLE_LEGACY=1).
julia scripts/coverage.jl --legacy

# CI-style threshold gate (exit non-zero when below 95 %).
julia scripts/coverage.jl --threshold 95

# Show more files in the per-file breakdown.
julia scripts/coverage.jl --top 60
```

The reporter prints two headline numbers:

- `Total coverage (all src/)` — covers everything under `src/`,
  including the `src/legacy/` tree that is gated behind
  `JULIAFEM_ENABLE_LEGACY=1`. Default test runs never load the legacy
  module, so its lines always weigh in as 0/N.
- `Live coverage (excl legacy)` — drops `src/legacy/` from the
  denominator. This is the more meaningful number for the active
  codebase.

Coverage tooling lives in its own environment at
`scripts/coverage/Project.toml` (only `Coverage.jl`) so the package's
runtime and test deps stay clean. The first run instantiates that env
on demand.

### `check_layer_contract.jl`

Static audit for the dependency directions described in
`docs/src/developer/architecture_layers.md`. Fails if forbidden patterns
appear under `src/domains/` (layer C) or under layer A directories
(`topology`, `quadrature`, `geometry`, `basis`, `sparse`). Uses only Base;
CI runs this on every job.

```bash
julia scripts/check_layer_contract.jl
```

### `check_namespace_collisions.jl`

Walks the loaded `JuliaFEM` module and reports symbol-name collisions
with the standard library and other commonly-used packages. Useful when
adding new exports or before merging large refactors.

```bash
julia --project=. scripts/check_namespace_collisions.jl
```

### `fix_vendor_element_types.py`

One-off cleanup script (Python) for normalising element-type names
inherited from older vendor packages. Kept for reference; not expected
to be re-run.

## Related machinery elsewhere

The Lagrange basis generator that some older notes refer to as
`scripts/generate_lagrange_basis.jl` is now an in-tree file under
`src/basis/`:

- `src/basis/basis_generator.jl` performs the symbolic generation and
  emits `src/basis/basis_generated.jl`.
- Run it directly with `julia --project=. src/basis/basis_generator.jl`
  whenever a basis description in `src/basis/basis_descriptions.jl`
  changes.

MPI regression drivers live under `test/mpi/` (`partitioned_matvec_smoke.jl`,
`partitioned_matvec_cg.jl`). They expect a throwaway project with `MPI.jl`
installed; the exact `julia -e '…'` incantation matches
`.github/workflows/CI.yml` (job `mpi-partitioned-matvec-smoke`).

See `src/basis/README.md` for the complete design and extension guide.
