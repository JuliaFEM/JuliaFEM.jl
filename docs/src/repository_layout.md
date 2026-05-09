# Repository layout

Guide for contributors (and editor tooling) on where files belong in the JuliaFEM.jl tree. Read this together with [`AGENTS.md`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/AGENTS.md) (architecture and invariants) and [`src/README.md`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/src/README.md) (per-module overview).

Last updated: 2026-05-09.

---

## Organisational philosophy

Prefer subdirectories over flat structures.

When several files deal with the same topic, group them into a topic subdirectory:

- The subdirectory has its own `README.md` describing purpose and status.
- Related files stay together.
- An optional `Project.toml` keeps extra dependencies local instead of growing the main package manifest.

Examples:

- `src/materials/` for material models.
- `test/materials/` for matching tests.
- `benchmarks/regression/` for CI regression benchmarks and reports.
- `benchmarks/analysis/` for exploratory or architecture benchmarks.

Avoid flat directories with dozens of unrelated files.

---

## Source code

```text
src/
├── topology/         # Reference shapes (Triangle, Tetrahedron, Hexahedron, ...)
├── basis/            # Lagrange / Serendipity / DKT shape functions
├── quadrature/       # Integration rules
├── geometry/         # Jacobians, physical derivatives, strain helpers
├── dofs/             # DOF{Quantity, Entity}, @DOFSet, DOFHandler, connectivity
├── elements/         # Element{K, P, S, N} template + extraction / interpolation
├── materials/        # LinearElastic, NeoHookean, PerfectPlasticity, HeatConductivity, ...
├── assemblers/       # caches/, element_based/, dof_based/, matrix_free/, kernel_interface
├── domains/          # Per-physics kernels: continuum/, heat/, thermo_elastic/, plates/, ...
├── fields/           # AbstractField, Displacement, Temperature, LocalField
├── physics/          # Type tags + microkernel hooks (Elasticity, Thermal, ...)
├── mesh/             # Mesh{N, Topo}, structured / unstructured / refine / ordering
├── io/               # Gmsh-oriented I/O (legacy mesh readers live under legacy/)
├── sparse/           # SparseMatrixCOO / SparseVectorCOO scratch helpers
├── legacy/           # Pre-2.0 API in `module Legacy`; load with JULIAFEM_ENABLE_LEGACY=1
├── exports.jl        # Grouped public API exports
└── JuliaFEM.jl       # Module entry point
```

Rules:

- One concept per directory.
- File names: `lowercase_with_underscores.jl`.
- Aim for roughly 500 lines per file as a soft limit; split when a file grows far beyond that.
- No extra standalone `.md` files inside `src/<topic>/` beyond that topic's `README.md`. Per-symbol documentation lives in docstrings.

---

## Tests

```text
test/
├── runtests.jl           # Topic list and runner
├── README.md
├── testdata/             # Small meshes and fixtures
├── topology/, basis/, mesh/, elements/, fields/, materials/, ...
├── domains/              # continuum/, heat/, darcy/, thermo_elastic/, ...
├── assemblers/, dofs/, physics/, sparse/, io/, quadrature/, geometry/, ...
├── interface/, docs/, mpi/, backend/metal/
├── validation/, verification/, reference/
└── solvers/              # Reserved; see README there if present
```

Rules:

- Mirror `src/` where practical: tests for `src/foo/bar.jl` belong under `test/foo/`.
- Topic folders may include a `README.md` for scope and conventions.
- Test files: prefer `test_<feature>.jl`.
- Long-form comparison or verification methodology can live in topic `README.md` files; timestamp machine-generated reports as `reports/YYYY-MM-DD_HHMMSS_name.txt` under the relevant folder when you add them.
- Avoid dropping loose session write-ups under `test/`; use a gitignored notes area or a dated report path as above.

---

## Examples, demos, benchmarks

Some clones use optional top-level folders:

```text
examples/           # Full programs for users (often one subdirectory per example)
demos/              # Short API demonstrations for developers
benchmarks/
├── regression/     # CI-oriented performance checks
├── analysis/       # Exploratory benchmarks
└── reports/        # Timestamped outputs (YYYY-MM-DD_HHMMSS_*.txt)
```

These directories are not guaranteed to exist in every checkout; add them when you introduce new material.

Distinction:

- `examples/`: runnable end-to-end scripts, usually with local `README.md`.
- `demos/`: small focused scripts.
- `benchmarks/`: timing and allocation measurements; regression versus analysis split by purpose.

Rules:

- Timestamp benchmark output files under a `reports/` subdirectory.
- Group related benchmarks in the same subtree.

---

## Documentation

```text
docs/
├── CONTRIBUTING.md
├── README.md
├── repository_layout.md     # Short pointer to docs/src/repository_layout.md
├── make.jl
├── Project.toml
├── logo/
├── tutorials/               # Historical notebooks (older API; reference only)
└── src/
    ├── index.md
    ├── api.md
    ├── assembler_choice.md
    ├── elements_multiphysics_teaser.md
    ├── thermo_elastic_walkthrough.md
    ├── legacy.md
    ├── repository_layout.md   # This guide (built into the manual)
    ├── developer/
    │   └── architecture_layers.md
    └── snippets/             # Executable snippets for Documenter pages
```

The high-level architecture narrative lives in the repository root [`AGENTS.md`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/AGENTS.md). `docs/src/api.md` is generated from docstrings and exports.

What belongs in `docs/`:

- Documenter sources for the published manual (pages listed in `docs/make.jl`).
- Logo assets and docs-specific `Project.toml`.
- Contributor-facing notes such as `CONTRIBUTING.md`.
- This layout guide (`docs/src/repository_layout.md`) plus the short pointer `docs/repository_layout.md`.

What does not:

- Per-module developer notes that belong next to code: use `src/<topic>/README.md` instead.
- Large generated logs or scratch files: keep them out of `docs/src/` or timestamp under a dedicated reports location.

---

## Experimental and disposable paths

```text
prototypes/             # Gitignored in normal setups: experiments by topic
.trash/                 # Gitignored: pending deletion or scratch outputs
```

Rules:

- Use `prototypes/` for spike code you might promote into `src/` or delete.
- Use `.trash/` for short-lived files; delete once obsolete.

---

## Gitignored local notes (`llm/`)

Many developer setups keep a **local-only**, gitignored `llm/` tree for dated session logs and drafts. Nothing under `llm/` should be committed. If your checkout does not use it, you can ignore this section.

Naming convention when present: `llm/sessions/YYYY-MM-DD-topic.md` (lowercase, datestamped).

---

## Scripts

```text
scripts/
├── README.md
└── *.jl / *.sh          # Dev tooling, CI helpers, mesh utilities (not package code)
```

Rules:

- Scripts are not loaded by `using JuliaFEM`; they are run explicitly.

---

## Configuration

```text
.github/workflows/       # CI
.github/prompts/         # Maintainer prompts (e.g. commit message conventions)
Project.toml / Manifest.toml
```

Optional on a developer machine (gitignored): `.cursor/` for editor-specific agent rules—not committed.

---

## Anti-patterns

Avoid:

1. Session logs or narrative scratch files mixed into `test/` without a clear reports convention.
2. Backup suffixes (`.old`, `.bak`) committed next to active sources.
3. Log or coverage clutter in the repository root (prefer deletion or `.trash/`).
4. Hidden scratch files (`.cleanup_*`, `.temp_*`) under `src/` or `test/`.
5. Long-form design documents inside `src/` (keep module READMEs short; larger write-ups belong in `docs/` or local notes).
6. Untimestamped benchmark dumps at the top level of `benchmarks/` (use `reports/` with a timestamp prefix).

---

## Decision tree

Executable Julia code?

- Package implementation: `src/<topic>/<feature>.jl`.
- Test: `test/<topic>/test_<feature>.jl`.
- Example program: `examples/<name>/` when that tree exists.
- Demo script: `demos/` when that tree exists.
- Benchmark driver: `benchmarks/<topic>/`.
- One-off maintainer script: `scripts/`.
- Spike: `prototypes/<topic>/` or `.trash/` if disposable.

Documentation?

- Manual page: add `docs/src/...` and wire it in `docs/make.jl`.
- API reference: docstrings and exports (see `docs/src/api.md`).
- Layering rules: `docs/src/developer/architecture_layers.md`.
- Module orientation: `src/<topic>/README.md`.

Data?

- Small fixtures: `test/testdata/`.
- Large meshes: keep outside the repo or under a clearly marked local path.
- Benchmark output: `benchmarks/<topic>/reports/YYYY-MM-DD_HHMMSS_*.txt`.

Temporary?

- Spike code: `prototypes/<topic>/`.
- Pending cleanup: `.trash/`.

---

## Pre-commit checklist

1. No stray `.old`, `.bak`, `.tmp` in tracked trees.
2. Tests mirror `src/` layout where applicable.
3. Examples and benchmarks include README or timestamped reports as appropriate.
4. Module-facing docs updated in `src/<topic>/README.md` when public behaviour changes.
5. `.gitignore` continues to exclude `.trash/`, `prototypes/`, `.cursor/`, and local-only trees such as `llm/` when used.

---

## Golden rules

1. Prefer mirroring `test/` after `src/` for unit tests.
2. Keep generated and experimental trees gitignored unless deliberately checked in.
3. Treat [`AGENTS.md`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/AGENTS.md) as the architecture source of truth.
4. Prefer short per-module READMEs in `src/` over growing orphan markdown in random folders.

---

When in doubt:

1. File placement: re-read this page.
2. Architecture: read [`AGENTS.md`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/AGENTS.md).
3. Module conventions: read `src/<topic>/README.md`.
4. Maintainer automation: see `.github/copilot-instructions.md` (and local `.cursor/` if you use Cursor; it is gitignored).
5. Ask before introducing a new top-level directory.
