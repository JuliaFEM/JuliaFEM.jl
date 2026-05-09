# docs/

Sources for the JuliaFEM.jl documentation site, plus a small set of
kept-on-purpose historical material.

## Layout

- `src/`: Documenter-built documentation source.
  - `index.md`: landing page and a current-API quick start.
  - `developer/`: maintainer-facing notes (for example architecture layers).
  - `repository_layout.md`: where to place new files in the repository.
  - `api.md`: auto-extracted API reference.
  - `figs/`: diagrams referenced from the docs.
- `make.jl`: Documenter build script. From the repository root:
  `julia --project=docs -e 'using Pkg; Pkg.instantiate()'` once, then
  `julia --project=docs docs/make.jl`. This uses **`docs/Project.toml`** (includes
  **Documenter**); the main package environment does not need Documenter for normal
  `Pkg.test()`.
- `Project.toml`, `Manifest.toml`: build environment for the docs.
- `NEWS.md`: short changelog for 0.x releases; keep bullets factual.
- `CONTRIBUTING.md`: short contributor quick-start. The full
  contributor-facing description of architecture, invariants and
  workflow lives in the repository root `AGENTS.md`.
- `repository_layout.md`: pointer to `src/repository_layout.md` (full layout guide).
- `tutorials/`: Jupyter notebooks from 2015-2016. They predate the
  current API and are kept only as a historical record. They are not
  built into the documentation site and are not exercised by CI.
- `logo/`: JuliaFEM logo assets.
- `build/`: generated output of `make.jl` (gitignored).
- `deploy.jl`: deployment helper.

## What does not live here

- Session logs, design notes, vision documents, planning checklists and
  AI-assistant working files belong under `llm/` (gitignored), not in
  `docs/`. See [`src/repository_layout.md`](src/repository_layout.md)
  for the full decision tree (or the pointer [`repository_layout.md`](repository_layout.md)).
- Per-module developer notes live next to the code as `src/<topic>/README.md`.
- The authoritative current architecture summary for any agent or new
  contributor is `AGENTS.md` in the repository root.
