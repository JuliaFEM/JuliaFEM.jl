# AGENTS.md: Quick guide for AI coding agents working on JuliaFEM.jl

This file is the entry point for any AI agent (Cursor, Copilot, Codex, …)
that opens this repository. Read it first; then jump to the linked,
authoritative documents for details.

The package version is **0.x** (see `Project.toml`); the repository is in the
middle of a deliberate architectural reset toward a **stable 1.0** (monorepo,
type-stable, zero-allocation, GPU-ready). Many older files, README.md sections,
and design notes still describe the *previous* API. When in doubt, trust the
code and this file over older READMEs.

---

## 1. What this project is

JuliaFEM.jl is an open-source finite element framework written in Julia.
It is being rebuilt from first principles around four ideas:

1. Ciarlet's triple as a type. `Element{K, P, S, N}` encodes the
   reference domain `K`, polynomial space `P`, DOF specification `S`
   and total DOF count `N` purely at the type level. The element
   instance carries only `id` and `dof_indices::NTuple{N, UInt64}`.
2. Element as template, not bag of fields. Heavy structural
   information (which local DOF is which field/entity/component) is
   produced by `@generated` functions over the element type, so the
   compiler can fold it into constants. The canonical example is
   `local_dof_layout(::Type{Element{K,P,S,N}})` returning an
   `NTuple{N, DOFLayoutEntry}`.
3. Microkernels. Assembly is built from small `evaluate`-style
   functions that compute a single scalar (or block) and are
   dispatched at compile time. No `Dict`, no `Any`, no boxing.
4. Zero-allocation hot paths. Pre-allocated caches plus
   trait-based material dispatch let the inner loops run with zero GC
   allocation sites in the optimized LLVM IR.

The long-term vision (parallelism, GPU, Newton–Krylov, multiphysics,
billions of DOF) is documented in [`llm/vision/vision_2.0.md`](llm/vision/vision_2.0.md).
The filename is historical only; version numbering here is 0.x toward a stable 1.0, and that note captures stretch goals beyond the first stable release rather than a separate product line.

---

## 2. Where things live

The authoritative file-organization guide is
[`docs/src/repository_layout.md`](docs/src/repository_layout.md). Highlights:

| Folder | What goes there |
|---|---|
| `src/topology/` | Reference shapes (`Triangle`, `Tetrahedron`, `Hexahedron`, …). |
| `src/basis/` | Shape functions (`Lagrange`, `Serendipity`). |
| `src/quadrature/` | Integration rules. |
| `src/mesh/` | `Mesh{N,Topo}`, structured/unstructured meshes. |
| `src/dofs/` | `DOF{Q,E}`, `@DOFSet`, `DOFHandler` (type-stable), DOF connectivity. |
| `src/elements/` | `Element{K,P,S,N}` + `local_dof_layout`. |
| `src/materials/` | `LinearElastic`, `PerfectPlasticity`, …; trait-based dispatch. |
| `src/assemblers/` | `ElementBasedAssembler`, `DOFBasedCOOAssembler`, caches, scatter routines. |
| `src/domains/` | Physics kernels per discipline (`continuum/`, `beams/`, `plates/`, …). |
| `src/solvers/` | Linear/nonlinear solvers (still minimal). |
| `src/physics/` | High-level user API (BCs, problems). |
| `src/io/` | Mesh readers (Gmsh by default; Abaqus + Aster live in `Legacy`). |
| `src/legacy/` | Older pre-reset API (`Problem`/`Assembly`/`Solver`/`Analysis`, Dict-based fields, Abaqus reader, …) wrapped in `module Legacy`; loaded only when `JULIAFEM_ENABLE_LEGACY=1`. |
| `test/<topic>/` | Mirrors `src/<topic>/`. |
| `benchmarks/regression/`, `benchmarks/analysis/` | Timestamped reports. |
| `docs/src/` | Documenter-built `index.md` and `api.md`. |
| `docs/NEWS.md` | Short 0.x changelog (not the Quarto site). |
| `docs/tutorials/` | Historical Jupyter notebooks (2015–2016 API; reference only). |
| `prototypes/`, `.trash/`, `llm/` | Gitignored: experiments, throwaway, AI session logs. |

The per-module READMEs under `src/<topic>/` and the user-facing pages
under `docs/src/` were rewritten on 2026-05-08 to match the current
type-stable API. If you find a document that still references `DOFManager`,
`register_fields!`, `count_field_dofs`, `@NamedTuple{u::Tuple{...}}`,
the legacy `Physics`/`DirichletBC`/`add_dirichlet!` API or the
nonexistent `docs/contributor/` tree, treat it as stale and update or
delete it in the same change.

---

## 3. Current architecture (0.x)

### 3.1 DOF specification

```julia
# Single field
S = @DOFSet{u::DOF{Displacement{3}, Vertex}}

# Multi-field (e.g. thermo-mechanical)
S = @DOFSet{T::DOF{Temperature, Vertex},
            u::DOF{Displacement{3}, Vertex}}

# `S` is a NamedTuple type whose values are `DOF{Quantity, Entity}`.
```

`Quantity` is anything with a defined `dof_size` (`Float64 → 1`, `Vec{3} → 3`,
`Tensor{2,3} → 9`, …). `Entity` is one of `Vertex`, `Edge`, `Face`, `Cell`.

### 3.2 DOF handler

`DOFHandler{M, S, NF}` (in `src/dofs/dof_handler.jl`) is the type-stable
replacement for the legacy `Dict`-based `DOFManager`:

- One flat `Vector{Int}` per field stores the starting DOF for each
  entity ID.
- `total_dofs` is computed once at element creation.
- A `@generated _make_element_dofs(...)` unrolls the
  field/entity/component loop at compile time, so building
  `element.dof_indices` is allocation-free.
- `DOFManager` is kept as a backward-compat alias.

```julia
elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})
# handler isa DOFHandler{Mesh{...}, S, NF}
# elements::Vector{Element{Hex8, Lagrange{1}, S, 24}}
# handler.dof_connectivity::DOFConnectivity already built
```

### 3.3 Element template (compile-time DOF layout)

`local_dof_layout(::Type{Element{K,P,S,N}})` is a `@generated` function
returning `NTuple{N, DOFLayoutEntry}` where each entry describes
`(field_idx, entity_local, component)` for one local DOF. The compiler
folds this into `Core.Const(...)` at the call site, so DOF decoding
becomes a tuple lookup with no runtime arithmetic.

Use `field_idx`, `entity_local`, `component` accessors (exported).

### 3.4 Assemblers

Two production assemblers, both type-stable and zero-allocation after
warmup:

- `ElementBasedAssembler` (`src/assemblers/element_based/element_based_coo.jl`):
  classical element-by-element assembly, scatter into COO triplets.
  Gold standard for correctness.
- `DOFBasedCOOAssembler` (`src/assemblers/dof_based/dof_based_coo.jl`):
  walks DOF rows, dispatches into per-element scratch using
  `local_dof_layout(E)`. Stepping stone toward a GPU/matrix-free
  pipeline. Currently single-kernel, `ContinuumKernel`-only.

### 3.5 Materials and caches

- `AbstractMaterial` → behavior trait (`StatelessConstantTangent`,
  `StatelessStrainDependent`, `StatefulStrainDependent`).
- `compute_stress(material, ε, state, t) → (σ, 𝔻, new_state)`.
- Caches:
  - `GeometryCache`: coords, `∇N`, `detJ·w`.
  - `ElementCache`: element-level scratch (DOF mapping, IPs).
  - `AssemblyMaterialWorkspace{FieldType, StateType}`: per-IP
    NamedTuples of fields and states (type-stable, zero-allocation).
  - `GlobalMaterialCache`: state across timesteps.

### 3.6 Distributed matrix-free matvec (MPI)

`MPI` is a weak dependency; `using MPI` after `JuliaFEM` loads `JuliaFEMMPIExt`.
Partition metadata lives in `src/assemblers/partitioning.jl`, `halo_exchange.jl`,
and `packed_layout.jl` (`build_partition_packed_layout_for_matvec`,
`build_matvec_halo_exchanges`, `RankHaloExchange`, `PartitionPackedLayout`).

For Krylov solves without a full-length replicated workspace, use
`mpi_partitioned_operator_matvec_owned!` with `partitioned_mpi_owned_matvec_workspace`
and pass a persistent `mpi_requests` buffer from `allocate_exchange_matvec_halo_mpi_requests`
so each matvec does not allocate a fresh `Vector{MPI.Request}`.

Reference drivers: `test/mpi/partitioned_matvec_smoke.jl`,
`test/mpi/partitioned_matvec_cg.jl`. CI runs both under mpiexec (see
`.github/workflows/CI.yml`, job `mpi-partitioned-matvec-smoke`).

---

## 4. Invariants the agent must preserve

These are non-negotiable. Tests and code analysis enforce them.

1. Zero allocations in hot paths.
   `test/assemblers/test_dof_based_zero_alloc.jl` asserts
   `@allocated assemble!(...) == 0` and `0` GC allocation sites in the
   optimized LLVM IR. Don't introduce `Dict`, `Vector{Any}`, untyped
   closures, or `Vector` literals inside loops. For the intended split
   between tier 1 numeric kernels (always C-speed, no heap churn in
   loops), tier 2 warmed assembly drivers (setup may allocate), and tier 3
   IO/UI convenience code where flexible containers are acceptable, see
   `docs/src/developer/architecture_layers.md` (section Performance tiers).
2. Type stability everywhere on the hot path.
   `Base.promote_op(assemble!, ...) === Nothing` and the inferred
   types must be concrete. Use `NamedTuple` (typed), `NTuple`, and
   compile-time helpers, not `Dict`.
3. Element template == single source of truth.
   Anything you'd be tempted to compute as `div`/`mod` over local DOF
   indices probably belongs in a `@generated` function on
   `Element{K,P,S,N}` and queried via accessors like
   `local_dof_layout`.
4. Mirror src/ in test/. Tests for `src/foo/bar.jl` go in
   `test/foo/test_bar.jl`. A topic-level test file should be wired
   into `test/runtests.jl`.
5. Zero stdlib drift. New Julia stdlib deps (`InteractiveUtils`,
   `Profile`, …) must be added to `[extras]` and the relevant
   `[targets]` in `Project.toml`.

---

## 5. Critical workflow rules

Contributor-facing workflow text lives in `.github/prompts/commit.prompt.md`
and `.github/copilot-instructions.md`. Optional `.githooks/pre-commit`
(enabling `core.hooksPath`) caps staged paths at two per commit.
Editor-local Cursor rules under `.cursor/` are not part of the git tree.

- Commits: prefer **small** steps (default one file per commit). Combine
  multiple files only when they share one logical story (exception—justify
  why). Never `git add .` or `git add -A` unless the user asks; stage paths
  deliberately. Read the **full** staged diff (no `head`/`tail`). Message
  format: Conventional Commits **subject**, blank line, **1–3 sentence**
  summary, then bullets **scaled to the patch** (small change → few or none;
  large / multi-concern → grouped, substantive bullets). **Propose** paths + full message
  and wait for **explicit approval** before each `git commit`. See
  `.github/prompts/commit.prompt.md` for the full protocol.
- Never commit without an explicit user command to start committing. Do not
  propose or initiate commits unprompted.
- Never create files in the root except for `README*` files.
  Session logs go to `llm/sessions/YYYY-MM-DD-topic.md`. Throwaway
  scripts go to `.trash/`. Experimental code goes to `prototypes/`.
- `prototypes/`, `.trash/`, `llm/` are gitignored. Use freely; do
  not try to commit them.
- Run the full test suite before declaring done.
`julia --project=. -e 'using Pkg; Pkg.test()'`. All bundled topic tests must pass
with no errors (the exact test count grows with the tree under `test/`).
- SPDX tags at file tops must use comment syntax for that file type (for example
  `# …` in `.jl`, HTML comments in Markdown); see `docs/CONTRIBUTING.md`.

---

## 6. Documentation style

When writing documentation, session logs, READMEs, guides, comments, or
design notes:

- Avoid emoji.
- Avoid markdown bold for emphasis. Do not write `bold` prose unless
  preserving an existing quoted source that already uses it.
- Prefer plain technical writing with clear headings, short paragraphs,
  and precise code references.
- Do not over-emphasize ordinary terms. If every second word needs
  emphasis, the sentence should be rewritten instead.

The goal is professional engineering documentation, not marketing copy.

---

## 7. Common entry points

- Build a model and assemble:
  ```julia
  mesh = create_structured_box_mesh(...)
  S    = @DOFSet{u::DOF{Displacement{3}, Vertex}}
  ET   = Element{Hex8, Lagrange{1}, S}
  elements, handler = create_elements!(mesh, ET)

  material = LinearElastic(E=210e9, ν=0.3)
  kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                             material, Displacement{3}())

  asm   = DOFBasedCOOAssembler()           # or COOAssembler()
  cache = create_cache(asm, elements, handler, mesh, kernel)
  assemble!(cache, asm, kernel, mesh)
  K, f  = extract_system(cache)
  ```
- Inspect the compile-time DOF layout:
  ```julia
  using JuliaFEM
  S  = @DOFSet{u::DOF{Displacement{3}, Vertex}}
  ET = Element{Hex8, Lagrange{1}, S, 24}
  local_dof_layout(ET)   # NTuple{24, DOFLayoutEntry}
  ```
- Regression + code analysis:
  `test/assemblers/test_dof_based_zero_alloc.jl`
- Top-level architecture overview: `src/README.md`. Per-module
  details: the `README.md` files inside `src/<topic>/`.
- Vision and roadmap: `llm/vision/vision_2.0.md` (historical filename; not a release label).

### Documentation doors (which prose to open first)

| Goal | Start here |
|------|------------|
| Shortest runnable assembly in the package docs | `docs/src/index.md` and `docs/src/snippets/minimal_elasticity_quickstart.jl` |
| Ordered reading on the Quarto site | `juliafem.github.io/learning_path.md` |
| Authoritative versus archival site material | `juliafem.github.io/docs/documentation-map.md` |
| New domain kernel or assembler hook | `juliafem.github.io/docs/developer-guide/kernel_extension_contract.md` |
| Multi-field thermo-mechanical narrative | `docs/src/thermo_elastic_walkthrough.md` |
| Quarto site colors / logo / naming | `juliafem.github.io/docs/contributor-guide/design_system.md` |
| Where new files go in the repo | `docs/src/repository_layout.md` (pointer: `docs/repository_layout.md`) |

---

## 8. When you are unsure

1. Search the code first (`Grep` / `SemanticSearch`).
2. Check the relevant `test/<topic>/` for examples.
3. If there's a recent `llm/sessions/YYYY-MM-DD-*.md` log on the topic,
   read it for context.
4. Ask the user. Do not guess silently in a high-impact module.
