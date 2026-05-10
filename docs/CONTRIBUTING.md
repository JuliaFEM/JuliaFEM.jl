# Contributing to JuliaFEM.jl

Thank you for considering a contribution. Please read `AGENTS.md` in the
repository root before opening a pull request; it describes the current
architecture, the non-negotiable invariants (zero-allocation hot paths,
type stability, mirrored test/src layout) and the documentation style.

## Quick start

1. Fork and clone the repository.
2. Create a topic branch from `main`.
3. Make your change, keeping it small enough to review.
4. Run the full test suite from the repository root:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

5. Respect the layer dependency contract in
   `docs/src/developer/architecture_layers.md`. CI runs
   `julia scripts/check_layer_contract.jl`; run it locally before pushing if
   you touch `src/domains/` or foundation directories (`topology`,
   `quadrature`, `geometry`, `basis`, `sparse`).

6. Open a pull request with a clear description of the change and any
   relevant benchmark or test output.

## Time to first success

From a clean clone of JuliaFEM.jl, expect roughly this order:

1. `julia --project=. -e 'using Pkg; Pkg.instantiate()'` — resolve dependencies.
2. `julia --project=. -e 'using Pkg; Pkg.test()'` — full bundled suite (same
   command CI uses for the package tests).
3. Optional checks from the repository root:
   - `julia scripts/check_layer_contract.jl` — static layer dependency audit
     (same as CI).
4. Optional documentation smoke tests from the repository root:
   - `julia --project=. scripts/verify_docs_quickstart.jl` — keeps the
     Documenter minimal elasticity snippet aligned with the mesh in
     `docs/src/snippets/minimal_elasticity_quickstart.jl`.
   - `cd juliafem.github.io && julia scripts/check_website_docs.jl` — curated
     site Markdown links and patterns.
5. Package API HTML (when editing `docs/src/`):  
   `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`  
   then `julia --project=docs docs/make.jl` from the repository root.

If any of these fail before you touch application code, fix the environment
(Julia version versus `[compat]`, stale `Manifest.toml`, or missing Quarto
when rendering the website) rather than chasing false positives in the library.

## Coding rules worth highlighting

- Use ASCII identifiers in code (`u`, `v`, `w` for reference coordinates,
  not `xi`, `eta`, `zeta` or Greek letters). Greek may appear in
  comments and docstrings where it aids reading.
- Hot paths must remain zero-allocation. The regression for this lives
  in `test/assemblers/test_dof_based_zero_alloc.jl`. New code that
  touches the assembly path should not add `Dict`, `Vector{Any}`,
  untyped closures or growable buffers inside loops.
- Tests for `src/<topic>/<feature>.jl` go to
  `test/<topic>/test_<feature>.jl`.
- Prefer one file per commit; two paths is fine when they are one story
  (e.g. implementation + its test). Never `git add .` or `git add -A`
  unless you mean it. The full protocol is in
  `.github/prompts/commit.prompt.md`. With
  `git config core.hooksPath .githooks`, **pre-commit** allows at most **two**
  staged files per commit, and **commit-msg** enforces the subject / summary /
  bullet layout plus an **80**-character line cap (merge commits skip while
  `.git/MERGE_HEAD` exists).
- Match **commit message depth** to the patch: a short summary suffices for
  small edits; **large files, large diffs, or several concerns in one commit**
  should add **grouped bullets** (major API or behavior, wiring, migrations,
  caveats) so history stays readable without re-walking every hunk.

## SPDX file headers

SPDX tags belong at the top of new files when the rest of the tree uses them,
but they must always sit inside comment syntax that matches the file type.
Never paste raw `SPDX-*` lines into prose formats where they would render as
visible text or break the format.

- Julia (`.jl`), shell, Python, and other `#` line-comment languages:

  ```text
  # SPDX-FileCopyrightText: 2015-2026 Jukka Aho
  # SPDX-License-Identifier: MIT
  ```

- TOML, INI-style configs, and similar `#` comment formats: same `#` lines as above.

- Markdown, Quarto (`.md`, `.qmd`), and other HTML-friendly prose: wrap the tags
  in an HTML comment so renderers hide them:

  ```markdown
  <!--
  SPDX-FileCopyrightText: 2015-2026 Jukka Aho
  SPDX-License-Identifier: MIT
  -->
  ```

Formats that genuinely lack comments (for example strict JSON) cannot carry an
in-file SPDX block; keep licensing in repository-level files or omit an in-file
tag rather than emitting invalid syntax.

## Documentation pull requests

When you change **user-facing prose** (Quarto `juliafem.github.io/docs/`, examples
index, book landing pages) or **docstrings** that feed the site API page:

1. From `juliafem.github.io/`, run  
   `julia scripts/check_website_docs.jl`  
   (denylist + relative Markdown / `book/index.qmd` chapter targets).
2. If you edited **JuliaFEM docstrings** or `docs/api/` sources used by the site
   builder, run  
   `julia --project=. scripts/build_docs.jl api`  
   from `juliafem.github.io/` and commit the regenerated `api/` output if your
   project tracks it.
3. Add or adjust **`juliafem.github.io/docs/documentation-map.md`** when you
   introduce a new top-level guide readers should trust.
4. For package-only Documenter (`docs/make.jl`), use the **`docs/`** environment:  
   `julia --project=docs -e 'using Pkg; Pkg.instantiate()'` then  
   `julia --project=docs docs/make.jl` from the repository root (see `docs/README.md`).

## Documentation style

When writing READMEs, docstrings, design notes or session logs:

- No emoji.
- No markdown bold for emphasis. Plain prose carries enough weight.
- Prefer short, technical sentences and precise code references over
  marketing copy.

## Where things go

[`src/repository_layout.md`](src/repository_layout.md) is the
authoritative file organisation guide and contains a decision tree for
any new file. A short pointer lives at [`repository_layout.md`](repository_layout.md).
In particular, local session logs and scratch notes often live under a
gitignored `llm/` tree, not under `docs/` or `test/`.

## Website and Quarto CI

The **`juliafem.github.io/`** tree is a Quarto site checked by
**`.github/workflows/SiteDocs.yml`**. Visual tokens and navbar/logo rules live in
**`juliafem.github.io/docs/contributor-guide/design_system.md`**. After changing
site Markdown or book landing links, run from `juliafem.github.io`:

```bash
julia --project=. scripts/build_docs.jl api
julia scripts/check_website_docs.jl
```

A full local site build is `quarto render` from `juliafem.github.io/`. It fails
if two inputs share the same HTML stem (for example paired Literate `*.md` and
`*.qmd`, or Documenter `api/index.md` next to `api/index.qmd`); `build_docs.jl`
removes those duplicates after regenerating API and examples.

If you change `default_quadrature` or quadrature tables under `src/quadrature/`,
regenerate the user-guide table snippet from the **repository root** and commit
the updated file:

```bash
julia --project=. juliafem.github.io/scripts/generate_quadrature_defaults_snippet.jl
git diff juliafem.github.io/docs/user-guide/_quadrature_defaults_snippet.md
```

**`SiteDocs.yml`** runs that generator and fails the job when the snippet is out
of date (`git diff --exit-code` on the snippet path). The same workflow runs
**`scripts/check_curated_doc_vocabulary.jl`**, which fails if obsolete API
strings appear under **`juliafem.github.io/docs/user-guide/`** (for example
`register_fields!`).

`check_website_docs.jl` enforces a small denylist of obsolete example patterns,
scans relative `*.md` / `*.qmd` links under `docs/`, `examples/`, `showcase/`, and
`articles/`, and verifies that **chapter links in `book/index.qmd`** resolve to
real `.qmd` files.

## Getting help

- Open a GitHub issue with a minimal reproducible example for bugs.
- Open a GitHub discussion or issue for design questions before doing
  large pieces of work.
