# JuliaFEM.jl news

User-visible and contributor-facing changes. The package is **0.x**; treat this
file as a lightweight changelog until a **1.0** release with full SemVer discipline.

## 0.5.2 (current `Project.toml`)

- Type-stable **`Element{K,P,S,N}`** pipeline with **`@DOFSet`**, **`DOFHandler`**,
  and domain **`AbstractKernel`** implementations under `src/domains/`.
- **`DOFBasedCOOAssembler`** and matrix-free utilities under `src/assemblers/`
  (including preconditioners and MPC hooks where merged on your branch).
- Optional **`JuliaFEM.Legacy`** behind **`JULIAFEM_ENABLE_LEGACY=1`** for older
  `Problem` / Dict-field workflows.

For earlier history, use `git log` and archived notebooks under `docs/tutorials/`
(reference only).

When you land a user-visible change, add a dated bullet under a new **Unreleased**
heading in the same PR that updates code or docs.
