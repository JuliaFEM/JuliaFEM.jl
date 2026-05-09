# [Choosing an assembly strategy](@id assembler_choice)

JuliaFEM **0.x** exposes more than one assembly path. This page matches the
split under `src/assemblers/` and the regression tests.

## DOF-based COO (`DOFBasedCOOAssembler`)

- **Use when:** you want the modern, matrix-free-friendly pipeline: one cache
  drives both sparse triplets and `apply_K!` / Krylov operators.
- **Strengths:** walks global DOF rows with `local_dof_layout` decoding; the
  hot path is written for zero allocations after warmup
  (`test/assemblers/test_dof_based_zero_alloc.jl`).
- **Tests:** `test/assemblers/test_dof_based_apply_K.jl` and related
  `test_dof_based_*.jl` files.

## Element-based COO / CSC

- **Use when:** you want the classical element-by-element scatter into COO or
  CSC as a reference implementation or for tooling that expects element loops.
- **Strengths:** straightforward comparison to textbook element assembly.
- **Code:** `src/assemblers/element_based/`; see `src/assemblers/README.md`.

## Matrix-free operators

- **Use when:** you do not want to form `K` explicitly (large systems,
  matrix-free Newton–Krylov).
- **Strengths:** composable Dirichlet, MPC, loads, and preconditioners under
  `src/assemblers/matrix_free/`.
- **Tests:** `test/assemblers/test_matrix_free_operator.jl` and neighbors.
- **Site guide:** `juliafem.github.io/docs/user-guide/matrix_free_cookbook.md`.

## Rule of thumb

Start with **DOF-based COO** for new physics and examples unless you have a
specific reason to call the element-based scatter directly.
