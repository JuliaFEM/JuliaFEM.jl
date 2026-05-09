# src/assemblers/

Assembly machinery for the current type-stable pipeline (0.x).

See `AGENTS.md` for invariants and `docs/src/assembler_choice.md` for a short
comparison of DOF-based, element-based, and matrix-free paths. Regression tests
live under `test/assemblers/`, especially `test_dof_based_zero_alloc.jl` and
`test_matrix_free_operator.jl`.

The directory is split along the real fault lines. Each subdirectory
owns a specific assembly strategy plus the supporting types it
specialises:

| Path                     | Responsibility                                                                                                       |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| `abstract.jl`            | `AbstractAssembler` hierarchy plus the kernel defaults (`get_field`, `dofs_per_node`, `get_dof_mapping!`).           |
| `microkernel.jl`         | Microkernel trait (`evaluate_entry`, `qpoint_buffer_eltype`, ...) used by the DOF-based and matrix-free path.        |
| `caches/`                | Reusable per-element scratch (`GeometryCache`, `ElementCache`, `AssemblyMaterialWorkspace`) and the COO global cache.|
| `element_based/`         | Classic element-by-element assembler (COO) plus the scatter routines it dispatches to.                               |
| `dof_based/`             | DOF-by-DOF assembler (`DOFBasedCOOAssembler` / `DOFBasedCOOCache` and the backend-agnostic KernelAbstractions port). |
| `matrix_free/`           | Declarative Dirichlet / MPC constraints, Neumann loads, matrix-free preconditioners and the generalized eigensolver. |

The matrix-free directory builds on the DOF-based cache plus
`apply_K!` / `apply_M!`. All matrix-free constraint and load types
share the `apply_constraint_*` hook protocol so that Dirichlet, MPC
and load contributions compose freely with both the assembled and the
matrix-free solves.
