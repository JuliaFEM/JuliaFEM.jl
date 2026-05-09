# src/domains/

Physics kernels that plug into the shared assembler infrastructure. Each
subdirectory implements an `AbstractKernel` for one discipline or coupled
problem.

For architecture, invariants, and vocabulary, read `AGENTS.md` in the
repository root. For the microkernel checklist when adding a kernel, read
`juliafem.github.io/docs/developer-guide/kernel_extension_contract.md` in the
same checkout.

## Layout

| Directory | Kernel(s) | Tests |
|-----------|-----------|-------|
| `continuum/` | `ContinuumKernel`, small-strain mechanics helpers | `test/domains/continuum/` |
| `heat/` | `HeatKernel` | `test/domains/heat/` |
| `darcy/` | `DarcyPotentialKernel` (primal potential); `DarcyMixedRT0P0Kernel` (Tet4 RT₀–P₀ H(div)) | `test/domains/darcy/` |
| `thermo_elastic/` | `ThermoElasticKernel` (multi-field `u` + `T`) | `test/domains/thermo_elastic/` |

Narrative walkthrough for thermo-elasticity: `docs/src/thermo_elastic_walkthrough.md`
in the package Documenter tree.
