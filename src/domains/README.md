<!--
SPDX-FileCopyrightText: 2015-2026 Jukka Aho
SPDX-License-Identifier: MIT
-->

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
| `poroelastic/` | `BiotPoroelasticKernel` (multi-field `u` + pore pressure `p`) | `test/domains/poroelastic/` |
| `thermo_poroelastic/` | `ThermoPoroelasticKernel` (`u` + `T` + `p`; optional `kappa_tp`, `zeta_tp`, `heat_capacity`, `density` for `M_uu`). Module notes: `thermo_poroelastic/README.md`. | `test/domains/thermo_poroelastic/` (incl. column thermal decay vs Fourier) |

Narrative walkthrough for thermo-elasticity: `docs/src/thermo_elastic_walkthrough.md`
in the package Documenter tree.
