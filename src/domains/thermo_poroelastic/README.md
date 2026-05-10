<!--
SPDX-FileCopyrightText: 2015-2026 Jukka Aho
SPDX-License-Identifier: MIT
-->

# `src/domains/thermo_poroelastic/`

Linear quasi-steady **thermo-poroelasticity** on one mesh: vertex displacement
`u`, vertex temperature `T`, and vertex pore pressure `p`. The kernel is the
superposition of thermo-elastic `(u, T)` and Biot `(u, p)` structure with
optional cross-coupling between `T` and `p` (`kappa_tp`, `zeta_tp`). See
`kernel.jl` for the weak form, constructor variants, and mass entries
(`heat_capacity` on `T`, `storage_S` on `p`, Biot-style `M_pu` / `M_up` from
`α`, thermo-elastic `M_Tu` / `M_uT` from `β`, optional `M_uu` from solid
`density`).

Monolithic transient stepping (backward Euler on `(K + M/Δt)`) can use the same
`DOFBasedCOOCache` with `assemble!` + `assemble_M!`.

Tests: `test/domains/thermo_poroelastic/` (assembler parity, KA paths,
`PerElementKernelColumn`, and a column thermal diffusion check vs a 1-D Fourier
mode in `test_thm_column_thermal_decay.jl`). Poroelastic-only benchmarks live
under `test/domains/poroelastic/`.
