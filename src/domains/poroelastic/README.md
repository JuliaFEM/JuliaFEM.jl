# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

# `src/domains/poroelastic/`

Steady quasi-static **Biot** coupling: solid displacement `u` with vertex
pore pressure `p`, Darcy flow in the pressure block, Biot coefficient `α`, and
optional fluid **storage** `storage_S` on pressure (L² mass for `assemble_M!`).
See `kernel.jl` for the weak form and `PorePressure` in `src/fields/api.jl`
for the scalar field tag.

Monolithic time stepping (e.g. backward Euler on `(K + M/dt)`) can use the
same `DOFBasedCOOCache` with `assemble!` + `assemble_M!`. The mass operator
includes `M_pp` (`storage_S`), `M_pu` and `M_up` (`α` times the divergence
pairings aligned with the steady `K_pu` / `K_up` blocks), and optional
consistent `M_uu` from solid density `ρ` (same as `ContinuumKernel` mass). Solid
acceleration dynamics beyond this `M_uu` slot are not part of
this kernel.

Tests: `test/domains/poroelastic/` (Terzaghi-style single-drainage column with
uniform IC vs Fourier when `α = 0` in `test_biot_terzaghi_uniform_ic.jl`, and
coupled `α > 0` refinement checks in `test_biot_terzaghi_coupled_refinement.jl`).
The thermo-poroelastic analogue for uncoupled **thermal** diffusion on a column
is `test/domains/thermo_poroelastic/test_thm_column_thermal_decay.jl`.
