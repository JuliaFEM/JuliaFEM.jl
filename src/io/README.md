<!--
SPDX-FileCopyrightText: 2015-2026 Jukka Aho
SPDX-License-Identifier: MIT
-->

# src/io/

Policy for **external mesh formats** and **visualisation export** (Gmsh, VTK, …).

The core package owns the type-stable mesh model (`Mesh{N, T}` under
`src/mesh/`) and assembly; it does **not** ship format-specific parsers or writers
in unconditional `src/io/*.jl` runtime paths. Converters live in **package
extensions** wired through `[weakdeps]` / `[extensions]` in `Project.toml`, mirroring
the MPI pattern (`JuliaFEMMPIExt`).

Legacy `.inp` / `.med` readers tied to the pre-reset API remain under
`src/legacy/io/` and load only with `JULIAFEM_ENABLE_LEGACY=1`.

## VTK (WriteVTK)

- Weak dependency: `WriteVTK` → extension `JuliaFEMWriteVTKExt`.
- User workflow: `using JuliaFEM, WriteVTK` then [`write_vtu_mesh`](@ref) to write
  `.vtu` unstructured grids with optional nodal (`point_data`) and per-element
  (`cell_data`) fields.
- Supported element shapes for export: linear `Seg2`, `Tri3`, `Quad4`, `Tet4`,
  `Hex8` (same single-topology assumption as the rest of the mesh stack).

## Gmsh

- See [`read_gmsh_msh`](@ref) (`JuliaFEMGmshExt`).

## This directory

`write_vtu.jl` defines the [`write_vtu_mesh`](@ref) stub and docstring; the
implementation is in `ext/JuliaFEMWriteVTKExt.jl`. Do not add heavy or
format-specific I/O to unconditional `JuliaFEM.jl` includes beyond thin stubs
like this.
