# src/io/

Policy for **external mesh formats** (Gmsh, Netgen, Abaqus, Code Aster, …).

The core package owns the type-stable mesh model (`Mesh{N, T}` under
`src/mesh/`) and assembly; it does **not** ship format-specific parsers in
`src/io/`. Converters that read third-party files and build `Mesh{…}`,
`element_sets`, and `node_sets` should live in **separate packages** or in
**JuliaFEM package extensions** wired through `[weakdeps]` / `[extensions]` in
`Project.toml`, mirroring the MPI extension pattern (`JuliaFEMMPIExt`).

Legacy `.inp` / `.med` readers tied to the pre-reset API remain under
`src/legacy/io/` and load only with `JULIAFEM_ENABLE_LEGACY=1`.

## This directory

This folder holds **documentation only** until a new extension or companion
package is added. Do not grow unconditional `include("io/*.jl")` paths in
`JuliaFEM.jl` for heavy or format-specific I/O.
