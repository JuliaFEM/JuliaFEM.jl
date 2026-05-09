# src/io/

Mesh readers that ship with the current 0.x package surface.

The Abaqus `.inp` and Code Aster `.med` readers were built around the
older `Element(Poi1, ...)` constructors and the Dict-based field
system; they have been moved into the optional `JuliaFEM.Legacy`
submodule (see `src/legacy/io/`). Set the environment variable
`JULIAFEM_ENABLE_LEGACY=1` before `using JuliaFEM` to load them.

## Files

- `gmsh_reader.jl`
  Self-contained Gmsh `.msh` (ASCII format 4.1) reader.
  Defines its own `JuliaFEM.GmshReader` submodule with `GmshMesh`,
  `read_gmsh_mesh`. Has no dependency on legacy types and is loaded
  unconditionally.

## Future work

A current-API replacement for the Abaqus reader (returning
`Mesh{T<:AbstractTopology, N}` and elements built via `create_elements!`)
is on the roadmap. When it lands it will live here next to
`gmsh_reader.jl`.
