# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
    write_vtu_mesh(basepath::AbstractString, mesh::Mesh; point_data = (;), cell_data = (;)) -> String

Write a single-block VTU (VTK XML unstructured grid) for `mesh`.

`basepath` must not include a file extension; WriteVTK appends `.vtu`.

# Point and cell fields

- `point_data`: `NamedTuple` of nodal fields. Each value is either a length-`nnodes_total(mesh)`
  vector (scalar) or a `3 × nnodes_total` matrix (3-vector per node, e.g. displacement).
- `cell_data`: `NamedTuple` of per-element fields; each value is a length-`nelements(mesh)` vector.

# Weak dependency

This entry point is implemented in `JuliaFEMWriteVTKExt` when
[`WriteVTK.jl`](https://github.com/JuliaVTK/WriteVTK.jl) is loaded after JuliaFEM:

```julia
using JuliaFEM, WriteVTK
write_vtu_mesh(joinpath(outdir, "solution"), mesh; point_data = (; u = uvec))
```

Supported topologies: linear `Seg2`, `Tri3`, `Quad4`, `Tet4`, and `Hex8` only.

See also: [`read_gmsh_msh`](@ref) for the Gmsh reader extension.
"""
function write_vtu_mesh(args...; kwargs...)
    throw(
        ErrorException(
            "write_vtu_mesh requires the WriteVTK.jl package. " *
            "Add it to your environment (`import Pkg; Pkg.add(\"WriteVTK\")`) " *
            "and run `using WriteVTK` after `using JuliaFEM` so the " *
            "`JuliaFEMWriteVTKExt` extension can load.",
        ),
    )
end
