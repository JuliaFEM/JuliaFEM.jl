# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
    read_gmsh_msh(path::AbstractString; kwargs...) -> Mesh

Read a Gmsh `.msh` file into a concrete [`Mesh`](@ref).

Supported volume/surface Gmsh element types (linear only): **3-node triangle** (2),
**4-node quadrilateral** (3), **4-node tetrahedron** (4), **8-node hexahedron** (5).
Mixed cell types in the same file are not supported.

Keyword `dim` may be `2` or `3` to select surface or volume elements; the default
chooses dimension **3** if the model has 3D cells, otherwise **2**.

Physical groups on the mesh dimension populate [`element_sets`](@ref) using
sanitized physical names (fallback `physical_{dim}_{tag}`). Physical groups one
dimension lower (boundary curves in 2D, boundary faces in 3D) populate
[`node_sets`](@ref).

Implemented in `JuliaFEMGmshExt` when [`Gmsh.jl`](https://github.com/JuliaFEM/Gmsh.jl)
is loaded after `using JuliaFEM` (`import Pkg; Pkg.add("Gmsh")` then `using Gmsh`).
Without that package, calling `read_gmsh_msh` with a path string raises an
`ErrorException` with install instructions (other arities always raise here).
"""
function read_gmsh_msh(args...; kwargs...)
    throw(
        ErrorException(
            "read_gmsh_msh requires the Gmsh.jl package. " *
            "Add it to your environment (`import Pkg; Pkg.add(\"Gmsh\")`) " *
            "and run `using Gmsh` after `using JuliaFEM` so the " *
            "`JuliaFEMGmshExt` extension can load.",
        ),
    )
end

"""
    mesh_from_current_gmsh_model(; kwargs...) -> Mesh

Build a [`Mesh`](@ref) from the **currently active** Gmsh model (after
`Gmsh.initialize`, geometry, and `gmsh.model.mesh.generate`).

The same element-type rules and keyword `dim` apply as for [`read_gmsh_msh`](@ref).

Implemented in `JuliaFEMGmshExt` when `Gmsh` is loaded. Without it, you get a
`MethodError` (no method defined).
"""
function mesh_from_current_gmsh_model end
