# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    JuliaFEM.IO

Input/Output submodule for reading mesh files and writing results.

# Available (zero external dependencies)
- `abaqus_read_mesh(filename)`: Read ABAQUS .inp files
- `create_surface_elements(mesh, name)`: Create surface elements
- `create_nodal_elements(mesh, name)`: Create nodal elements

# Future (requires optional dependencies)
- Code Aster .med files: `aster_read_mesh` (requires HDF5.jl)
- Xdmf output format: `Xdmf` writer (requires HDF5.jl + LightXML.jl)
- VTK output: Native Julia implementation (no dependencies)

# Usage
```julia
using JuliaFEM
mesh = abaqus_read_mesh("model.inp")
body = create_elements(mesh, "BODY")
```
"""
module IO

using ..JuliaFEM: Element, Mesh, add_node!, add_element_to_element_set!,
    add_node_to_node_set!

# Re-export for convenience
import ..JuliaFEM.Preprocess
using SparseArrays, LinearAlgebra
using Logging

include("abaqus_reader.jl")
export abaqus_read_mesh, create_surface_elements, create_nodal_elements

# AsterReader would go here when HDF5 is added as optional dependency
# include("aster_reader.jl")
# export aster_read_mesh

end # module IO
