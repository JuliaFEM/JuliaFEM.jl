# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE
#
# Mesh readers consolidated from AbaqusReader.jl and AsterReader.jl

# AbaqusReader - ABAQUS .inp file format
include("readers/keyword_register.jl")
include("readers/parse_mesh.jl")
include("readers/parse_model.jl")
include("readers/create_surface_elements.jl")
include("readers/abaqus_download.jl")

# AsterReader - Code Aster .med file format  
include("readers/read_aster_mesh.jl")
include("readers/read_aster_results.jl")
