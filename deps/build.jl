# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

Pkg.clone("https://github.com/JuliaFEM/AbaqusReader.jl.git")
Pkg.build("AbaqusReader")
Pkg.clone("https://github.com/JuliaFEM/AsterReader.jl.git")
Pkg.build("AsterReader")
Pkg.clone("https://github.com/JuliaFEM/FEMQuad.jl.git")
Pkg.build("FEMQuad")
