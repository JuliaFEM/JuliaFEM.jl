# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Documenter

deploydocs(
    repo = "github.com/JuliaFEM/JuliaFEM.jl.git",
    julia = "0.6",
    target = "build",
    deps = nothing,
    make = nothing)
