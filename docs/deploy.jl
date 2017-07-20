# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

using Documenter
using JuliaFEM

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/JuliaFEM/JuliaFEM.jl.git",
    julia = "0.6")
