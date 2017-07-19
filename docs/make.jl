# This file is a part of project JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/NodeNumbering.jl/blob/master/LICENSE

using Documenter
using JuliaFEM

makedocs(
    modules = [JuliaFEM],
    checkdocs = :all,
    strict = false)
