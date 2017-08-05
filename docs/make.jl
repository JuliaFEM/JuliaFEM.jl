# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Documenter, JuliaFEM

if haskey(ENV, "TRAVIS")
    println("inside TRAVIS, installing PyPlot + matplotlib")
    Pkg.add("PyPlot")
    run(`pip install matplotlib`)
end

makedocs(modules=[JuliaFEM],
         format = :html,
         sitename = "JuliaFEM",
         pages = [
                  "Introduction" => "index.md",
                  "API" => "api.md"
                 ])
