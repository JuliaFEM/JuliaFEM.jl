# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

using Lint
using Base.Test

results = lintpkg("JuliaFEM")
if !isempty(results)
    info("Lint.jl is a tool that uses static analysis to assist in the development process by detecting common bugs and potential issues.")
    info("For this package, Lint.jl report is following:")
    display(results)
    info("For more information, see https://lintjl.readthedocs.io/en/stable/")
    warn("Package syntax test has failed.")
    @test isempty(results)
else
    info("Lint.jl: syntax check pass.")
end
