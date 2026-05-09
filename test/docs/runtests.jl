# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test

const SNIPPET = joinpath(@__DIR__, "..", "..", "docs", "src", "snippets", "minimal_elasticity_quickstart.jl")

@testset "Documentation snippets" begin
    @test isfile(SNIPPET)
    include(SNIPPET)
    r = minimal_elasticity_quickstart()
    @test r.ndofs == 375
    @test r.nnz_stiffness == 19773
end

include("readme_example.jl")
