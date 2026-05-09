using Test
using JuliaFEM

@testset "Validation" begin
    include("test_cantilever_regression.jl")
    include("test_cantilever_materials_showcase.jl")
    include("test_plasticity_simple.jl")
end
