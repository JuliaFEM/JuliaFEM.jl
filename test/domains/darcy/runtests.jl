# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "Darcy domain (primal potential)" begin
    include("test_primal_darcy_potential.jl")
end

@testset "Darcy domain (mixed RT₀–P₀)" begin
    include("test_mixed_rt0_darcy.jl")
end
