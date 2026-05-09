# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "interface" begin
    include("test_interface_mesh.jl")
    include("test_volume_interface_coupling.jl")
end
