# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM
using StaticArrays: SVector
using Tensors: Vec

@testset "Quadrature Module" begin
    include("test_api.jl")
    include("test_triangles.jl")
    include("test_tetrahedra.jl")
    include("test_tensor_products.jl")
    include("test_wedges.jl")
    include("test_pyramids.jl")
    include("test_accuracy.jl")
    include("test_legacy_api.jl")
end
