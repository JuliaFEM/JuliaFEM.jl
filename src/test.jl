# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

if VERSION >= v"0.5-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

export @test, @testset, @test_throws
