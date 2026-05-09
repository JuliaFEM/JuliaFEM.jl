# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Triangle reference_coordinates" begin
    for T in (Triangle{3}, Triangle{6}, Triangle{7}, Triangle{10})
        rc = reference_coordinates(T())
        @test length(rc) == nnodes(T())
        @test rc[1] isa Vec{2,Float64}
    end
end
