# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "fields/api.jl mapping and traits" begin
    nodes = [10, 20, 30, 40]
    @testset "Displacement{3}" begin
        buf = zeros(Int, 12)
        get_dof_mapping!(buf, Displacement{3}(), nodes)
        @test buf[1:3] == [28, 29, 30]
        @test buf[4:6] == [58, 59, 60]
    end
    @testset "Temperature & PressurePotential" begin
        buf = zeros(Int, 4)
        get_dof_mapping!(buf, Temperature(), nodes)
        @test buf == nodes
        get_dof_mapping!(buf, PressurePotential(), nodes)
        @test buf == nodes
    end
    @testset "facet / edge field tags" begin
        @test dofs_per_node(RT0FaceFlux()) == 1
        @test dofs_per_node(Nedelec1Edge()) == 1
        @test quantity_type(RT0FaceFlux) == Float64
        @test quantity_type(Nedelec1Edge) == Float64
    end
    @testset "DisplacementRotation" begin
        @test dofs_per_node(DisplacementRotation{3}()) == 6
        @test quantity_type(DisplacementRotation{2}) == Vec{4}
    end
end
