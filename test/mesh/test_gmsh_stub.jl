# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Gmsh bridge stubs (extension not loaded)" begin
    @test_throws ErrorException JuliaFEM.read_gmsh_msh("nonexistent.msh")
    @test_throws MethodError JuliaFEM.mesh_from_current_gmsh_model()
end
