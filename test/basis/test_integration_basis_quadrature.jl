# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Test
using JuliaFEM

@testset "integration_points(topology, basis) tracks basis order" begin
    ips_l1 = integration_points(Tet4(), Lagrange{1}())
    ips_l2 = integration_points(Tet4(), Lagrange{2}())
    @test length(ips_l2) > length(ips_l1)
    @test basis_quadrature_order(Lagrange{2}()) == 2
end

@testset "topology-only integration matches Lagrange{1} basis on Tet4" begin
    ips_topo = integration_points(Tet4())
    ips_b = integration_points(Tet4(), Lagrange{1}())
    @test length(ips_topo) == length(ips_b)
end
