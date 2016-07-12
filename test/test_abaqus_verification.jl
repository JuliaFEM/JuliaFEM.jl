# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Abaqus
using JuliaFEM.Testing

#=
test_name = "ecs4sfs1"
@testset "$test_name" begin
    abaqus_run_test(test_name) || return
    results = abaqus_read_results(test_name)
end
=#

@testset "ec38sfs2" begin
    abaqus_run_test("ec38sfs2"; print_test_file=true) || return
    #=
    results = abaqus_read_results("ec38sfs2")
    side = get_results(results, "SECTION"; name="side")
    @test isapprox(side["SOFM"], 3464.0)
    @test isapprox(side["SOF1"], 2000.0)
    @test isapprox(side["SOF2"], 2000.0)
    @test isapprox(side["SOF3"], 2000.0)
    @test isapprox(side["SOMM"], 2828.0)
    @test isapprox(side["SOM1"], 0.0)
    @test isapprox(side["SOM2"], 2000.0)
    @test isapprox(side["SOM3"], -2000.0)
    @test isapprox(side["SOAREA"], 2.000)
    @test isapprox(side["SOCF1"], 2/3)
    @test isapprox(side["SOCF2"], 2/3)
    @test isapprox(side["SOCF3"], 1/6)
    =#
end
