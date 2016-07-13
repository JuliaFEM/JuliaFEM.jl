# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Abaqus
using JuliaFEM.Testing

# to turn on automatic file download, set
# ENV["ABAQUS_DOWNLOAD_URL"] = http://<domain>:2080/v2016/books/eif
# if don't want to download all stuff to current directory,
# set also e.g. ENV["ABAQUS_DOWNLOAD_DIR"] = "/tmp"

#=
test_name = "ecs4sfs1"
@testset "$test_name" begin
    abaqus_run_test(test_name) || return
    results = abaqus_read_results(test_name)
end
=#

@testset "ec38sfs2" begin
    return_code = abaqus_run_model("ec38sfs2"; fetch=true, verbose=true)
    return_code == 0 || return
    @test return_code == 0
#=
    xdmf = abaqus_open_results("ec38sfs2")
    side, opts = read_result(xdmf, "SECTION/side")
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
