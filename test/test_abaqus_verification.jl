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
    results = abaqus_read_results("ec38sfs2")
end
