# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Abaqus
using JuliaFEM.Testing

@testset "parse abaqus inp file to AbaqusModel" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/cube_tet4.inp"
    model = abaqus_read_model(fn)

    @test length(model.properties) == 1
    section = first(model.properties)
    @test section.element_set == :CUBE
    @test section.material_name == :MAT

    @test haskey(model.materials, :MAT)
    material = model.materials[:MAT]
    @test isapprox(first(material.properties).E, 208.0e3)

    @test length(model.steps) == 1
    step = first(model.steps)
    @test length(step.boundary_conditions) == 2

    bc = step.boundary_conditions[1]
    @test bc.data[1] == [:SYM12, 3]
    @test bc.data[2] == [:SYM23, 1]
    @test bc.data[3] == [:SYM13, 2]

    load = step.boundary_conditions[2]
    @test load.data[1] == [:LOAD, :P, 1.00000]
end

#=
@testset "given abaqus model solve field" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/cube_tet4.inp"
    model = abaqus_read_model(fn)
    problems = model()
    body = first(problems)
    info(body("displacement", 0.0))
    result = XDMF()
    xdmf_new_result!(result, body, 0.0)
    xdmf_save_field!(result, body, 0.0, "displacement"; field_type="Vector")
    xdmf_save!(result, "/tmp/cube_tet4.xmf")
end
=#
