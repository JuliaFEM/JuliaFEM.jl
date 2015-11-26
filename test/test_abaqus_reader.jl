# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module AbaqusReaderTests

using JuliaFEM
using JuliaFEM.Test

using JuliaFEM: parse_abaqus, parse_element_section

function test_read_abaqus_model()
  # FIXME: get_test_data()
    model = open(parse_abaqus, Pkg.dir("JuliaFEM")*"/geometry/3d_beam/palkki.inp")
    @test length(model["nodes"]) == 298
    @test length(model["elements"]) == 120
    @test length(model["elsets"]["Body1"]) == 120
    @test length(model["nsets"]["SUPPORT"]) == 9
    @test length(model["nsets"]["LOAD"]) == 9
    @test length(model["nsets"]["TOP"]) == 83
end

#=
facts("test that reader throws error when dimension information of elemenet is missing") do
    #   *ELEMENT, TYPE=neverseenbefore, ELSET=Body1
    data = """
    1,       243,       240,       191,       117,       245,       242,       244,
    1,         2,       196
    """
    model = Dict()
    header = Dict("section"=>"ELEMENT", "options" => Dict("TYPE" => "neverseenbefore", "ELSET"=>"Body1"))
    @fact_throws parse_element_section(model, header, data)
end
=#

function test_read_element_section()
    data = """
    1,       243,       240,       191,       117,       245,       242,       244,
    1,         2,       196
    2,       204,       199,       175,       130,       207,       208,       209,
    3,         4,       176
    """
    model = Dict()
    header = Dict(
        "section" => "ELEMENT",
        "options" => Dict("TYPE" => "C3D10", "ELSET" => "BEAM"))
    parse_element_section(model, header, data)
    @test length(model["elements"]) == 2
    @test model["elements"][1] == [243, 240, 191, 117, 245, 242, 244, 1, 2, 196]
    @test model["elements"][2] == [204, 199, 175, 130, 207, 208, 209, 3, 4, 176]
    @test model["elsets"]["BEAM"] == [1, 2]
end

function test_read_surface_set_section()
    data = """
    31429,S1
    31481,S3
    """
    model = Dict()
    header = Dict(
        "section" => "SURFACE",
        "options" => Dict("TYPE" => "ELEMENT", "NAME" => "LOAD"))
    parse_surface_section(model, header, data)
    @test model["surfaces"]["LOAD"] = [(31429,1), (31481,3)]

end

function test_unknown_handler_warning_message()
    fn = tempname()
    fid = open(fn, "w")
    testdata = """
    *ELEMENT2, TYPE=C3D10, ELSET=Body1
    1,       243,       240,       191,       117,       245,       242,       244,
    1,         2,       196
    """
    write(fid, testdata)
    close(fid)
    model = open(parse_abaqus, fn)
    # empty model expected, parser doesn't know what to do with unknown section
    @test length(model) == 0
end

end
