# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Abaqus
using JuliaFEM.Testing

@testset "read inp file" begin
    model = open(parse_abaqus, Pkg.dir("JuliaFEM")*"/geometry/3d_beam/palkki.inp")
    @test length(model["nodes"]) == 298
    @test length(model["elements"]) == 120
    @test length(model["elsets"]["Body1"]) == 120
    @test length(model["nsets"]["SUPPORT"]) == 9
    @test length(model["nsets"]["LOAD"]) == 9
    @test length(model["nsets"]["TOP"]) == 83
end

@testset "test read element section" begin
    data = """*ELEMENT, TYPE=C3D10, ELSET=BEAM
    1,       243,       240,       191,       117,       245,       242,       244,
    1,         2,       196
    2,       204,       199,       175,       130,       207,       208,       209,
    3,         4,       176
    """
    data = split(data, "\n")
    model = Dict{AbstractString, Any}()
    model["nsets"] = Dict{AbstractString, Vector{Int}}()
    model["elsets"] = Dict{AbstractString, Vector{Int}}()
    model["elements"] = Dict{Integer, Any}()
    parse_section(model, data, :ELEMENT, 1, 5, Val{:ELEMENT})
    @test length(model["elements"]) == 2
    @test model["elements"][1]["connectivity"] == [243, 240, 191, 117, 245, 242, 244, 1, 2, 196]
    @test model["elements"][2]["connectivity"] == [204, 199, 175, 130, 207, 208, 209, 3, 4, 176]
    @test model["elsets"]["BEAM"] == [1, 2]
end

@testset "parse nodes from abaqus .inp file to Mesh" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/cube_tet4.inp"
    mesh = abaqus_read_mesh(fn)
    info(mesh.surfaces)
    @test length(mesh.nodes) == 10
    @test length(mesh.elements) == 17
    @test haskey(mesh.elements, 1)
    @test mesh.elements[1] == [8, 10, 1, 2]
    @test mesh.element_types[1] == :Tet4
    @test haskey(mesh.node_sets, :SYM12)
    @test haskey(mesh.element_sets, :CUBE)
    @test haskey(mesh.surfaces, :LOAD)
    @test haskey(mesh.surfaces, :ORDER)
    @test length(mesh.surfaces[:LOAD]) == 2
    @test mesh.surfaces[:LOAD][1] == (16, :S1)
    @test mesh.surface_types[:LOAD] == :ELEMENT
    @test length(Set(map(size, values(mesh.nodes)))) == 1
    elements = create_surface_elements(mesh,:LOAD)
    @test get_connectivity(elements[1]) == [8,9,10]
end
