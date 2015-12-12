# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module AsterReaderTests

using JuliaFEM
using JuliaFEM.Test

#using JuliaFEM: parse
using JuliaFEM.Preprocess: aster_parse_nodes

function test_read_mesh()
mesh = """
    COOR_2D
    N1          0.0 0.0
    N2          1.0 0.0
    N3          1.0 1.0
    N4          0.0 1.0
    FINSF

    QUAD4
    E1         N1 N2 N3 N4
    FINSF

    SEG2
    E2         N3 N4
    FINSF

    GROUP_NO NOM=NALL
    N1 N2
    FINSF

    GROUP_MA NOM=BODY1
    E1 E2
    FINSF

    FIN
    """

    m = parse(mesh, Val{:CODE_ASTER_MAIL})
    @test m["nodes"]["N1"] == [0.0, 0.0]
    @test m["elements"]["E1"] == ["QUAD4", ["N1", "N2", "N3", "N4"]]
    @test m["elements"]["E2"] == ["SEG2", ["N3", "N4"]]
    @test m["elsets"]["BODY1"] == ["E1", "E2"]
    @test m["nsets"]["NALL"] == ["N1", "N2"]

end
#test_read_mesh()

function test_parse_nodes()
    section = """
    N9  2.0 3.0 4.0
    COOR_3D
    N1          0.0 0.0 0.0
    N2          1.0 0.0 0.0
    N3          1.0 1.0 0.0
    N4          0.0 1.0 0.0
    N5          0.0 0.0 1.0
    N6          1.0 0.0 1.0
    N7          1.0 1.0 1.0
    N8          0.0 1.0 1.0
    FINSF
    absdflasdf
    N12 3.0 4.0 5.0 6.0
    N13 3.0 4.0 5.0
    """
    nodes = aster_parse_nodes(section)
    @test nodes[1] == Float64[0.0, 0.0, 0.0]
    @test nodes[8] == Float64[0.0, 1.0, 1.0]
    @test length(nodes) == 8
end
test_parse_nodes()

end
