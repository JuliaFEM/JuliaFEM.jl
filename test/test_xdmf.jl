# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module XDMFTests

using JuliaFEM
using JuliaFEM.Postprocess
using JuliaFEM.Test

testdata = """\
<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection" Name="Collection">
      <Grid Name="Grid">
      <Time Value="123"/>
        <Geometry Type="XYZ">
          <DataItem DataType="Float" Dimensions="60" Format="XML" Precision="8">
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.5 0.0 0.0
0.5 0.5 0.0
0.0 0.5 0.0
0.0 0.0 0.5
0.5 0.0 0.5
0.0 0.5 0.5
1.0 1.0 1.0
2.0 1.0 1.0
1.0 2.0 1.0
1.0 1.0 2.0
1.5 1.0 1.0
1.5 1.5 1.0
1.0 1.5 1.0
1.0 1.0 1.5
1.5 1.0 1.5
1.0 1.5 1.5
</DataItem>
        </Geometry>
        <Topology TopologyType="Mixed" NumberOfElements="2" >
          <DataItem Format="XML" DataType="Int" Dimensions="22">
38 0 1 2 3 4 5 6 7 8 9
38  10 11 12 13 14 15 16 17 18 19
</DataItem>
        </Topology>
        <Attribute Center="Node" Name="Displacement" Type="Vector">
        <DataItem DataType="Float" Dimensions="60" Format="XML" Precision="8">
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.5 0.0 0.0
0.5 0.5 0.0
0.0 0.5 0.0
0.0 0.0 0.5
0.5 0.0 0.5
0.0 0.5 0.5
1.0 1.0 1.0
2.0 1.0 1.0
1.0 2.0 1.0
1.0 1.0 2.0
1.5 1.0 1.0
1.5 1.5 1.0
1.0 1.5 1.0
1.0 1.0 1.5
1.5 1.0 1.5
1.0 1.5 1.5
</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

function test_write_to_xml()
    nodes = Vector{Float64}[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        [1.5, 1.0, 1.0],
        [1.5, 1.5, 1.0],
        [1.0, 1.5, 1.0],
        [1.0, 1.0, 1.5],
        [1.5, 1.0, 1.5],
        [1.0, 1.5, 1.5]]

    elements = [
        (:Tet10, [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
        (:Tet10, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]
    
    displacement_field = nodes # same structure
    xdoc, model = JuliaFEM.Postprocess.xdmf_new_model()
    temporal_collection = JuliaFEM.Postprocess.xdmf_new_temporal_collection(model)
    grid = JuliaFEM.Postprocess.xdmf_new_grid(temporal_collection; time=1)
    JuliaFEM.Postprocess.xdmf_new_mesh!(grid, nodes, elements)
    JuliaFEM.Postprocess.xdmf_new_nodal_field!(grid, "Displacement", displacement_field)
    JuliaFEM.Postprocess.xdmf_save_model(xdoc, "/tmp/foo.xmf")
    #info("exported data model: \n$(string(xdoc))")
    #@test string(xdoc) == testdata
    d1 = split(string(xdoc), "\n")
#   d2 = split(testdata, "\n")
    d2 = open(readlines, Pkg.dir("JuliaFEM")*"/test/testdata/quad_two_tet10.xmf")
    println("comparing string")
    for i in 1:length(d1)
        println("d1: $(d1[i])")
        println("d2: $(d2[i])")
        #status = d1 == d2 ? "MATCHES" : "NO MATCH"
        #info("line: $(d1[i]) $status")
        #if d1 != d2
        #    info("should be:\n$(d2[i])")
        #end
        d1 == d2 || error("No match")
    end
end

end
