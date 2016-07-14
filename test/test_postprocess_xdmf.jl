# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Postprocess
using JuliaFEM.Testing

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

@testset "write simple xmf file" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    u = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [0.5, 1.0],
        4 => [0.0, 0.0])
    n = Dict{Int64, Vector{Float64}}(
        2 => [1.0, 0.0],
        3 => [1.0, 0.0])
    el1 = Element(Quad4, [1, 2, 3, 4])
    el2 = Element(Seg2, [2, 3])
    update!([el1, el2], "geometry", X)
    update!([el1, el2], "displacement", u)
    update!(el2, "normal", n)
    xdmf = XDMF()
    xdmf.dimension = 2
    xdmf_new_result!(xdmf, [el1, el2], 0.0)
    xdmf_save_field!(xdmf, [el1, el2], 0.0, "displacement"; field_type="Vector")
    xdmf_save_field!(xdmf, [el1, el2], 0.0, "normal"; field_type="Vector")
    xdmf_save!(xdmf, "/tmp/test.xmf")
    # TODO: how to test?
end

