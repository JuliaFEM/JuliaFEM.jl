using FactCheck
using Logging
@Logging.configure(level=INFO)

using LightXML

testdata = """\
<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection" Name="Collection">
      <Geometry Type="None"/>
      <Topology Dimensions="0" Type="NoTopology"/>
      <Grid Name="Grid">
        <Time Value="123"/>
        <Geometry Type="XYZ">
          <DataItem DataType="Float" Dimensions="12" Format="XML" Precision="4">0 0 0 1 0 0 0 1 0 0 0 1</DataItem>
        </Geometry>
        <Topology Dimensions="1" Type="Mixed">
          <DataItem DataType="Int" Dimensions="5" Format="XML" Precision="4">6 0 1 2 3</DataItem>
        </Topology>
        <Attribute Center="Cell" Name="Temperature field" Type="Scalar">
          <DataItem DataType="Int" Dimensions="1" Format="XML" Precision="4">56</DataItem>
        </Attribute>
        <Attribute Center="Node" Name="Density" Type="Vector">
          <DataItem DataType="Float" Dimensions="4" Format="XML" Precision="4">1 2 4 7</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

facts("test test data") do
  xdoc = XMLDocument()
  xroot = create_root(xdoc, "Xdmf")
  set_attribute(xroot, "xmlns:xi", "http://www.w3.org/2001/XInclude")
  set_attribute(xroot, "Version", "2.1")
  domain = new_child(xroot, "Domain")
  temporal_collection = new_child(domain, "Grid")
  set_attribute(temporal_collection, "CollectionType", "Temporal")
  set_attribute(temporal_collection, "GridType", "Collection")
  set_attribute(temporal_collection, "Name", "Collection")
  geometry = new_child(temporal_collection, "Geometry")
  set_attribute(geometry, "Type", "None")
  topology = new_child(temporal_collection, "Topology")
  set_attribute(topology, "Dimensions", "0")
  set_attribute(topology, "Type", "NoTopology")

  grid = new_child(temporal_collection, "Grid")
  set_attribute(grid, "Name", "Grid")
  time = new_child(grid, "Time")
  set_attribute(time, "Value", "123")
  geometry = new_child(grid, "Geometry")
  set_attribute(geometry, "Type", "XYZ")
  dataitem = new_child(geometry, "DataItem")
  set_attribute(dataitem, "DataType", "Float")
  set_attribute(dataitem, "Dimensions", "12")
  set_attribute(dataitem, "Format", "XML")
  set_attribute(dataitem, "Precision", "4")
  add_text(dataitem, "0 0 0 1 0 0 0 1 0 0 0 1")

  topology = new_child(grid, "Topology")
  set_attribute(topology, "Dimensions", "1")
  set_attribute(topology, "Type", "Mixed")
  dataitem = new_child(topology, "DataItem")
  set_attribute(dataitem, "DataType", "Int")
  set_attribute(dataitem, "Dimensions", "5")
  set_attribute(dataitem, "Format", "XML")
  set_attribute(dataitem, "Precision", 4)
  add_text(dataitem, "6 0 1 2 3")
  attribute = new_child(grid, "Attribute")
  set_attribute(attribute, "Center", "Cell")
  set_attribute(attribute, "Name", "Temperature field")
  set_attribute(attribute, "Type", "Scalar")
  dataitem = new_child(attribute, "DataItem")
  set_attribute(dataitem, "DataType", "Int")
  set_attribute(dataitem, "Dimensions", 1)
  set_attribute(dataitem, "Format", "XML")
  set_attribute(dataitem, "Precision", 4)
  add_text(dataitem, "56")
  attribute = new_child(grid, "Attribute")
  set_attribute(attribute, "Center", "Node")
  set_attribute(attribute, "Name", "Density")
  set_attribute(attribute, "Type", "Vector")
  dataitem = new_child(attribute, "DataItem")
  set_attribute(dataitem, "DataType", "Float")
  set_attribute(dataitem, "Dimensions", 4)
  set_attribute(dataitem, "Format", "XML")
  set_attribute(dataitem, "Precision", 4)
  add_text(dataitem, "1 2 4 7")
  #save_file(xdoc, "/tmp/model.xmf")
  @fact string(xdoc) => testdata
end
