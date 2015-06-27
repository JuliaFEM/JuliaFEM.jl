# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using Logging
@Logging.configure(level=INFO)

using JuliaFEM.abaqus_reader: parse_abaqus, parse_element_section

facts("test import abaqus model") do
  # FIXME: get_test_data()
    fid = open(Pkg.dir("JuliaFEM")*"/geometry/3d_beam/palkki.inp")
    model = parse_abaqus(fid)
    close(fid)
    @fact length(model["nodes"]) => 298
    @fact length(model["elements"]) => 120
    @fact length(model["elsets"]["Body1"]) => 120
    @fact length(model["nsets"]["SUPPORT"]) => 9
    @fact length(model["nsets"]["LOAD"]) => 9
    @fact length(model["nsets"]["TOP"]) => 83
end

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

facts("test unknown handler warning message") do
  fn = tempname()
  fid = open(fn, "w")
  testdata = """
  *ELEMENT2, TYPE=C3D10, ELSET=Body1
         1,       243,       240,       191,       117,       245,       242,       244,
         1,         2,       196
  """
  write(fid, testdata)
  close(fid)
  fid = open(fn)
  model = parse_abaqus(fid)
  close(fid)
  # empty model expected, parser doesn't know what to do with unknown section
  @fact length(model) => 0
end
