using FactCheck
using Logging
@Logging.configure(level=INFO)

using JuliaFEM.abaqus_reader: parse_abaqus

facts("test import abaqus model") do
    fid = open("../geometry/3d_beam/palkki.inp")
    model = parse_abaqus(fid)
    close(fid)
    @fact length(model["nodes"]) => 298
    @fact length(model["elements"]) => 120
    @fact length(model["elsets"]["Body1"]) => 120
    @fact length(model["nsets"]["SUPPORT"]) => 9
    @fact length(model["nsets"]["LOAD"]) => 9
    @fact length(model["nsets"]["TOP"]) => 83
end