import FactCheck
import Logging
include("badges.jl")

cd(dirname(@__FILE__)) do
    Logging.configure(output=open("quality/unittests_report.rst", "w"))
    include("../test/runtests.jl")
    s = FactCheck.getstats()
    make_badge("unittests", s["nSuccesses"], sum(values(s)), "badges/unittests-status.svg")
end

