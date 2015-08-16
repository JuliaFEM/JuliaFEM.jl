# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
module JuliaFEM


VERSION < v"0.4-" && using Docile
using Lexicon

include("types.jl") # type definitions
include("math.jl") # basic mathematical operations

include("elasticity_solver.jl")
include("xdmf.jl")
include("abaqus_reader.jl")
include("interfaces.jl")

end # module
