# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
module JuliaFEM

VERSION < v"0.4-" && using Docile
using Lexicon
using Logging
@Logging.configure(level=DEBUG)

include("types.jl") # type definitions
include("elements.jl") # elements
include("equations.jl") # formulations
include("math.jl") # basic mathematical operations

include("elasticity_solver.jl")
include("xdmf.jl")
include("abaqus_reader.jl")
include("interfaces.jl")

<<<<<<< HEAD
export set_coordinates, get_coordinates #,  set_material
=======
#export set_coordinates, get_coordinates, set_material
>>>>>>> 18394347dd9874ca301e7612fc776ccab08f4601

end # module
