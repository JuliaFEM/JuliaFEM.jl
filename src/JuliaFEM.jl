# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
module JuliaFEM

using Lexicon

using Logging
@Logging.configure(level=DEBUG)

include("types.jl") # type definitions

### ELEMENTS ###
include("elements.jl")
include("lagrange.jl") # Lagrange elements
#include("hierarchical.jl") # P-elements


include("equations.jl") # formulations
include("problems.jl") # problems



include("math.jl") # basic mathematical operations -- obsolete ..?
include("elasticity_solver.jl")
include("xdmf.jl")
include("abaqus_reader.jl")
include("interfaces.jl")

end # module
