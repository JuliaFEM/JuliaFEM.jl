# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
module JuliaFEM

using Lexicon

using Logging
@Logging.configure(level=DEBUG)

"""
Simple linspace extension to multidimensional values. Contribute to julialang?
"""
function Base.linspace(X1, X2, n)
    [1/2*(1-ti)*X1 + 1/2*(1+ti)*X2 for ti in linspace(-1, 1, n)]
end


include("types.jl")  # type definitions

include("interpolate.jl")  # interpolation routines

### ELEMENTS ###
include("elements.jl")
include("lagrange.jl") # Lagrange elements
#include("hierarchical.jl") # P-elements

include("equations.jl")
include("problems.jl")
include("solvers.jl")

#include("math.jl") # basic mathematical operations -- obsolete ..?
# pre- and postprocess
include("xdmf.jl")
include("abaqus_reader.jl")
#include("interfaces.jl")

include("dirichlet.jl")
include("heat.jl")
#include("elasticity_solver.jl")

end # module
