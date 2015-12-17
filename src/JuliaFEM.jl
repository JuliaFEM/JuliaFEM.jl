# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
This is JuliaFEM -- Finite Element Package
"""
module JuliaFEM

# include("common.jl")

""" JuliaFEM Core module. """
module Core
include("core.jl")
end

module API
include("api.jl")

# export ....

end

module Preprocess
macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end
include("abaqus_reader.jl")
include("preprocess_aster_reader.jl")
end

module Postprocess
include("xdmf.jl")
end

""" JuliaFEM testing routines. """
module Test
include("test.jl")
end

module MaterialModels
include("vonmises.jl")
end

module Interfaces
include("interfaces.jl")
end

end # module
