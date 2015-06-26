# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using Logging
@Logging.configure(level=DEBUG)

using JuliaFEM.interfaces: solve_elasticity_interface!

facts("test solve_elasticity_interface!") do
  @pending
end