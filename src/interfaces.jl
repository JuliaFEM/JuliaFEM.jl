# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module interfaces

using Logging
@Logging.configure(level=DEBUG)

VERSION < v"0.4-" && using Docile

#using JuliaFEM.elasticity_solver
export solve_elasticity_interface!

"""
This is generic interface that reads data from data model, solves elasticity
problem and updates model.

Parameters
----------
model : to be defined
"""
function solve_elasticity_interface!()
  return 0
end

end
