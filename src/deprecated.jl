# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Old ways to do things, we deprecate these in some schedule

function Solver{S}(::Type{S}, name::String)
    solver = Solver(S)
    solver.name = name
    return solver
end

function Problem{P<:FieldProblem}(::Type{P}, name::AbstractString, dimension::Int64)
    return Problem{P}(name, dimension, "none", [], Dict(), Assembly(), Dict(), Vector(), P())
end
