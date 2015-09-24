# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
These fields are used as a drop-in replacement for element fields. They
support time and increments so field values can be time dependent, even
when interpolating. By default the last values are used.

Examples
========

>>> fs = FieldSet()
>>> fs[:displacement] = Vector[[1.0]]

>>> fs[:displacement]
1-element Array{Array{T,1},1}:
 [1.0]

>>> fs[1][1][:displacement]
1-element Array{Array{T,1},1}:
 [1.0]

>>> fs[end][end]:displacement]
1-element Array{Array{T,1},1}:
 [1.0]

>>> fs[1].time
0.0

>>> fs[1][end].increment_number
0

=#

type Increment
    increment_number :: Int
    fields :: Dict{Any, Any}
end
function Increment()
    Increment(1, Dict())
end
function Base.getindex(inc::Increment, key)
    inc.fields[key]
end
function Base.setindex!(inc::Increment, vals, key)
    inc.fields[key] = vals
end


type TimeStep
    time :: Float64
    increments :: Array{Increment, 1}
end
function TimeStep()
    TimeStep(0.0, [Increment()])
end
function Base.getindex(ts::TimeStep, idx::Int64)
    ts.increments[idx]
end
function Base.endof(ts::TimeStep)
    return length(ts.increments)
end


type FieldSet
    timesteps :: Array{TimeStep, 1}
end
function FieldSet()
    FieldSet() = FieldSet([TimeStep()])
end
function Base.getindex(fs::FieldSet, idx::Int64)
    fs.timesteps[idx]
end
function Base.getindex(fs::FieldSet, key)
    fs[end][end][key]
end
function Base.setindex!(fs::FieldSet, vals, key)
    fs[end][end][key] = vals
end
function Base.endof(fs::FieldSet)
    return length(fs.timesteps)
end

