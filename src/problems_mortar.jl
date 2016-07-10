# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

type Mortar <: BoundaryProblem
    dimension :: Int
    rotate_normals :: Bool
    adjust :: Bool
    dual_basis :: Bool
    use_forwarddiff :: Bool
    distval :: Float64
    store_fields :: Vector{Symbol}
end

function Mortar()
    default_fields = []
    return Mortar(-1, false, false, false, false, Inf, default_fields)
end

function get_unknown_field_name(problem::Problem{Mortar})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Mortar})
    if problem.properties.use_forwarddiff
        return :forwarddiff
    else
        return :incremental
    end
end

function assemble!(problem::Problem{Mortar}, time::Float64)
    if problem.properties.dimension == -1
        problem.properties.dimension = dim = size(first(problem.elements), 1)
        info("assuming dimension of mesh tie surface is $dim")
        info("if this is wrong set is manually using problem.properties.dimension")
    end
    dimension = Val{problem.properties.dimension}
    use_forwarddiff = Val{problem.properties.use_forwarddiff}
    assemble!(problem, time, dimension, use_forwarddiff)
end

