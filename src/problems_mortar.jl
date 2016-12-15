# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Parameters
----------
dimension
    dimension of surface, 1 for 2d problems (plane strain, plane stress,
    axisymmetric) and 2 for 3d problems. It not given, try to determine problem
    dimension from first element
rotate_normals
    if all surface elements are in cw order instead of ccw, this can be used to
    swap normal directions so that normals point to outward of body
adjust
    for elasticity problems only; closes any gaps between surfaces if found
dual_basis
    use bi-orthogonal basis when interpolating Lagrange multiplier space
use_forwarddiff
    use forwarddiff to linearize contact constraints directly from weighted
    gap function
distval
    charasteristic measure, contact pairs with distance over this value are
    skipped from contact segmentation algorithm
linear_surface_elements
    convert quadratic surface elements to linear elements on the fly, notice
    that middle nodes are missing Lagrange multipliers
split_quadratic_slave_elements
    split quadratic surface elements to several linear sub-elements to get
    Lagrange multiplier to middle nodes also
split_quadratic_master_elements
    split quadratic master elements to several linear sub-elements
store_fields
    not used
"""
type Mortar <: BoundaryProblem
    dimension :: Int
    rotate_normals :: Bool
    adjust :: Bool
    dual_basis :: Bool
    use_forwarddiff :: Bool
    distval :: Float64
    linear_surface_elements :: Bool
    split_quadratic_slave_elements :: Bool
    split_quadratic_master_elements :: Bool
    store_fields :: Vector{Symbol}
end

function Mortar()
    default_fields = []
    return Mortar(-1, false, false, false, false, Inf, true, true, true, default_fields)
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

