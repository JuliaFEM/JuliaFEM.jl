# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

abstract type PlaneStressElasticityProblem <: AbstractProblem end

function PlaneStressElasticityProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressElasticityProblem}(dim, elements)
end

function get_unknown_field_name(::Type{P}) where P<:PlaneStressElasticityProblem
    return "displacement"
end

function get_unknown_field_type(::Type{P}) where P<:PlaneStressElasticityProblem
    return Vector{Float64}
end

function get_residual_vector(problem::Problem{PlaneStressElasticityProblem}, element::Element, ip::IntegrationPoint, time::Number; variation=nothing)


    basis = element(ip, time)
    dbasis = element(ip, time, Val{:grad})

    # material parameters
    E = element("youngs modulus", ip, time)
    nu = element("poissons ratio", ip, time)
    mu = E/(2*(1+nu))
    la = E*nu/((1+nu)*(1-2*nu))
    la = 2*la*mu/(la + 2*mu)  # <- correction for 2d

    # elasticity formulation
    u = element("displacement", ip, time, variation)
    gradu = element("displacement", ip, time, Val{:grad}, variation)
    F = I + gradu

    E = 1/2*(F'*F - I)
    S = la*trace(E)*I + 2*mu*E
    r = F*S*dbasis

    b = element("displacement volume load", ip, time)
    r -= b*basis

    return vec(r)
end

function test_residual_form()
    # create model -- start
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0,0.0], [10.0,0.0], [10.0,1.0], [0.0,1.0]]
    element["youngs modulus"] = 500.0
    element["poissons ratio"] = 0.3
    element["displacement volume load"] = Vector[[0.0,-10.0], [0.0,-10.0], [0.0,-10.0], [0.0,-10.0]]
    element["displacement"] = (0.0 => Vector{Float64}[zeros(2) for i=1:length(element)])
    problem = PlaneStressElasticityProblem()
    push!(problem, element)
    # create model -- end

    free_dofs = [3, 4, 5, 6]
    solve!(problem, free_dofs, 0.0)  # launch a newton solver for single element
    disp = element("displacement", [1.0, 1.0], 0.0)
    @info("displacement at tip: $disp")

    # verified using Code Aster.
    @test isapprox(disp[2], -8.77303119819776E+00)
end
