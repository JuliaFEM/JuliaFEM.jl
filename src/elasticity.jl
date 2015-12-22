# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

include("elasticplastic.jl")

# Elasticity problems
abstract ElasticityProblem            <: AbstractProblem
abstract PlaneStressElasticityProblem <: ElasticityProblem

function get_unknown_field_name{P<:ElasticityProblem}(::Type{P})
    return "displacement"
end

function get_unknown_field_type{P<:ElasticityProblem}(::Type{P})
    return Vector{Float64}
end

# 3D Elasticity problems
function ElasticityProblem(dim::Int=3, elements=[])
    return Problem{ElasticityProblem}("elasticity problem", dim, elements)
end

# 2D Plane stress elasticity problems
function PlaneStressElasticityProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressElasticityProblem}("plane stress elasticity problem", dim, elements)
end
function PlaneStressElasticityProblem(problem_name::ASCIIString, dim::Int=2, elements=[])
    return Problem{PlaneStressElasticityProblem}(problem_name, dim, elements)
end


""" Elasticity equations.

Formulation
-----------

Field equation is:
∂u/∂t = ∇⋅f - b

Weak form is: find u∈U such that ∀v in V

    δW := ∫ρ₀∂²u/∂t²⋅δu dV₀ + ∫S:δE dV₀ - ∫b₀⋅δu dV₀ - ∫t₀⋅δu dA₀ = 0

where

    ρ₀ = density
    b₀ = displacement load
    t₀ = displacement traction

References
----------

https://en.wikipedia.org/wiki/Linear_elasticity
https://en.wikipedia.org/wiki/Finite_strain_theory
https://en.wikipedia.org/wiki/Stress_measures
https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
https://en.wikipedia.org/wiki/Strain_energy_density_function
https://en.wikipedia.org/wiki/Plane_stress
https://en.wikipedia.org/wiki/Hooke's_law

"""
function get_residual_vector{P<:ElasticityProblem}(problem::Problem{P}, element::Element, ip::IntegrationPoint, time::Number; variation=nothing)
    r = zeros(Float64, problem.dim, length(element))

    J = get_jacobian(element, ip, time)

    # internal forces
    if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
        u = element("displacement", time, variation)
        grad = element(ip, time, Val{:grad})
        gradu = grad*u

        # deformation gradient
        F = I + gradu

        # material
        young = element("youngs modulus", ip, time)
        poisson = element("poissons ratio", ip, time)
        mu = young/(2*(1+poisson))
        lambda = young*poisson/((1+poisson)*(1-2*poisson))
        if P == PlaneStressElasticityProblem
            lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for 2d problems
        end

        # strain
        E = 1/2*(F'*F - I)
        # stress
        S = lambda*trace(E)*I + 2*mu*E

        r += F*S*grad*det(J)
    end

    # external forces - volume load
    if haskey(element, "displacement load")
        basis = element(ip, time)
        b = element("displacement load", ip, time)
        r -= b*basis*det(J)
    end

    # external forces - surface traction force
    if haskey(element, "displacement traction force")
        basis = element(ip, time)
        T = element("displacement traction force", ip, time)
        JT = transpose(J)
        s = size(JT, 2) == 1 ? JT : cross(JT[:,1], JT[:,2])
        r -= T*basis*norm(s)
    end

    return vec(r)
end
