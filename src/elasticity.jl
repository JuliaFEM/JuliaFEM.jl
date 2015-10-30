# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Elasticity problems

abstract ElasticityProblem <: Problem
abstract ElasticityEquation <: Equation

### Formulation ###

""" Calculate internal energy for elasticity equation.

Override this to define your own material model. By default we use
Saint Venant-Kirchhoff material model, which is simply

    S(E) = λtr(E) + 2μE
"""
function get_internal_energy(equation::Equation, ip::IntegrationPoint,
                             time::Number, F::Matrix)
    element = get_element(equation)
    basis = get_basis(element)
    dbasis = grad(basis)

    # material parameters
    young = basis("youngs modulus", ip, time)
    poisson = basis("poissons ratio", ip, time)
    mu = young/(2*(1+poisson))
    lambda = young*poisson/((1+poisson)*(1-2*poisson))
    if isa(equation, PlaneStressElasticityEquation)
        lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for 2d
    end

    # material model
    E = 1/2*(F'*F - I)  # strain
    S = lambda*trace(E)*I + 2*mu*E
    P = F*S

    return P*dbasis(ip, time)
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
function get_residual_vector(equation::ElasticityEquation, ip::IntegrationPoint,
                             time::Number; variation=nothing)

    element = get_element(equation)
    basis = get_basis(element)
    dbasis = grad(basis)

    u = basis("displacement", ip, time, variation)
    gradu = dbasis("displacement", ip, time, variation)
    F = I + gradu # deformation gradient

    # residual vector - internal energy
    r = get_internal_energy(equation, ip, time, F)

    # external forces - volume load
    if haskey(element, "displacement load")
        b = basis("displacement load", ip, time)
        r -= b*basis(ip, time)
    end

    return vec(r)
end
has_residual_vector(equation::ElasticityEquation) = true

### Problem 1 - plane elasticity ###

abstract PlaneElasticityProblem <: ElasticityProblem

type PlaneStressElasticityProblem <: PlaneElasticityProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Array{ElasticityEquation, 1}
    element_mapping :: Dict{DataType, DataType}
end

function PlaneStressElasticityProblem(equations=[])
    element_mapping = Dict(
        Quad4 => CPS4)
    return PlaneStressElasticityProblem("displacement", 2, equations, element_mapping)
end

### Equations ###

abstract PlaneElasticityEquation <: ElasticityEquation
abstract PlaneStressElasticityEquation <: PlaneElasticityEquation

type CPS4 <: PlaneStressElasticityEquation
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
end
function CPS4(element::Quad4)
    integration_points = get_default_integration_points(element)
    if !haskey(element, "displacement")
        element["displacement"] = zeros(2, 4)
    end
    CPS4(element, integration_points)
end
Base.size(equation::CPS4) = (2, 4)

