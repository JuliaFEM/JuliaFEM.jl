# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Elasticity problems

abstract ElasticityProblem <: AbstractProblem

abstract PlaneStressElasticityProblem <: ElasticityProblem

function PlaneStressElasticityProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressElasticityProblem}(dim, elements)
end

function get_unknown_field_name{P<:ElasticityProblem}(::Type{P})
    return "displacement"
end

function get_unknown_field_type{P<:ElasticityProblem}(::Type{P})
    return Vector{Float64}
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
function get_residual_vector{EL<:CG}(problem::Problem{PlaneStressElasticityProblem}, element::Element{EL}, ip::IntegrationPoint, time::Number; variation=nothing)

    basis = element(ip, time)
    dbasis = element(ip, time, Val{:grad})

    u = element("displacement", ip, time, variation)
    gradu = element("displacement", ip, time, Val{:grad}, variation)
    F = I + gradu # deformation gradient

    # internal forces
    young = element("youngs modulus", ip, time)
    poisson = element("poissons ratio", ip, time)
    mu = young/(2*(1+poisson))
    lambda = young*poisson/((1+poisson)*(1-2*poisson))
    lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for 2d
    E = 1/2*(F'*F - I)  # strain
    S = lambda*trace(E)*I + 2*mu*E

    r = F*S*dbasis

    # external forces - volume load
    if haskey(element, "displacement load")
        b = element("displacement load", ip, time)
        r -= b*basis
    end

    return vec(r)
end

""" Surface load for plane stress model. """
function get_residual_vector(problem::Problem{PlaneStressElasticityProblem}, element::Element{Seg2}, ip::IntegrationPoint, time::Number; variation=nothing)

    u = element("displacement", ip, time, variation)
    r = zeros(problem.dim, length(element))

    if haskey(element, "displacement traction force")
        T = element("displacement traction force", ip, time)
        r -= T*element(ip, time)
    end

    return vec(r)
end


#=

### 3d continuum elasticity ###

type ContinuumElasticityProblem <: ElasticityProblem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{ElasticityEquation}
end

function ContinuumElasticityProblem(equations=[])
    return PlaneStressElasticityProblem("displacement", 3, equations)
end

### Equations ###

""" 4-node plane stress element. """
type C3D10 <: ContinuumElasticityEquation
    element :: Quad4
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::CPS4)
    return (2, 4)
end

function Base.convert(::Type{PlaneStressElasticityEquation}, element::Quad4)
    integration_points = get_integration_points(element)
    if !haskey(element, "displacement")
        element["displacement"] = 0.0 => [zeros(2) for i=1:4]
    end
    CPS4(element, integration_points)
end

""" Boundary element for plane stress problem for surface loads. """
type CPS2 <: PlaneStressElasticityEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::CPS2)
    return (2, 2)
end

function Base.convert(::Type{PlaneStressElasticityEquation}, element::Seg2)
    integration_points = get_integration_points(element)
    if !haskey(element, "displacement")
        element["displacement"] = 0.0 => [zeros(2) for i=1:2]
    end
    CPS2(element, integration_points)
end

function get_residual_vector(equation::CPS2, ip::IntegrationPoint, time::Number; variation=nothing)

    element = get_element(equation)
    basis = get_basis(element)

    u = basis("displacement", ip, time, variation)
    r = zeros(size(equation))

    if haskey(element, "displacement traction force")
        T = basis("displacement traction force", ip, time)
        r -= T*basis(ip, time)
    end

    return vec(r)
end

=#


