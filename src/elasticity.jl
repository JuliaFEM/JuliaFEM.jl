# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

include("vonmises.jl")

# Elasticity problems

abstract ElasticityProblem <: AbstractProblem
abstract ElasticPlasticProblem <: AbstractProblem

function get_unknown_field_name{P<:ElasticityProblem}(::Type{P})
    return "displacement"
end

function get_unknown_field_type{P<:ElasticityProblem}(::Type{P})
    return Vector{Float64}
end

function get_unknown_field_name{P<:ElasticPlasticProblem}(::Type{P})
    return "displacement"
end

function get_unknown_field_type{P<:ElasticPlasticProblem}(::Type{P})
    return Vector{Float64}
end

function ElasticityProblem(dim::Int=3, elements=[])
    return Problem{ElasticityProblem}(dim, elements)
end

function ElasticPlasticProblem(dim::Int=3, elements=[])
    return Problem{ElasticPlasticProblem}(dim, elements)
end

abstract PlaneStressElasticityProblem <: ElasticityProblem

function PlaneStressElasticityProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressElasticityProblem}(dim, elements)
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


    # u = element("displacement", ip, time, variation)

    r = zeros(Float64, problem.dim, length(element))

    # internal forces
    if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
        u = element("displacement", time, variation)
        grad = element(ip, time, Val{:grad})
#        gradu = element("displacement", ip, time, Val{:grad}, variation)
        gradu = grad*u

        F = I + gradu # deformation gradient

        young = element("youngs modulus", ip, time)
        poisson = element("poissons ratio", ip, time)
        mu = young/(2*(1+poisson))
        lambda = young*poisson/((1+poisson)*(1-2*poisson))
        if P == PlaneStressElasticityProblem
            lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for 2d problems
        end
        E = 1/2*(F'*F - I)  # strain
        S = lambda*trace(E)*I + 2*mu*E

        #J = det(element, ip, time)
        #T = J^-1*F*S*F'
        #ip["cauchy stress"] = T
        #ip["gl strain"] = E

        r += F*S*grad
    end

    # external forces - volume load
    if haskey(element, "displacement load")
        basis = element(ip, time)
        b = element("displacement load", ip, time)
        r -= b*basis
    end

    # external forces - surface traction force
    if haskey(element, "displacement traction force")
        basis = element(ip, time)
        T = element("displacement traction force", ip, time)
        r -= T*basis
    end

    return vec(r)
end

"""

"""
function get_residual_vector{P<:ElasticPlasticProblem}(problem::Problem{P}, element::Element, ip::IntegrationPoint, time::Number; variation=nothing)


    # u = element("displacement", ip, time, variation)

    r = zeros(Float64, problem.dim, length(element))

    # internal forces
    if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
        
        if !haskey(element, "integration points")
            last_stress = zeros(3,3)
            last_strain = zeros(3,3)
        else
            for each_ip in element("integration points", time)
                if isapprox(each_ip.xi, ip.xi)
                    last_stress = ip("stress", time)
                    last_strain = ip("stress", time)
                    break
                end
            end
        end

        # last_ip = get_last_ip(problem, element, ip, time)
        # stress_base = last_ip("stress")
        u = element("displacement", time, variation)
        grad = element(ip, time, Val{:grad})
#        gradu = element("displacement", ip, time, Val{:grad}, variation)
        gradu = grad*u

        F = I + gradu # deformation gradient

        young = element("youngs modulus", ip, time)
        poisson = element("poissons ratio", ip, time)

        C = stiffnessTensor(young, poisson)
        mu = young/(2*(1+poisson))
        lambda = young*poisson/((1+poisson)*(1-2*poisson))
        if P == PlaneStressElasticityProblem
            lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for 2d problems
        end
        stress_y = element("yield stress", time).data
        #E = 1/2*(F'*F - I)        # large strain
        E = 1/2*(gradu + gradu') # finite strain (total)
        dstrain = E - last_strain
        material_model = element("material model", time)
        s = last_stress
        de = ForwardDiff.get_value(dstrain)
        s_v = [s[1,1], s[2,2], s[3,3], s[2,3], s[1,3], s[1,2]]
        de_ = [de[1,1], de[2,2], de[3,3], de[2,3], de[1,3], de[1,2]]
        #println("stress: ", s_v)
        #println("de: ",  de_)
        #println(C)
        #println("yield stress: ", stress_y)
        plastic_multiplier = calculate_stress!(de_, s_v, C, stress_y, Val{:vonMises})
        # dep = lambda * dfds(s)
        # upate_material_parameters!(...)
        S = [s_v[1] s_v[6] s_v[5];
             s_v[6] s_v[2] s_v[4];
             s_v[5] s_v[4] s_v[3]]
        # S = C * (E - dep)
        #S = lambda*trace(E)*I + 2*mu*E

        #J = det(element, ip, time)
        #T = J^-1*F*S*F'
        #ip["cauchy stress"] = T
        #ip["gl strain"] = E

        r += F*S*grad
    end

    # external forces - volume load
    if haskey(element, "displacement load")
        basis = element(ip, time)
        b = element("displacement load", ip, time)
        r -= b*basis
    end

    # external forces - surface traction force
    if haskey(element, "displacement traction force")
        basis = element(ip, time)
        T = element("displacement traction force", ip, time)
        r -= T*basis
    end

    return vec(r)
end
