# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Concrete Elasticity type. """
type Elasticity <: FieldProblem
    # these are found from problem.properties for type Problem{Elasticity}
    formulation :: Symbol
    nonlinear_geometry :: Bool
end
function Elasticity()
    Elasticity(
        :continuum,   # formulations: :plane_stress, :continuum
        false,        # geometrically nonlinear analysis
    )
end

# in case of experimenting new things;
# 1. import JuliaFEM.Core: assemble!
# 2. copy/paste assemble! code to notebook
# 3. change to last argument, i.e. ::Type{Val{:plane_stress}} to ::Type{Val{:my_formulation}}
# 4. when running code: set problem.properties.formulation = :my_formulation
# 5. let multiple dispatch do the magic for you

function get_unknown_field_name(::Type{Elasticity})
    return "displacement"
end

function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time::Real)
    return assemble!(assembly, problem, element, time, Val{problem.properties.formulation})
end

""" Elasticity equations, plane stress formulation. """
function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time::Real, ::Type{Val{:plane_stress}})

    gdofs = get_gdofs(problem, element)
    ndim, nnodes = size(element)
    B = zeros(3, 2*nnodes)
    for ip in get_integration_points(element)
        w = ip.weight
        J = get_jacobian(element, ip, time)
        N = element(ip, time)
        if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
            nu = element("poissons ratio", ip, time)
            E_ = element("youngs modulus", ip, time)
            C = E_/(1.0 - nu^2) .* [
                1.0  nu 0.0
                nu  1.0 0.0
                0.0 0.0 (1.0-nu)/2.0]
            dN = element(ip, time, Val{:grad})
            fill!(B, 0.0)
            for i=1:size(dN, 2)
                B[1, 2*(i-1)+1] = dN[1,i]
                B[2, 2*(i-1)+2] = dN[2,i]
                B[3, 2*(i-1)+1] = dN[2,i]
                B[3, 2*(i-1)+2] = dN[1,i]
            end
            Kt = w*B'*C*B*det(J)
            add!(assembly.K, gdofs, gdofs, Kt)
        end
        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            add!(assembly.f, gdofs, w*N'*b*det(J))
        end
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            L = w*T*N*norm(J)
            add!(assembly.f, gdofs, vec(L))
        end
        for dim in 1:get_unknown_field_dimension(problem)
            if haskey(element, "displacement traction force $dim")
                T = element("displacement traction force $dim", ip, time)
                ldofs = gdofs[dim:problem.dim:end]
                L = w*T*N*norm(J)
                add!(assembly.f, ldofs, vec(L))
            end
        end
        if haskey(element, "displacement traction force N")
            # surface pressure
            p = zeros(2)
            p[1] = element("displacement traction force N", ip, time)
            R = element("normal-tangential coordinates", ip, time)
            T = R'*p
            L = w*T*N*norm(J)
            add!(assembly.f, gdofs, vec(L))
        end
    end
end


""" Elasticity equations, continuum formulation. """
function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time::Real, ::Type{Val{:continuum}})

    gdofs = get_gdofs(problem, element)
    ndim, nnodes = size(element)
    B = zeros(6, 3*nnodes)
    for ip in get_integration_points(element)
        w = ip.weight
        J = get_jacobian(element, ip, time)
        N = element(ip, time)
        if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
            v = element("poissons ratio", ip, time)
            E_ = element("youngs modulus", ip, time)
            a = 1 - v
            b = 1 - 2*v
            c = 1 + v
            C = E_/(b*c) .* [
                a v v 0 0 0
                v a v 0 0 0
                v v a 0 0 0
                0 0 0 b 0 0
                0 0 0 0 b 0
                0 0 0 0 0 b]
            dN = element(ip, time, Val{:grad})
            fill!(B, 0.0)
            for i=1:size(dN, 2)
                B[1, 3*(i-1)+1] = dN[1,i]
                B[2, 3*(i-1)+2] = dN[2,i]
                B[3, 3*(i-1)+3] = dN[3,i]
                B[4, 3*(i-1)+1] = dN[2,i]
                B[4, 3*(i-1)+2] = dN[1,i]
                B[5, 3*(i-1)+2] = dN[3,i]
                B[5, 3*(i-1)+3] = dN[2,i]
                B[6, 3*(i-1)+1] = dN[3,i]
                B[6, 3*(i-1)+3] = dN[1,i]
            end
            # L = b * B'
            # D = 0.5 * (L' + L)
            # F = ...
            # E = 0.5 * (F'*F - I)
            # de = E - E_last
            # S = vonMisesStress(de, stress)
            # K = B' * S * J * w
            Kt = w*B'*C*B*det(J)
            add!(assembly.K, gdofs, gdofs, Kt)
        end
        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            add!(assembly.f, gdofs, w*N'*b*det(J))
        end
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            JT = transpose(J)
            L = w*T*N*norm(cross(JT[:,1], JT[:,2]))
            add!(assembly.f, gdofs, vec(L))
        end
        for dim in 1:problem.dim
            if haskey(element, "displacement traction force $dim")
                T = element("displacement traction force $dim", ip, time)
                ldofs = gdofs[dim:problem.dim:end]
                JT = transpose(J)
                L = w*T*N*norm(cross(JT[:,1], JT[:,2]))
                add!(assembly.f, ldofs, vec(L))
            end
        end
    end
end


###############################
#     Plastic material        #
###############################
#=
include("vonmises.jl")
abstract PlaneStressLinearElasticPlasticProblem <: LinearElasticityProblem

function PlaneStressLinearElasticPlasticProblem(name="plane stress linear elasticity", dim::Int=2, elements=[])
    return Problem{PlaneStressLinearElasticPlasticProblem}(name, dim, elements)
end

""" Elasticity equations, plane stress. """
function assemble!{E<:CG, P<:PlaneStressLinearElasticPlasticProblem}(assembly::Assembly, problem::Problem{P}, element::Element{E}, time::Real)

    gdofs = get_gdofs(element, problem.dim)
    ndim, nnodes = size(E)
    B = zeros(3, 2*nnodes)
    for ip in get_integration_points(element)
        w = ip.weight
        J = get_jacobian(element, ip, time)
        N = element(ip, time)
        if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
            nu = element("poissons ratio", ip, time)
            E_ = element("youngs modulus", ip, time)
            C = E_/(1.0 - nu^2) .* [
                1.0  nu 0.0
                nu  1.0 0.0
                0.0 0.0 (1.0-nu)/2.0]
            dN = element(ip, time, Val{:grad})
            fill!(B, 0.0)
            for i=1:size(dN, 2)
                B[1, 2*(i-1)+1] = dN[1,i]
                B[2, 2*(i-1)+2] = dN[2,i]
                B[3, 2*(i-1)+1] = dN[2,i]
                B[3, 2*(i-1)+2] = dN[1,i]
            end
            add!(assembly.stiffness_matrix, gdofs, gdofs, w*B'*C*B*det(J))
        end
        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*b*det(J))
        end
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            L = w*T*N*norm(J)
            add!(assembly.force_vector, gdofs, vec(L))
        end
    end
end





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
        if problem.properties.formulation == :plane_stress
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

=#
