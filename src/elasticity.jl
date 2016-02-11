# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Concrete Elasticity type. """
type Elasticity <: FieldProblem
    # these are found from problem.properties for type Problem{Elasticity}
    formulation :: Symbol
end
function Elasticity()
    # formulations: plane_stress, plane_strain, continuum
    return Elasticity(:continuum)
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

function get_formulation_type(problem::Problem{Elasticity})
    info("INCREMENTAL FORMULATION")
    return :incremental
end

function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time::Real)
    props = problem.properties
    if props.formulation == :continuum
        return assemble!(assembly, problem, element, time, Val{:continuum})
    elseif (props.formulation == :plane_stress) || (props.formulation == :plane_strain)
        gdofs = get_gdofs(problem, element)
        Kt, f = assemble(problem, element, time, Val{:plane})
        add!(assembly.K, gdofs, gdofs, Kt)
        add!(assembly.f, gdofs, f)
    end
end


""" Elasticity equations for 2d cases. """
function assemble{El<:Union{Tri3,Tri6,Quad4}}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:plane}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)
    BL = zeros(3, dim*nnodes)
    BNL = zeros(4, dim*nnodes)
    Kt = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)

        J = get_jacobian(element, ip, time)
        w = ip.weight*det(J)
        N = element(ip, time)
        dN = element(ip, time, Val{:grad})

        # kinematics; calculate deformation gradient and strain
        F = eye(dim)
        if haskey(element, "displacement")
            gradu = element("displacement", ip, time, Val{:grad})
            F += gradu
        end
        GL = 1/2*(F'*F - I) # green-lagrange strain

        # constitutive equations; material model (isotropic linear material here)
        # get_material(problem, element, ...)
        E = element("youngs modulus", ip, time)
        nu = element("poissons ratio", ip, time)
        if props.formulation == :plane_stress
            D = E/(1.0 - nu^2) .* [
                1.0  nu 0.0
                nu  1.0 0.0
                0.0 0.0 (1.0-nu)/2.0]
        elseif props.formulation == :plane_strain
            D = E/((1+nu)*(1-2*nu)) .* [
                1-nu   nu 0
                nu   1-nu 0
                0    0    (1-2*nu)/2]
        else
            error("unknown plane formulation: $(props.formulation)")
        end
        S = D*[GL[1,1]; GL[2,2]; 2*GL[1,2]] # PK2 stress tensor in voigt notation

        # add contributions: material and geometric stiffness + internal forces
        fill!(BL, 0.0)
        for i=1:size(dN, 2)
            BL[1, 2*(i-1)+1] = F[1,1]*dN[1,i]
            BL[1, 2*(i-1)+2] = F[2,1]*dN[1,i]
            BL[2, 2*(i-1)+1] = F[1,2]*dN[2,i]
            BL[2, 2*(i-1)+2] = F[2,2]*dN[2,i]
            BL[3, 2*(i-1)+1] = F[1,1]*dN[2,i] + F[1,2]*dN[1,i]
            BL[3, 2*(i-1)+2] = F[2,1]*dN[2,i] + F[2,2]*dN[1,i]
        end
        fill!(BNL, 0.0)
        for i=1:size(dN, 2)
            BNL[1, 2*(i-1)+1] = dN[1,i]
            BNL[2, 2*(i-1)+1] = dN[2,i]
            BNL[3, 2*(i-1)+2] = dN[1,i]
            BNL[4, 2*(i-1)+2] = dN[2,i]
        end
        S2 = zeros(2*dim, 2*dim)
        S2[1,1] = S[1]
        S2[2,2] = S[2]
        S2[1,2] = S2[2,1] = S[3]
        S2[3:4,3:4] = S2[1:2,1:2]

        Kt += w*(BL'*D*BL + BNL'*S2*BNL)
        f -= w*BL'*S

        # volume load
        if haskey(element, "displacement load")
            T = element("displacement load", ip, time)
            f += vec(w*T*N)
        end

    end

    return Kt, f
end

function assemble{El<:Union{Seg2,Seg3}}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:plane}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)
    Kt = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)

        J = get_jacobian(element, ip, time)
        N = element(ip, time)
        w = ip.weight*norm(J)

        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            f += vec(w*T*N)
        end

        for i=1:dim
            # traction force for ith component
            if haskey(element, "displacement traction force $i")
                T = element("displacement traction force $i", ip, time)
                f[i:dim:end] += vec(w*T*N)
            end
        end

        if haskey(element, "nt displacement traction force")
            # traction force given in normal-tangential direction
            T = element("nt displacement traction force", ip, time)
            Q = element("normal-tangential coordinates", ip, time)
            f += vec(w*Q'*T*N)
        end

    end

    return Kt, f
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
        for dim in 1:get_unknown_field_dimension(problem)
            if haskey(element, "displacement traction force $dim")
                T = element("displacement traction force $dim", ip, time)
                ldofs = gdofs[dim:unknown_field_dimension(problem):end]
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
