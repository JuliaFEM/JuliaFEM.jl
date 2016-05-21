# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Concrete Elasticity type. """
type Elasticity <: FieldProblem
    # these are found from problem.properties for type Problem{Elasticity}
    formulation :: Symbol
    finite_strain :: Bool
    use_forwarddiff :: Bool
end
function Elasticity()
    # formulations: plane_stress, plane_strain, continuum
    return Elasticity(:continuum, true, false)
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
    # we are solving residual and add increment to previous solution vector
    return :incremental
end

function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time::Real)
    props = problem.properties
    gdofs = get_gdofs(problem, element)
    if props.use_forwarddiff
        Kt, f = assemble(problem, element, time, Val{:forwarddiff})
    elseif props.formulation == :continuum
        Kt, f = assemble(problem, element, time, Val{:continuum})
    elseif (props.formulation == :plane_stress) || (props.formulation == :plane_strain)
        Kt, f = assemble(problem, element, time, Val{:plane})
    end
    add!(assembly.K, gdofs, gdofs, Kt)
    add!(assembly.f, gdofs, f)
end

function assemble(problem::Problem{Elasticity}, element::Element, time=0.0)
    assemble(problem, element, time, Val{problem.properties.formulation})
end

""" Elasticity equations for 2d cases. """
function assemble{El<:Union{Tri3,Tri6,Quad4}}(problem::Problem{Elasticity}, element::Element{El}, time, ::Type{Val{:plane_stress}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    BL = zeros(3, dim*nnodes)
    BNL = zeros(4, dim*nnodes)
    Kt = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for (w, xi) in get_integration_points(element)

        J = element(xi, time, Val{:Jacobian})
        w = w*det(J)
        N = element(xi, time)
        dN = element(xi, time, Val{:Grad})

        # kinematics; calculate deformation gradient and strain
        gradu = zeros(dim, dim)
        if haskey(element, "displacement")
            gradu += element("displacement", xi, time, Val{:Grad})
        end
        strain = zeros(dim , dim)
        strain += 1/2*(gradu' + gradu)
        F = eye(dim)
        if props.finite_strain
            F += gradu
            strain += 1/2*gradu'*gradu
        end

        # constitutive equations; material model (isotropic linear material here)
        # get_material(problem, element, ...)
        E = element("youngs modulus", xi, time)
        nu = element("poissons ratio", xi, time)
        D = E/(1.0 - nu^2) .* [
            1.0  nu 0.0
            nu  1.0 0.0
            0.0 0.0 (1.0-nu)/2.0]

        # calculate stress
        S = D*[strain[1,1]; strain[2,2]; 2*strain[1,2]]

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

        Kt += w*BL'*D*BL # material stiffness
        if props.finite_strain # add geometric stiffness
            Kt += w*BNL'*S2*BNL # geometric stiffness
        end
        f -= w*BL'*S # internal force

        # volume load
        if haskey(element, "displacement load")
            b = element("displacement load", xi, time)
            f += vec(w*N'*b)
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
function assemble{El<:Union{Tet4, Tet10, Hex8}}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)
    BL = zeros(6, dim*nnodes)
    BNL = zeros(9, dim*nnodes)
    Kt = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)
        J = get_jacobian(element, ip, time)
        w = ip.weight*det(J)
        N = element(ip, time)
        dN = element(ip, time, Val{:grad})

        # kinematics; calculate deformation gradient and strain
        gradu = zeros(dim, dim)
        if haskey(element, "displacement")
            gradu += element("displacement", ip, time, Val{:grad})
        end
        strain = zeros(dim , dim)
        strain += 1/2*(gradu' + gradu)
        F = eye(dim)
        if props.finite_strain
            F += gradu
            strain += 1/2*gradu'*gradu
        end

        E = element("youngs modulus", ip, time)
        nu = element("poissons ratio", ip, time)

        a = 1 - nu
        b = 1 - 2*nu
        c = 1 + nu
        D = E/(b*c) .* [
            a nu nu 0 0 0
            nu a nu 0 0 0
            nu nu a 0 0 0
            0 0 0 b 0 0
            0 0 0 0 b 0
            0 0 0 0 0 b]

        # # PK2 stress tensor in voigt notation
        S = D*[strain[1,1]; strain[2,2]; strain[3,3]; 2*strain[2,3]; 2*strain[1,3]; 2*strain[1,2]]

        # add contributions: material and geometric stiffness + internal forces
        fill!(BL, 0.0)
        for i=1:size(dN, 2)
            BL[1, 3*(i-1)+1] = F[1,1]*dN[1,i]
            BL[1, 3*(i-1)+2] = F[2,1]*dN[1,i]
            BL[1, 3*(i-1)+3] = F[3,1]*dN[1,i]
            BL[2, 3*(i-1)+1] = F[1,2]*dN[2,i]
            BL[2, 3*(i-1)+2] = F[2,2]*dN[2,i]
            BL[2, 3*(i-1)+3] = F[3,2]*dN[2,i]
            BL[3, 3*(i-1)+1] = F[1,3]*dN[3,i]
            BL[3, 3*(i-1)+2] = F[2,3]*dN[3,i]
            BL[3, 3*(i-1)+3] = F[3,3]*dN[3,i]
            BL[4, 3*(i-1)+1] = F[1,1]*dN[2,i] + F[1,2]*dN[1,i]
            BL[4, 3*(i-1)+2] = F[2,1]*dN[2,i] + F[2,2]*dN[1,i]
            BL[4, 3*(i-1)+3] = F[3,1]*dN[2,i] + F[3,2]*dN[1,i]
            BL[5, 3*(i-1)+1] = F[1,2]*dN[3,i] + F[1,3]*dN[2,i]
            BL[5, 3*(i-1)+2] = F[2,2]*dN[3,i] + F[2,3]*dN[2,i]
            BL[5, 3*(i-1)+3] = F[3,2]*dN[3,i] + F[3,3]*dN[2,i]
            BL[6, 3*(i-1)+1] = F[1,3]*dN[1,i] + F[1,1]*dN[3,i]
            BL[6, 3*(i-1)+2] = F[2,3]*dN[1,i] + F[2,1]*dN[3,i]
            BL[6, 3*(i-1)+3] = F[3,3]*dN[1,i] + F[3,1]*dN[3,i]
        end
        fill!(BNL, 0.0)
        for i=1:size(dN, 2)
            BNL[1, 3*(i-1)+1] = dN[1,i]
            BNL[2, 3*(i-1)+1] = dN[2,i]
            BNL[3, 3*(i-1)+1] = dN[3,i]
            BNL[4, 3*(i-1)+2] = dN[1,i]
            BNL[5, 3*(i-1)+2] = dN[2,i]
            BNL[6, 3*(i-1)+2] = dN[3,i]
            BNL[7, 3*(i-1)+3] = dN[1,i]
            BNL[8, 3*(i-1)+3] = dN[2,i]
            BNL[9, 3*(i-1)+3] = dN[3,i]
        end
        S3 = zeros(3*dim, 3*dim)
        S3[1,1] = S[1]
        S3[2,2] = S[2]
        S3[3,3] = S[3]
        S3[2,3] = S3[3,2] = S[4]
        S3[1,3] = S3[3,1] = S[5]
        S3[1,2] = S3[2,1] = S[6]
        S3[4:6,4:6] = S3[7:9,7:9] = S3[1:3,1:3]

        Kt += w*BL'*D*BL
        if props.finite_strain
            Kt += w*BNL'*S3*BNL
        end
        f -= w*BL'*S

        # volume load
        if haskey(element, "displacement load")
            T = element("displacement load", ip, time)
            f += vec(w*T*N)
        end
    end

    return Kt, f
end

""" Elasticity equations, surface traction for continuum formulation. """
function assemble{El<:Union{Tri3, Tri6, Quad4}}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)
    Kt = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)
        JT = transpose(get_jacobian(element, ip, time))
        N = element(ip, time)
        w = ip.weight*norm(cross(JT[:,1], JT[:,2]))
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            f += vec(w*T*N)
        end
        for i in 1:dim
            if haskey(element, "displacement traction force $i")
                T = element("displacement traction force $i", ip, time)
                f[i:dim:end] += vec(w*T*N)
            end
        end
    end
    return Kt, f
end

""" Elasticity equations using ForwardDiff

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
function assemble(problem::Problem{Elasticity}, element::Element, time::Real, ::Type{Val{:forwarddiff}})

    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)

    function get_residual_vector(u::Vector)
        u = reshape(u, dim, nnodes)
        u = Field([u[:,i] for i=1:nnodes])
        r = zeros(dim, nnodes)

        for ip in get_integration_points(element)

            JT = transpose(get_jacobian(element, ip, time))
            n, m = size(JT)
            if n == m
                w = ip.weight*det(JT)
            elseif m == 1
                w = ip.weight*norm(JT)
            elseif m == 2
                w = ip.weight*norm(cross(JT[:,1], JT[:,2]))
            else
                error("jacobian $JT")
            end

            # calculate internal forces
            if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
                grad = element(ip, time, Val{:grad})
                gradu = grad*u

                # kinematics
                F = I + gradu
                E = 1/2*(F'*F - I)

                # material
                young = element("youngs modulus", ip, time)
                poisson = element("poissons ratio", ip, time)
                mu = young/(2*(1+poisson))
                lambda = young*poisson/((1+poisson)*(1-2*poisson))
                if problem.properties.formulation == :plane_stress
                    lambda = 2*lambda*mu/(lambda + 2*mu)  # <- correction for plane stress
                end

                # stress
                S = lambda*trace(E)*I + 2*mu*E

                r += w*F*S*grad
            end

            # calculate external forces - volume load
            if haskey(element, "displacement load")
                basis = element(ip, time)
                b = element("displacement load", ip, time)
                r -= w*b*basis
            end

            # external forces - surface traction force
            if haskey(element, "displacement traction force")
                basis = element(ip, time)
                T = element("displacement traction force", ip, time)
                r -= w*T*basis
            end

        end

        return vec(r)

    end

    field = element("displacement", time)
    Kt, allresults = ForwardDiff.jacobian(get_residual_vector, vec(field),
                     AllResults, cache=autodiffcache)
    f = -ForwardDiff.value(allresults)
    return Kt, f
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



=#
