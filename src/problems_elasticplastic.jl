# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Elasticity problems
abstract ElasticPlasticProblem            <: AbstractProblem
abstract PlaneStressElasticPlasticProblem <: ElasticPlasticProblem

function get_unknown_field_name{P<:ElasticPlasticProblem}(::Type{P})
    return "displacement"
end

function get_unknown_field_type{P<:ElasticPlasticProblem}(::Type{P})
    return Vector{Float64}
end

# 3D Elasticity problems
function ElasticPlasticProblem(dim::Int=3, elements=[])
    return Problem{ElasticPlasticProblem}("elasticplastic problem", dim, elements)
end

# 2D Plane stress elasticity problems
function PlaneStressElasticPlasticProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressElasticPlasticProblem}("plane stress elasticplastic problem", dim, elements)
end


function get_residual_vector{P<:PlaneStressElasticPlasticProblem}(problem::Problem{P}, element::Element, ip::IntegrationPoint, time::Number; variation=nothing)
    r = zeros(Float64, problem.dim, length(element))

    J = get_jacobian(element, ip, time)

    # internal forces
    if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")
        if !haskey(element, "integration points")
            if P == PlaneStressElasticPlasticProblem
                last_stress = zeros(2,2)
                last_strain = zeros(2,2)
            else
                last_stress = zeros(3,3)
                last_strain = zeros(3,3)
            end
        else
            for each_ip in element("integration points", time)
                if isapprox(each_ip.xi, ip.xi)
                    last_stress = ip("stress", time)
                    last_strain = ip("stress", time)
                    break
                end
            end
        end
        u = element("displacement", time, variation)
        grad = element(ip, time, Val{:grad})
        gradu = grad*u

        # deformation gradient
        F = I + gradu
        E = 1/2*(F'*F - I)

        #E = 1/2*(gradu + gradu') # finite strain (total)

        # material
        young = element("youngs modulus", ip, time)
        poisson = element("poissons ratio", ip, time)
        stress_y = element("yield stress", time).data
        dstrain = E - last_strain
        de_v = [dstrain[1,1], dstrain[2,2], dstrain[1,2]]
        material_model = element("material model", time)
        s = last_stress
        de = copy(ForwardDiff.get_value(dstrain))

        if P == PlaneStressElasticPlasticProblem
            C = stiffnessTensorPlaneStress(young, poisson)
            s_v = [s[1,1], s[2,2], s[1,2]]
            de_ = [de[1,1], de[2,2], de[1,2]]
            problem_stress_type = :PlaneStressElasticPlasticProblem
        else
            C = stiffnessTensor(young, poisson)
            s_v = [s[1,1], s[2,2], s[3,3], s[2,3], s[1,3], s[1,2]]
            de_ = [de[1,1], de[2,2], de[3,3], de[2,3], de[1,3], de[1,2]]
            problem_stress_type = :ElasticPlasticProblem
        end
        dep = zeros(3)
        stress_inc, dep = calculate_stress(de_,
                                           s_v,
                                           C,
                                           stress_y,
                                           Val{:vonMises},
                                           Val{problem_stress_type})
        info("%% ", dep)
        s_v += C * (de_v - dep)
        info("--: ", ForwardDiff.get_value(s_v))
        # stress
        if P == PlaneStressElasticPlasticProblem
            S = [s_v[1] s_v[3];
                 s_v[3] s_v[2]]
        else
        S = [s_v[1] s_v[6] s_v[5];
             s_v[6] s_v[2] s_v[4];
             s_v[5] s_v[4] s_v[3]]
        end
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



#=
function get_residual_vector{P<:ElasticPlasticProblem}(problem::Problem{P}, element::Element, ip::IntegrationPoint, time::Number; variation=nothing)
    r = zeros(Float64, problem.dim, length(element))

    J = get_jacobian(element, ip, time)

    info("_____________________")
    # internal forces
    if haskey(element, "youngs modulus") && haskey(element, "poissons ratio")

        if !haskey(element, "integration points")
            if P == PlaneStressElasticPlasticProblem
                last_stress = zeros(2,2)
                last_strain = zeros(2,2)
            else
                last_stress = zeros(3,3)
                last_strain = zeros(3,3)
            end
        else
            for each_ip in element("integration points", time)
                if isapprox(each_ip.xi, ip.xi)
                    last_stress = ip("stress", time)
                    last_strain = ip("stress", time)
                    break
                end
            end
        end
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
        #E = 1/2*(gradu + gradu') # finite strain (total)

        young = element("youngs modulus", ip, time)
        poisson = element("poissons ratio", ip, time)
        stress_y = element("yield stress", time).data
        dstrain = E - last_strain
        material_model = element("material model", time)
        s = last_stress
        de = ForwardDiff.get_value(dstrain)

        if P == PlaneStressElasticPlasticProblem
            C = stiffnessTensorPlaneStress(young, poisson)
            s_v = [s[1,1], s[2,2], s[1,2]]
            de_ = [de[1,1], de[2,2], de[1,2]]
            problem_stress_type = :PlaneStressElasticPlasticProblem
        else
            C = stiffnessTensor(young, poisson)
            s_v = [s[1,1], s[2,2], s[3,3], s[2,3], s[1,3], s[1,2]]
            de_ = [de[1,1], de[2,2], de[3,3], de[2,3], de[1,3], de[1,2]]
            problem_stress_type = :ElasticPlasticProblem
        end

        stress_inc, lambda = plastic_multiplier = calculate_stress(de_,
                                                           s_v,
                                                           C,
                                                           stress_y,
                                                           Val{:vonMises},
                                                           Val{problem_stress_type})

        # dep = lambda * dfds(s)
        # upate_material_parameters!(...)
        s_new = s_v + stress_inc
        #S = [s_v[1] s_v[6] s_v[5];
    #         s_v[6] s_v[2] s_v[4];
#             s_v[5] s_v[4] s_v[3]]
        S = [s_new[1] s_new[3];
             s_new[3] s_new[2]]
        # S = C * (E - dep)


        info("Stress: ", vec(ForwardDiff.get_value(S)))
        # stress
        #S = lambda*trace(E)*I + 2*mu*E

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
=# #fff
