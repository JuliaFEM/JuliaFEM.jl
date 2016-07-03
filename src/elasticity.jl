# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Elasticity equations.

Field equation is:
    
    m∂²u/∂t² = ∇⋅σ - b

Weak form is: find u∈U such that ∀v in V

    δW := ∫ρ₀∂²u/∂t²⋅δu dV₀ + ∫S:δE dV₀ - ∫b₀⋅δu dV₀ - ∫t₀⋅δu dA₀ = 0

where

    ρ₀ = density
    b₀ = displacement load
    t₀ = displacement traction

Formulations
------------
plane stress, plane strain, 3D

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
type Elasticity <: FieldProblem
    # these are found from problem.properties for type Problem{Elasticity}
    formulation :: Symbol
    finite_strain :: Bool
    geometric_stiffness :: Bool
    store_fields :: Vector{ASCIIString}
end
function Elasticity()
    # formulations: plane_stress, plane_strain, continuum
    return Elasticity(:continuum, false, false, [])
end

function get_unknown_field_name(problem::Problem{Elasticity})
    return "displacement"
end

function get_formulation_type(problem::Problem{Elasticity})
    return :incremental
end

function assemble!(assembly::Assembly, problem::Problem{Elasticity}, element::Element, time=0.0)
    props = problem.properties
    gdofs = get_gdofs(problem, element)
    formulation = props.formulation
    if formulation in [:plane_stress, :plane_strain]
        formulation = :plane
    end
    Km, Kg, f = assemble(problem, element, time, Val{formulation})
    add!(assembly.K, gdofs, gdofs, Km)
    add!(assembly.Kg, gdofs, gdofs, Kg)
    add!(assembly.f, gdofs, f)
end

typealias Elasticity2DSurfaceElements Union{Poi1, Seg2, Seg3}
typealias Elasticity2DVolumeElements Union{Tri3, Tri6, Quad4, Quad8, Quad9}
typealias Elasticity3DSurfaceElements Union{Poi1, Tri3, Tri6, Quad4, Quad8, Quad9}
typealias Elasticity3DVolumeElements Union{Tet4, Tet10, Hex8, Hex20, Hex27}


""" Elasticity equations for 2d cases. """
function assemble{El<:Elasticity2DVolumeElements}(problem::Problem{Elasticity}, element::Element{El}, time, ::Type{Val{:plane}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    BL = zeros(3, dim*nnodes)
    BNL = zeros(4, dim*nnodes)
    Km = zeros(dim*nnodes, dim*nnodes)
    Kg = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)

        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        dN = element(ip, time, Val{:Grad})

        # kinematics

        gradu = element("displacement", ip, time, Val{:Grad})
        fill!(BL, 0.0)

        if props.finite_strain
            strain = 1/2*(gradu + gradu' + gradu'*gradu)
            F = eye(dim) + gradu
            for i=1:size(dN, 2)
                BL[1, 2*(i-1)+1] += F[1,1]*dN[1,i]
                BL[1, 2*(i-1)+2] += F[2,1]*dN[1,i]
                BL[2, 2*(i-1)+1] += F[1,2]*dN[2,i]
                BL[2, 2*(i-1)+2] += F[2,2]*dN[2,i]
                BL[3, 2*(i-1)+1] += F[1,1]*dN[2,i] + F[1,2]*dN[1,i]
                BL[3, 2*(i-1)+2] += F[2,1]*dN[2,i] + F[2,2]*dN[1,i]
            end
        else # linearized strain
            strain = 1/2*(gradu + gradu')
            F = eye(dim)
            for i=1:size(dN, 2)
                BL[1, 2*(i-1)+1] = dN[1,i]
                BL[2, 2*(i-1)+2] = dN[2,i]
                BL[3, 2*(i-1)+1] = dN[2,i]
                BL[3, 2*(i-1)+2] = dN[1,i]
            end
        end

        strain_vec = [strain[1,1]; strain[2,2]; strain[1,2]]

        # calculate stress
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
        # calculate stress
        stress_vec = D * ([1.0, 1.0, 2.0] .* strain_vec)

        "strain" in props.store_fields && update!(ip, "strain", time => strain_vec)
        "stress" in props.store_fields && update!(ip, "stress", time => stress_vec)
        "stress 11" in props.store_fields && update!(ip, "stress 11", time => stress_vec[1])
        "stress 22" in props.store_fields && update!(ip, "stress 22", time => stress_vec[2])
        "stress 12" in props.store_fields && update!(ip, "stress 12", time => stress_vec[3])

        Km += w*BL'*D*BL

        # stress = [stress_vec[1] stress_vec[3]; stress_vec[3] stress_vec[2]]
        # cauchy_stress = F'*stress*F/det(F)
        # cauchy_stress = [cauchy_stress[1,1]; cauchy_stress[2,2]; cauchy_stress[1,2]]
        # update!(ip, "cauchy stress", time => cauchy_stress)
        
        # material stiffness end

        if props.geometric_stiffness
            # take geometric stiffness into account

            fill!(BNL, 0.0)
            for i=1:size(dN, 2)
                BNL[1, 2*(i-1)+1] = dN[1,i]
                BNL[2, 2*(i-1)+1] = dN[2,i]
                BNL[3, 2*(i-1)+2] = dN[1,i]
                BNL[4, 2*(i-1)+2] = dN[2,i]
            end

            S2 = zeros(2*dim, 2*dim)
            S2[1,1] = stress_vec[1]
            S2[2,2] = stress_vec[2]
            S2[1,2] = S2[2,1] = stress_vec[3]
            S2[3:4,3:4] = S2[1:2,1:2]
            
            Kg += w*BNL'*S2*BNL # geometric stiffness

        end

        # rhs, internal and external load
        
        f -= w*BL'*stress_vec

        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            f += w*vec(N'*b)
        end

        for i=1:dim
            if haskey(element, "displacement load $i")
                b = element("displacement load $i", ip, time)
                f[i:dim:end] += w*vec(b*N)
            end
        end

    end

    return Km, Kg, f
end

function assemble{El<:Elasticity2DSurfaceElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:plane}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    Km = zeros(dim*nnodes, dim*nnodes)
    Kg = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)

        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)

        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            f += w*vec(T*N)
        end

        for i=1:dim
            # traction force for ith component
            if haskey(element, "displacement traction force $i")
                T = element("displacement traction force $i", ip, time)
                f[i:dim:end] += w*vec(T*N)
            end
        end

        if haskey(element, "nt displacement traction force")
            # traction force given in normal-tangential direction
            T = element("nt displacement traction force", ip, time)
            Q = element("normal-tangential coordinates", ip, time)
            f += w*vec(Q'*T*N)
        end

    end

    return Km, Kg, f
end

""" Elasticity equations, 3d, linear. """
function assemble{El<:Elasticity3DVolumeElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum_linear}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    ndofs = dim*nnodes
    BL = zeros(6, ndofs)
    Km = zeros(ndofs, ndofs)
    Kg = zeros(ndofs, ndofs)
    f = zeros(ndofs)

    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        dN = element(ip, time, Val{:Grad})

        fill!(BL, 0.0)
        for i=1:nnodes
            BL[1, 3*(i-1)+1] = dN[1,i]
            BL[2, 3*(i-1)+2] = dN[2,i]
            BL[3, 3*(i-1)+3] = dN[3,i]
            BL[4, 3*(i-1)+1] = dN[2,i]
            BL[4, 3*(i-1)+2] = dN[1,i]
            BL[5, 3*(i-1)+2] = dN[3,i]
            BL[5, 3*(i-1)+3] = dN[2,i]
            BL[6, 3*(i-1)+1] = dN[3,i]
            BL[6, 3*(i-1)+3] = dN[1,i]
        end

        E = element("youngs modulus", ip, time)
        nu = element("poissons ratio", ip, time)

        D = E/((1.0+nu)*(1.0-2.0*nu)) * [
            1.0-nu nu nu 0.0 0.0 0.0
            nu 1.0-nu nu 0.0 0.0 0.0
            nu nu 1.0-nu 0.0 0.0 0.0
            0.0 0.0 0.0 0.5-nu 0.0 0.0
            0.0 0.0 0.0 0.0 0.5-nu 0.0
            0.0 0.0 0.0 0.0 0.0 0.5-nu]

        Km += w*BL'*D*BL

        if haskey(element, "displacement load")
            T = element("displacement load", ip, time)
            f += w*vec(T*N)
        end
        for i=1:dim
            if haskey(element, "displacement load $i")
                b = element("displacement load $i", ip, time)
                f[i:dim:end] += w*vec(b*N)
            end
        end

    end

    if get_formulation_type(problem) == :incremental
        if haskey(element, "displacement")
            u = vec(element["displacement"](time))
            f -= Kt*u
        end
    end

    return Km, Kg, f
end

""" Material and geometric stiffness for linear buckling analysis. """
function assemble{El<:Elasticity3DVolumeElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum_buckling}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    ndofs = dim*nnodes
    BL = zeros(6, ndofs)
    BNL = zeros(9, ndofs)
    Km = zeros(ndofs, ndofs)
    Kg = zeros(ndofs, ndofs)
    f = zeros(ndofs)

    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        dN = element(ip, time, Val{:Grad})

        gradu = element("displacement", ip, time, Val{:Grad})
        strain = 1/2*(gradu' + gradu)

        fill!(BL, 0.0)
        for i=1:nnodes
            BL[1, 3*(i-1)+1] = dN[1,i]
            BL[2, 3*(i-1)+2] = dN[2,i]
            BL[3, 3*(i-1)+3] = dN[3,i]
            BL[4, 3*(i-1)+1] = dN[2,i]
            BL[4, 3*(i-1)+2] = dN[1,i]
            BL[5, 3*(i-1)+2] = dN[3,i]
            BL[5, 3*(i-1)+3] = dN[2,i]
            BL[6, 3*(i-1)+1] = dN[3,i]
            BL[6, 3*(i-1)+3] = dN[1,i]
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

        E = element("youngs modulus", ip, time)
        nu = element("poissons ratio", ip, time)
        D = E/((1.0+nu)*(1.0-2.0*nu)) * [
            1.0-nu nu nu 0.0 0.0 0.0
            nu 1.0-nu nu 0.0 0.0 0.0
            nu nu 1.0-nu 0.0 0.0 0.0
            0.0 0.0 0.0 0.5-nu 0.0 0.0
            0.0 0.0 0.0 0.0 0.5-nu 0.0
            0.0 0.0 0.0 0.0 0.0 0.5-nu]

        strain_vec = [strain[1,1]; strain[2,2]; strain[3,3]; strain[1,2]; strain[2,3]; strain[1,3]]
        stress_vec = D * ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0].*strain_vec)

        S3 = zeros(3*dim, 3*dim)
        S3[1,1] = stress_vec[1]
        S3[2,2] = stress_vec[2]
        S3[3,3] = stress_vec[3]
        S3[1,2] = S3[2,1] = stress_vec[4]
        S3[2,3] = S3[3,2] = stress_vec[5]
        S3[1,3] = S3[3,1] = stress_vec[6]
        S3[4:6,4:6] = S3[7:9,7:9] = S3[1:3,1:3]

        Km += w*BL'*D*BL
        Kg += w*BNL'*S3*BNL

    end

    return Km, Kg, f
end

""" Elasticity equations, 3d nonlinear. """
function assemble{El<:Elasticity3DVolumeElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    ndofs = dim*nnodes
    BL = zeros(6, ndofs)
    BNL = zeros(9, ndofs)
    Km = zeros(ndofs, ndofs)
    Kg = zeros(ndofs, ndofs)
    f = zeros(ndofs)

    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        dN = element(ip, time, Val{:Grad})

        # kinematics; calculate deformation gradient and strain

        gradu = zeros(dim, dim)
        if haskey(element, "displacement")
            gradu += element("displacement", ip, time, Val{:Grad})
        end
        strain = 1/2*(gradu' + gradu)

        F = eye(dim)
        if props.finite_strain
            F += gradu
            strain += 1/2*gradu'*gradu
        end

        # material stiffness start

        fill!(BL, 0.0)
        for i=1:nnodes
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

        strain_vec = [strain[1,1]; strain[2,2]; strain[3,3]; strain[1,2]; strain[2,3]; strain[1,3]]

        # calculate stress
        E = element("youngs modulus", ip, time)
        nu = element("poissons ratio", ip, time)
        D = E/((1.0+nu)*(1.0-2.0*nu)) * [
            1.0-nu nu nu 0.0 0.0 0.0
            nu 1.0-nu nu 0.0 0.0 0.0
            nu nu 1.0-nu 0.0 0.0 0.0
            0.0 0.0 0.0 0.5-nu 0.0 0.0
            0.0 0.0 0.0 0.0 0.5-nu 0.0
            0.0 0.0 0.0 0.0 0.0 0.5-nu]
        stress_vec = D * ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0].*strain_vec)

        "strain" in props.store_fields && update!(ip, "strain", time => strain_vec)
        "stress" in props.store_fields && update!(ip, "stress", time => stress_vec)
        "stress 11" in props.store_fields && update!(ip, "stress 11", time => stress_vec[1])
        "stress 22" in props.store_fields && update!(ip, "stress 22", time => stress_vec[2])
        "stress 33" in props.store_fields && update!(ip, "stress 33", time => stress_vec[3])
        "stress 12" in props.store_fields && update!(ip, "stress 12", time => stress_vec[4])
        "stress 23" in props.store_fields && update!(ip, "stress 23", time => stress_vec[5])
        "stress 13" in props.store_fields && update!(ip, "stress 13", time => stress_vec[6])

        Km += w*BL'*D*BL

        # material stiffness end

        if props.geometric_stiffness
            # take geometric stiffness into account

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
            S3[1,1] = stress_vec[1]
            S3[2,2] = stress_vec[2]
            S3[3,3] = stress_vec[3]
            S3[1,2] = S3[2,1] = stress_vec[4]
            S3[2,3] = S3[3,2] = stress_vec[5]
            S3[1,3] = S3[3,1] = stress_vec[6]
            S3[4:6,4:6] = S3[7:9,7:9] = S3[1:3,1:3]

            Kg += w*BNL'*S3*BNL

        end

        # external load start

        if haskey(element, "displacement load")
            T = element("displacement load", ip, time)
            f += w*vec(T*N)
        end

        for i=1:dim
            if haskey(element, "displacement load $i")
                b = element("displacement load $i", ip, time)
                f[i:dim:end] += w*vec(b*N)
            end
        end

        # external load end

        if get_formulation_type(problem) == :incremental
            f -= w*BL'*stress_vec
        end

    end

    return Km, Kg, f
end

""" Elasticity equations, surface traction for continuum formulation. """
function assemble{El<:Elasticity3DSurfaceElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum}})

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = size(element, 2)
    Km = zeros(dim*nnodes, dim*nnodes)
    Kg = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)

    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            f += w*vec(T*N)
        end
        for i in 1:dim
            if haskey(element, "displacement traction force $i")
                T = element("displacement traction force $i", ip, time)
                f[i:dim:end] += w*vec(T*N)
            end
        end
        if haskey(element, "displacement traction force n")
            J = element(ip, time, Val{:Jacobian})'
            n = cross(J[:,1], J[:,2])
            n /= norm(n)
            p = element("displacement traction force n", ip, time)
            f += w*p*vec(n*N)
        end
    end
    return Km, Kg, f
end

function assemble{El<:Elasticity3DSurfaceElements}(problem::Problem{Elasticity}, element::Element{El}, time::Real, ::Type{Val{:continuum_linear}})
    return assemble(problem, element, time, Val{:continuum})
end

""" Elasticity equations using ForwardDiff
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
    Km, allresults = ForwardDiff.jacobian(get_residual_vector, vec(field),
                     AllResults, cache=autodiffcache)
    Kg = zeros(Km)
    f = -ForwardDiff.value(allresults)
    return Km, Kg, f
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
