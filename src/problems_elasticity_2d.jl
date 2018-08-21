# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

const Elasticity2DSurfaceElements = Union{Poi1,Seg2,Seg3}
const Elasticity2DVolumeElements = Union{Tri3,Tri6,Quad4,Quad8,Quad9}

function assemble!(assembly::Assembly, problem::Problem{Elasticity},
                   elements::Union{Vector{Element}, Vector{Element{T}}},
                   time, ::Type{Val{:plane_stress}}) where T
    for element in elements
        gdofs = get_gdofs(problem, element)
        Km, Kg, f = assemble(problem, element, time, Val{:plane})
        add!(assembly.K, gdofs, gdofs, Km)
        add!(assembly.Kg, gdofs, gdofs, Kg)
        add!(assembly.f, gdofs, f)
    end
end

function assemble!(assembly::Assembly, problem::Problem{Elasticity},
                   elements::Union{Vector{Element}, Vector{Element{T}}},
                   time, ::Type{Val{:plane_strain}}) where T
    for element in elements
        gdofs = get_gdofs(problem, element)
        Km, Kg, f = assemble(problem, element, time, Val{:plane})
        add!(assembly.K, gdofs, gdofs, Km)
        add!(assembly.Kg, gdofs, gdofs, Kg)
        add!(assembly.f, gdofs, f)
    end
end

""" Plane elasticity equations (plane stress, plane strain). """
function assemble(problem::Problem{Elasticity},
                  element::Element{El}, time,
                  ::Type{Val{:plane}}) where El<:Elasticity2DVolumeElements

    props = problem.properties
    dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    BL = zeros(3, dim*nnodes)
    BNL = zeros(4, dim*nnodes)
    Km = zeros(dim*nnodes, dim*nnodes)
    Kg = zeros(dim*nnodes, dim*nnodes)
    f = zeros(dim*nnodes)
    Dtan = zeros(3,3)

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
            F = I + gradu
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
            F = I
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
            D = E/((1.0+nu)*(1.0-2.0*nu)) .* [
                1.0-nu      nu               0.0
                    nu  1.0-nu               0.0
                   0.0     0.0  (1.0-2.0*nu)/2.0]
        else
            error("unknown plane formulation: $(props.formulation)")
        end

        # calculate stress

        if haskey(element, "plasticity")
            plastic_def = element("plasticity")[ip.id]

            calculate_stress! = plastic_def["type"]
            yield_surface_ = plastic_def["yield_surface"]
            params = plastic_def["params"]

            initialize_internal_params!(params, ip, Val{:type_2d})

            if time == 0.0
                error("Given step time = $(time). Please select time > 0.0")
            end

            t_last = ip("prev_time", time)
            update!(ip, "prev_time", time => t_last)
            dt = time - t_last

            stress_last = ip("stress", t_last)
            strain_last = ip("strain", t_last)

            dstrain_vec = strain_vec - strain_last
            stress_vec = [0.0, 0.0, 0.0]
            pstrain = zeros(3)
            calculate_stress!(stress_vec, stress_last, dstrain_vec, pstrain, D, params, Dtan, yield_surface_, time, dt, Val{:type_2d})
        else
            stress_vec = D * ([1.0, 1.0, 2.0] .* strain_vec)
            Dtan[:,:] = D[:,:]
        end

        :strain in props.store_fields && update!(ip, "strain", time => strain_vec)
        :stress in props.store_fields && update!(ip, "stress", time => stress_vec)
        :stress11 in props.store_fields && update!(ip, "stress11", time => stress_vec[1])
        :stress22 in props.store_fields && update!(ip, "stress22", time => stress_vec[2])
        :stress12 in props.store_fields && update!(ip, "stress12", time => stress_vec[3])

        Km += w*BL'*Dtan*BL


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
            f += w*vec(b*N)
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

function assemble(problem::Problem{Elasticity},
                  element::Element{El},
                  time, ::Type{Val{:plane}}) where El<:Elasticity2DSurfaceElements

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
