# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Linear elasticity

abstract LinearElasticityProblem <: ElasticityProblem

function LinearElasticityProblem(dim::Int=3, elements=[])
    return Problem{LinearElasticityProblem}(dim, elements)
end

""" Elasticity equations, general 3D case. """
function assemble!{E<:CG, P<:LinearElasticityProblem}(assembly::Assembly, problem::Problem{P}, element::Element{E}, time::Real)

    gdofs = get_gdofs(element, problem.dim)
    ndim, nnodes = size(E)
    B = zeros(6, 3*nnodes)
    for ip in get_integration_points(element)
        w = ip.weight*det(element, ip, time)
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
            add!(assembly.stiffness_matrix, gdofs, gdofs, w*B'*C*B)
        end
        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*b)
        end
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            L = w*T*N
#           dump(L)
            add!(assembly.force_vector, gdofs, vec(L))
        end
    end
end

abstract PlaneStressLinearElasticityProblem <: LinearElasticityProblem

function PlaneStressLinearElasticityProblem(dim::Int=2, elements=[])
    return Problem{PlaneStressLinearElasticityProblem}(dim, elements)
end

""" Elasticity equations, plane stress. """
function assemble!{E<:CG, P<:PlaneStressLinearElasticityProblem}(assembly::Assembly, problem::Problem{P}, element::Element{E}, time::Real)

    gdofs = get_gdofs(element, problem.dim)
    ndim, nnodes = size(E)
    B = zeros(3, 2*nnodes)
    for ip in get_integration_points(element)
        w = ip.weight*det(element, ip, time)
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
            add!(assembly.stiffness_matrix, gdofs, gdofs, w*B'*C*B)
        end
        if haskey(element, "displacement load")
            b = element("displacement load", ip, time)
            add!(assembly.force_vector, gdofs, w*N'*b)
        end
        if haskey(element, "displacement traction force")
            T = element("displacement traction force", ip, time)
            L = w*T*N
#           dump(L)
            add!(assembly.force_vector, gdofs, vec(L))
        end
    end
end
