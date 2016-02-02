# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract DirichletProblem{T} <: AbstractProblem
abstract StandardBasis
abstract DualBasis
global const BiorthogonalBasis = DualBasis

type Dirichlet <: AbstractProblem
    dual_basis :: Bool
end

function Dirichlet()
    Dirichlet(true)
end

function assemble!(assembly::BoundaryAssembly, problem::BoundaryProblem{Dirichlet}, element::Element, time::Real)

    @assert problem.properties.dual_basis

    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name
    gdofs = get_gdofs(element, field_dim)

    # calculate bi-orthogonal basis transformation matrix Ae
    nnodes = size(element, 2)
    De = zeros(nnodes, nnodes)
    Me = zeros(nnodes, nnodes)
    for ip in get_integration_points(element, Val{2})
        w = ip.weight
        J = get_jacobian(element, ip, time)
        JT = transpose(J)
        if size(JT, 2) == 1  # plane problem
            w *= norm(JT)
        else
            w *= norm(cross(JT[:,1], JT[:,2]))
        end
        N = element(ip, time)
        De += w*diagm(vec(N))
        Me += w*N'*N
    end
    Ae = De*inv(Me)

    # do the actual integration
    for ip in get_integration_points(element, Val{2})
        w = ip.weight
        J = get_jacobian(element, ip, time)
        JT = transpose(J)
        if size(JT, 2) == 1  # plane problem
            w *= norm(JT)
        else
            w *= norm(cross(JT[:,1], JT[:,2]))
        end
        N = element(ip, time)
        Phi = (Ae*N')'
        A = w*Phi'*N
        A[abs(A) .< 1.0e-12] = 0

        if haskey(element, field_name)
            for i=1:field_dim
                g = element(field_name, ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*Phi')
            end
        else
            for i=1:field_dim
                ldofs = gdofs[i:field_dim:end]
                if haskey(element, field_name*" $i")
                    g = element(field_name*" $i", ip, time)
                    add!(assembly.C1, ldofs, ldofs, A)
                    add!(assembly.C2, ldofs, ldofs, A)
                    add!(assembly.g, ldofs, w*g*Phi')
                end
            end
        end
    end
end


function assemble!(assembly::BoundaryAssembly, problem::BoundaryProblem{DirichletProblem}, element::Element, time::Real)

    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name

    gdofs = get_gdofs(element, field_dim)
    for ip in get_integration_points(element, Val{2})
        w = ip.weight
        J = get_jacobian(element, ip, time)
        JT = transpose(J)
        if size(JT, 2) == 1  # plane problem
            w *= norm(JT)
        else
            w *= norm(cross(JT[:,1], JT[:,2]))
        end
        N = element(ip, time)
        A = w*N'*N

        if haskey(element, field_name)
            # add all dimensions at once if defined
            # element["blaa"] = 0.0
            # or
            # element["blaa"] = Vector{Float64}[[0.1, 0.2], [0.3, 0.4]]
            g = element(field_name, ip, time)
            if length(g) != length(N)
                g = g*ones(length(N))
            end
            for i=1:field_dim
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
            end
            add!(assembly.g, gdofs, w*g*N)
        end

        for i=1:field_dim
            if haskey(element, field_name*" $i")
                g = element(field_name*" $i", ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*N)
            end
        end
    end
end

