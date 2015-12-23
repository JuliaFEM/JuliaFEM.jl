# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract DirichletProblem{T} <: AbstractProblem
abstract StandardBasis
abstract BiorthogonalBasis

function DirichletProblem(parent_field_name::ASCIIString, parent_field_dim::Int, dim::Int=1, elements=Element[]; basis=StandardBasis)
    return BoundaryProblem{DirichletProblem{basis}}("dirichlet boundary", parent_field_name, parent_field_dim, dim, elements)
end
function DirichletProblem(problem_name::ASCIIString, parent_field_name::ASCIIString, parent_field_dim::Int, dim::Int=1, elements=Element[]; basis=StandardBasis)
    return BoundaryProblem{DirichletProblem{basis}}(problem_name, parent_field_name, parent_field_dim, dim, elements)
end

function assemble!(assembly::BoundaryAssembly, problem::BoundaryProblem{DirichletProblem{StandardBasis}}, element::Element, time::Real)

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
            # add all dimensions at once if defined element["blaa"] = 0.0
            for i=1:field_dim
                g = element(field_name, ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*N')
            end
        end
  
        for i=1:field_dim
            if haskey(element, field_name*" $i")
                g = element(field_name*" $i", ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*N')
            end
        end
    end
end

function assemble!(assembly::BoundaryAssembly, problem::BoundaryProblem{DirichletProblem{BiorthogonalBasis}}, element::Element, time::Real)

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
        A[abs(A) .< 1.0e-9] = 0

        # C1 matrix is always the same
        for i=1:field_dim
            ldofs = gdofs[i:field_dim:end]
            add!(assembly.C1, ldofs, ldofs, A)
        end

        if haskey(element, field_name)
            # add all dimensions at once if defined element["blaa"] = 0.0
            for i=1:field_dim
                g = element(field_name, ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*Phi')
            end
        else
            for i=1:field_dim
                ldofs = gdofs[i:field_dim:end]
                if haskey(element, field_name*" $i")
                    g = element(field_name*" $i", ip, time)
                    add!(assembly.C2, ldofs, ldofs, A)
                    add!(assembly.g, ldofs, w*g*Phi')
                else
                    add!(assembly.D, ldofs, ldofs, A)
                end
            end
        end
    end
end
