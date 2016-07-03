# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM

"""
Calculate field values to nodal points from Gauss points using least-squares fitting.
"""
function calc_nodal_values!(elements::Vector, field_name, field_dim, time;
                            F=nothing, nz=nothing, b=nothing, return_F_and_nz=false)

    if F == nothing
        A = SparseMatrixCOO()
        for element in elements
            gdofs = get_connectivity(element)
            for ip in get_integration_points(element)
                detJ = element(ip, time, Val{:detJ})
                w = ip.weight*detJ
                N = element(ip, time)
                add!(A, gdofs, gdofs, w*kron(N', N))
            end
        end
        nz = get_nonzero_rows(A)
        A = sparse(A)
        A = 1/2*(A + A')
        F = ldltfact(A[nz,nz])
    end

    if b == nothing
        b = SparseMatrixCOO()
        for element in elements
            gdofs = get_connectivity(element)
            for ip in get_integration_points(element)
                if !haskey(ip, field_name)
                    info("warning: integration point does not have field $field_name")
                    continue
                end
                detJ = element(ip, time, Val{:detJ})
                w = ip.weight*detJ
                f = ip(field_name, time)
                N = element(ip, time)
                for dim=1:field_dim
                    add!(b, gdofs, w*f[dim]*N, dim)
                end
            end
        end
        b = sparse(b)
    end

    x = zeros(size(b)...)
    x[nz, :] = F \ b[nz, :]
    nodal_values = Dict()
    for i=1:size(x,1)
        nodal_values[i] = vec(x[i,:])
    end
    update!(elements, field_name, time => nodal_values)
    if return_F_and_nz
        return F, nz
    end
end

function calc_nodal_values!(problem::Problem, field_name, field_dim, time)
    # after all, it's just a mass matrix ...
#   isempty(problem.assembly.M) && assemble!(problem, time, Val{:mass_matrix}; density=1.0, dual_basis=false, dim=1)
#   M = sparse(problem.assembly.M)
    # TODO: make test before implementation
    calc_nodal_values!(problem.elements, field_name, field_dim, time)
end

"""
Return node ids + vector of values 
"""
function get_nodal_vector(elements, field_name, time)
    f = Dict()
    for element in elements
        for (c, v) in zip(get_connectivity(element), element[field_name](time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    node_ids = sort(collect(keys(f)))
    field = [f[nid] for nid in node_ids]
    return node_ids, field
end

""" Return nodal values in Dict format. """
function get_nodal_dict(T::DataType, elements, field_name, time)
    f = T()
    for element in elements
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    return f
end

""" Update nodal field values from set of elements to another. Can be used to
transform e.g. reaction force from boundary element set to surface of
volume elements for easier postprocess.
"""
function copy_field!(src_elements::Vector, dst_elements::Vector, field_name, time)
    dst_nodes = Set{Int64}()
    for element in dst_elements
        push!(dst_nodes, get_connectivity(element)...)
    end
    node_ids, field = get_nodal_vector(src_elements, field_name, time)
    z = 0.0*first(field)
    d = Dict()
    for j in dst_nodes
        d[j] = z
    end
    for (j, f) in zip(node_ids, field)
        d[j] = f
    end
    for element in dst_elements
        c = get_connectivity(element)
        f = [d[j] for j in c]
        update!(element, field_name, time => f)
    end
end

function copy_field!(src_problem::Problem, dst_problem::Problem, field_name, time)
    copy_field!(src_problem.elements, dst_problem.elements, field_name, time)
end

""" Return field calculated to nodal points for elements in problem p. """
function call(problem::Problem, field_name, time=0.0)
    f = Dict()
    for element in get_elements(problem)
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    return f
end

