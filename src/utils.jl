# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function calculate_rotated_nodal_vector(field_name, field_dim, elements::Vector{Element},
                                        time, vec_dim=0)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    for element in elements
        haskey(element, field_name) || continue
        gdofs = get_gdofs(element, 1)
        for ip in get_integration_points(element, Val{5})
            J = get_jacobian(element, ip, time)
            w = ip.weight*norm(J)
            Q = element("normal-tangential coordinates", ip, time)
            f = element(field_name, ip, time)
            f = Q'*f
            N = element(ip, time)
            add!(A, gdofs, gdofs, w*kron(N', N))
            for dim=1:field_dim
                add!(b, gdofs, w*f[dim]*N, dim)
            end
        end
    end
    A = sparse(A)
    b = sparse(b)
    nz = sort(unique(rowvals(A)))
    x = zeros(size(b)...)
    x[nz, :] = A[nz,nz] \ b[nz, :]
    if vec_dim != 0
        v = zeros(vec_dim)
        v[1:length(x)] = x
        return v
    else
        return x
    end
end

""" Collect normal-tangential coordinates to rotation matrix Q.
"""
function get_rotation_matrix(elements, time)
    Q = Dict{Int64, Matrix{Float64}}()
    ndim = 0
    for element in elements
        node_ids = get_connectivity(element)
        q = element("normal-tangential coordinates", time).data
        ndim == 0 && (ndim = size(q, 1))
        ndim != size(q, 1) && error("2d and 3d rotation matrices in one element set?")
        for (qi, node_id) in zip(q, node_ids)
            if haskey(Q, node_id)
                @assert isapprox(Q[node_id], qi)
            else
                Q[node_id] = qi
            end
        end
    end
    R = SparseMatrixCOO()
    for (k, q) in Q
        dofs = Int[ndim*(k-1)+j for j=1:ndim]
        add!(R, dofs, dofs, q)
    end
    return R
end
