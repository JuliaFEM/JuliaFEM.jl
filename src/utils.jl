# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Calculate nodal vector from set of elements.

For example element 1 with dofs [1, 2, 3, 4] has [1, 1, 1, 1] and
element 2 with dofs [3, 4, 5, 6] has [2, 2, 2, 2] the result will
be sparse matrix with values [1, 1, 3, 3, 2, 2].

Parameters
----------
field_name
    name of field, e.g. "geometry"
field_dim
    degrees of freedom / node
elements
    elements used to calculate vector
vec_dim
    used to resize solution vector if given
time
"""
function calculate_nodal_vector(field_name, field_dim, elements::Vector{Element},
                                time, vec_dim=0)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    for element in elements
        haskey(element, field_name) || continue
        gdofs = get_gdofs(element, 1)
        for ip in get_integration_points(element, Val{3})
            J = get_jacobian(element, ip, time)
            w = ip.weight*norm(J)
            f = element(field_name, ip, time)
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
    x = vec(transpose(x))
    if vec_dim != 0
        v = zeros(vec_dim)
        v[1:length(x)] = x
        return v
    else
        return x
    end
end

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
