# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Calculate field values to nodal points from Gauss points using least-squares fitting.
"""
function calc_nodal_values!(elements, field_name, field_dim, time)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    for element in elements
        gdofs = get_connectivity(element)
        for ip in get_integration_points(element)
            detJ = element(ip, time, Val{:detJ})
            w = ip.weight*detJ
            f = ip(field_name, time)
            N = element(ip, time)
            add!(A, gdofs, gdofs, w*kron(N', N))
            for dim=1:field_dim
                add!(b, gdofs, w*f[dim]*N, dim)
            end
        end
    end
    A = sparse(A)
    b = sparse(b)
    nz = get_nonzero_rows(A)
    x = zeros(size(b)...)
    x[nz, :] = A[nz,nz] \ b[nz, :]
    nodal_values = Dict()
    for i=1:size(x,1)
        nodal_values[i] = vec(x[i,:])
    end
    update!(elements, field_name, nodal_values)
end
