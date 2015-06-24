# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module elasticity_solver

using Logging
@Logging.configure(level=DEBUG)

VERSION < v"0.4-" && using Docile

# Below this line is internal functions related to solver. They can be used
# directly if needed or using general interface combining data model and
# solver.


@doc """
Interpolate field variable using basis functions f for point ip.
This function tries to be as general as possible and allows interpolating
lot of different fields.

Parameters
----------
field :: Array{Number, dim}
  Field variable
basis :: Function
  Basis functions
ip :: Array{Number, 1}
  Point to interpolate
""" ->
function interpolate(field::Array{Float64,1}, basis::Function, ip)
    result = dot(field, basis(ip))
    return result
end
function interpolate(field::Array{Float64,2}, basis::Function, ip)
    m, n = size(field)
    bip = basis(ip)
    tmp = size(bip)
    if length(tmp) == 1
        ndim = 1
        nnodes = tmp[1]
    else
        ndim, nnodes = size(bip)
    end
    if ndim == 1
        if n == nnodes
            result = field * bip
        elseif m == nnodes
            result = field' * bip
        end
    else
      if n == nnodes
        result = bip' * field
    elseif m == nnodes
      result = bip' * field'
    end
    end
    if length(result) == 1
        result = result[1]
    end
    return result
end



@doc """
Calculate local tangent stiffness matrix and residual force vector R = T - F
""" ->
function calc_local_matrices!(X, u, R, Kt, N, dNdξ, λ_, μ_, ipoints, iweights)
  dim, nnodes = size(X)
  I = eye(dim)
  R[:,:] = 0.0
  Kt[:,:] = 0.0

  dF = zeros(dim, dim)

  for m = 1:length(iweights)
    w = iweights[m]
    ξ = ipoints[m, :]
    # interpolate material parameters from element node fields
    #λ = (λ_*N(ξ))[1]
    #μ = (μ_*N(ξ))[1]
    #    Jᵀ = X*dNdξ(ξ)
    #@debug("Jt:\n",Jᵀ)
    λ = interpolate(λ_, N, ξ)
    μ = interpolate(μ_, N, ξ)
    Jᵀ = interpolate(X, dNdξ, ξ)'
        detJ = det(Jᵀ)
        ∇N = inv(Jᵀ)*dNdξ(ξ)'
        ∇u = u*∇N'
        F = I + ∇u  # Deformation gradient
        E = 1/2*(∇u' + ∇u + ∇u'*∇u)  # Green-Lagrange strain tensor
        S = λ*trace(E)*I + 2*μ*E  # PK2 stress tensor
        P = F*S  # PK1 stress tensor
        R[:,:] += w*P*∇N*detJ

        for p = 1:nnodes
            for i = 1:dim
                dF[:,:] = 0.0
                dF[i,:] = ∇N[:,p]
                dE = 1/2*(F'*dF + dF'*F)
                dS = λ*trace(dE)*I + 2*μ*dE
                dP = dF*S + F*dS
                for q = 1:nnodes
                    for j = 1:dim
                        Kt[dim*(p-1)+i,dim*(q-1)+j] += w*(dP[j,:]*∇N[:,q])[1]*detJ
                    end
                end
            end
        end

    end
end


@doc """
Assemble global stiffness matrix to I,J,V ready for sparse format

Parameters
----------
ke : local matrix
eldofs_ : Array
  degrees of freedom
I,J,V : Arrays for sparse matrix

Notes
-----
eldofs can also be node ids for convenience. In that case dimension
is calculated and eldofs are "extended" to problem dimension.
""" ->
function assemble!(ke, eldofs_, I, J, V)
    n, m = size(ke)
    dim = round(Int, n/length(eldofs_))
    @debug("problem dim = ", dim)
    if dim == 1
        eldofs = eldofs_
    else
        eldofs = Int64[]
        for i in eldofs_
            for d in 1:dim
                push!(eldofs, dim*(i-1)+d)
            end
        end
        @debug("old eldofs", eldofs_)
        @debug("new eldofs", eldofs)
    end
    for i in 1:n
        for j in 1:m
            push!(I, eldofs[i])
            push!(J, eldofs[j])
            push!(V, ke[i,j])
        end
    end
end


@doc """
Assemble global RHS to I,V ready for sparse format

Parameters
----------
fe : local vector
eldofs_ : Array
  degrees of freedom
I,V : Arrays for sparse matrix

Notes
-----
eldofs can also be node ids for convenience. In that case dimension
is calculated and eldofs are "extended" to problem dimension.
""" ->
function assemble!(fe, eldofs_, I, V)
    n = length(fe)
    dim = round(Int, n/length(eldofs_))
    if dim == 1
        eldofs = eldofs_
    else
        eldofs = Int64[]
        for i in eldofs_
            for d in 1:dim
                push!(eldofs, dim*(i-1)+d)
            end
        end
        @debug("old eldofs", eldofs_)
        @debug("new eldofs", eldofs)
    end

    for i in 1:n
        push!(I, eldofs[i])
        push!(V, fe[i])
    end
end


@doc """
Eliminate Dirichlet boundary conditions from matrix

Parameters
----------
dirichletbc : array [dim x nnodes]
I, J, V : sparse matrix arrays

Returns
-------
I, J, V : boundary conditions removed

Notes
-----
pros:
- matrix assembly remains positive definite
cons:
- maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
- if u != 0 in dirichlet boundary requires extra care

Raises
------
Exception, if displacement boundary conditions given, i.e.
DX=2 for some node, for example.

"""
function eliminate_boundary_conditions(dirichletbc, I, J, V)
    if any(dirichletbc .> 0)
        throw("displacement boundary condition not supported")
    end
    # dofs to remove
    free_dofs = find(isnan(dirichletbc))
    remove_dofs = find(!isnan(dirichletbc))
    @debug("Removing dofs: ", remove_dofs)
    # this can be done more clever by removing corresponging indexes from I, J, and V
    A = sparse(I, J, V)
    A = A[free_dofs, free_dofs]
    return findnz(A)
end

@doc """
Eliminate Dirichlet boundary conditions from vector

Parameters
----------
dirichletbc : array [dim x nnodes]
I, V : sparse vector arrays

Returns
-------
I, V : boundary conditions removed

Notes
-----
pros:
- matrix assembly remains positive definite
cons:
- maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
- if u != 0 in dirichlet boundary requires extra care

Raises
------
Exception, if displacement boundary conditions given, i.e.
DX=2 for some node, for example.
""" ->
function eliminate_boundary_conditions(dirichletbc, I, V)
    if any(dirichletbc .> 0)
        throw("displacement boundary condition not supported")
    end
    # dofs to remove
    free_dofs = find(isnan(dirichletbc))
    remove_dofs = find(!isnan(dirichletbc))
    @debug("Removing dofs: ", remove_dofs)
    # this can be done more clever by removing corresponging indexes from I, J, and V
    A = sparsevec(I, V)
    A = A[free_dofs]
    @debug("new vector: ", A)
    i, j, v = findnz(A)
    return i, v
end



@doc """
Solve one increment of elasticity problem
""" ->
function solve_elasticity_increment!(X, u, du, elmap, nodalloads,
                                     dirichletbc, λ, μ, N, dNdξ, ipoints,
                                     iweights)
    if length(size(elmap)) == 1
        # quick hack for just one element
        elmap = elmap''
    end
    nelnodes, nelements = size(elmap)
    #@debug("elements in model: ", nelements)
    #@debug("nodes / element in model: ", nelnodes)


    dim = size(u)[1]
    Imat = Int64[]
    Jmat = Int64[]
    Vmat = Float64[]
    Ivec = Int64[]
    Vvec = Float64[]
    # FIXME: different number of elements / node
    #@debug("problem dimension: ", dim)

    dofs = dim*nelnodes
    R = zeros(dim, nelnodes)
    Kt = zeros(dofs, dofs)
    #@debug("size of R: ", size(R))
    #@debug("size of Kt: ", size(Kt))

    # this can be parallelized
    for i in 1:nelements
        eldofs = elmap[:,i]
        #@debug("element dofs ", eldofs)
        #@debug("Assembling element ", i)
        #@debug("Local coords:\n", X[:, eldofs])
        #@debug("Local u:\n", u[:, eldofs])
        calc_local_matrices!(X[:, eldofs], u[:, eldofs], R, Kt, N, dNdξ, λ, μ, ipoints, iweights)
        #@debug("Assemble Kt")
        assemble!(Kt, eldofs, Imat, Jmat, Vmat)
        #@debug("Assemble R")
        assemble!(R, eldofs, Ivec, Vvec)
    end

    # add additional neumann boundary conditions
    for (i, nodal_load) in enumerate(nodalloads)
        if nodal_load == 0
            continue
        end
        push!(Ivec, i)
        push!(Vvec, -nodal_load)
    end

    # Remove dirichlet boundary conditions
    Imat, Jmat, Vmat = eliminate_boundary_conditions(dirichletbc, Imat, Jmat, Vmat)
    Ivec, Vvec = eliminate_boundary_conditions(dirichletbc, Ivec, Vvec)
  #R -= nodalloads

    # Create sparse matrices
    A = sparse(Imat, Jmat, Vmat)
    b = sparsevec(Ivec, Vvec)

    # solution
    free_dofs = find(isnan(dirichletbc))
    #du[free_dofs] = Kt[free_dofs, free_dofs] \ -reshape(R, 8)[free_dofs]
    du[free_dofs] = lufact(A) \ -full(b)
end

end
