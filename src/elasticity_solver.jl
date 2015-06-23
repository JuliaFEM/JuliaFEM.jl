# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module elasticity_solver

VERSION < v"0.4-" && using Docile

export solve_elasticity_interface!

@doc """
This is generic interface that reads data from data model, solves elasticity
problem and updates model.

Parameters
----------
model : to be defined
""" ->
function solve_elasticity_interface!()
  return 0
end



# Below this line is internal functions related to solver. They can be used
# directly if needed or using general interface combining data model and
# solver.

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
    λ = (λ_*N(ξ))[1]
    μ = (μ_*N(ξ))[1]
        Jᵀ = X*dNdξ(ξ)
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
Solve one increment of elasticity problem
""" ->
function solve_elasticity_increment!(X, u, du, R, Kt, elmap, nodalloads,
                                     dirichletbc, λ, μ, N, dNdξ, ipoints,
                                     iweights)
  calc_local_matrices!(X, u, R, Kt, N, dNdξ, λ, μ, ipoints, iweights)
  # FIXME: boundary conditions
  free_dofs = find(isnan(dirichletbc))
  R -= nodalloads
  du[free_dofs] = Kt[free_dofs, free_dofs] \ -reshape(R, 8)[free_dofs]
end


end
