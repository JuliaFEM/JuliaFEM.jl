JuliaFEM.elasticity_solver
==========================

Internal
--------

.. jl:function:: assemble!(fe,  eldofs_,  I,  V)

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
    

.. jl:function:: assemble!(ke,  eldofs_,  I,  J,  V)

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
    

.. jl:function:: calc_local_matrices!(X,  u,  R,  Kt,  N,  dNdchi,  lambda_,  mu_,  ipoints,  iweights)

    Calculate local tangent stiffness matrix and residual force vector R = T - F
    

.. jl:function:: eliminate_boundary_conditions(dirichletbc,  I,  J,  V)

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
    
    

.. jl:function:: eliminate_boundary_conditions(dirichletbc,  I,  V)

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
    

.. jl:function:: interpolate{T<:Real}(field::Array{T<:Real, 1},  basis::Function,  ip)

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
    

.. jl:function:: solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights)

    Solve one increment of elasticity problem
    

