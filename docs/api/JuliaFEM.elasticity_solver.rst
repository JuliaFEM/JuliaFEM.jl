JuliaFEM.elasticity_solver
==========================

Internal
--------

 .. function:: assemble!(fe,  eldofs_,  I,  V)

    Assemble global RHS to I,V ready for sparse format

    :param fe :  local vector
    :param eldofs_ :  Array
    :param I,V :  Arrays for sparse matrix
    :notes: eldofs can also be node ids for convenience. In that case dimension
            is calculated and eldofs are "extended" to problem dimension.
**source**
[JuliaFEM/src/elasticity_solver.jl:174]

 .. function:: assemble!(ke,  eldofs_,  I,  J,  V)

    Assemble global stiffness matrix to I,J,V ready for sparse format

    :param ke :  local matrix
    :param eldofs_ :  Array
    :param I,J,V :  Arrays for sparse matrix
    :notes: eldofs can also be node ids for convenience. In that case dimension
            is calculated and eldofs are "extended" to problem dimension.
**source**
[JuliaFEM/src/elasticity_solver.jl:133]

 .. function:: calc_local_matrices!(X,  u,  R,  K,  basis,  dbasis,  lambda_,  mu_,  ipoints,  iweights)

    Calculate local tangent stiffness matrix and residual force vector
    R = T - F for elasticity problem.

    :param X :  Element coordinates
    :param u :  Displacement field
    :param R :  Residual force vector
    :param K :  Tangent stiffness matrix
    :param basis :  Basis functions
    :param dbasis :  Derivative of basis functions
    :param lambda :  Material parameter
    :param mu :  Material parameter
    :param ipoints :  integration points
    :param iweights :  integration weights
    :returns: None
    :notes: If material parameters are given in list, they are interpolated to gauss
            points using shape functions.
**source**
[JuliaFEM/src/elasticity_solver.jl:76]

 .. function:: dummy(a)

    This is dummy function. Testing doctests and documentation.

    :param x :  Array{Float64, 1}
    :returns: Array{float64, 1}
                x + 1
    :notes: This is dummy function
    :raises: Exception
               if things are not going right
**source**
[JuliaFEM/src/elasticity_solver.jl:44]

 .. function:: eliminate_boundary_conditions(dirichletbc,  I,  J,  V)

    Eliminate Dirichlet boundary conditions from matrix

    :param dirichletbc :  array [dim x nnodes]
    :param I, J, V :  sparse matrix arrays
    :returns: I, J, V : boundary conditions removed
    :notes: pros:
            - matrix assembly remains positive definite
            cons:
            - maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
            - if u != 0 in dirichlet boundary requires extra care
    :raises: Exception, if displacement boundary conditions given, i.e.
             DX=2 for some node, for example.
**source**
[JuliaFEM/src/elasticity_solver.jl:221]

 .. function:: eliminate_boundary_conditions(dirichletbc,  I,  V)

    Eliminate Dirichlet boundary conditions from vector

    :param dirichletbc :  array [dim x nnodes]
    :param I, V :  sparse vector arrays
    :returns: I, V : boundary conditions removed
    :notes: pros:
            - matrix assembly remains positive definite
            cons:
            - maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
            - if u != 0 in dirichlet boundary requires extra care
    :raises: Exception, if displacement boundary conditions given, i.e.
             DX=2 for some node, for example.
**source**
[JuliaFEM/src/elasticity_solver.jl:260]

 .. function:: solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights)

**source**
[JuliaFEM/src/elasticity_solver.jl:281]

