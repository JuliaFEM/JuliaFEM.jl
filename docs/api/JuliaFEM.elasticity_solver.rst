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
source:
-------
[JuliaFEM/src/elasticity_solver.jl:171](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L171)

 .. function:: assemble!(ke,  eldofs_,  I,  J,  V)

    Assemble global stiffness matrix to I,J,V ready for sparse format

    :param ke :  local matrix
    :param eldofs_ :  Array
    :param I,J,V :  Arrays for sparse matrix
    :notes: eldofs can also be node ids for convenience. In that case dimension
            is calculated and eldofs are "extended" to problem dimension.
source:
-------
[JuliaFEM/src/elasticity_solver.jl:130](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L130)

 .. function:: calc_local_matrices!(X,  u,  R,  Kt,  N,  dNdchi,  lambda_,  mu_,  ipoints,  iweights)

source:
-------
[JuliaFEM/src/elasticity_solver.jl:68](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L68)

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
source:
-------
[JuliaFEM/src/elasticity_solver.jl:218](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L218)

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
source:
-------
[JuliaFEM/src/elasticity_solver.jl:257](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L257)

 .. function:: interpolate{T<:Real}(field::Array{T<:Real, 1},  basis::Function,  ip)

    Interpolate field variable using basis functions f for point ip.
    This function tries to be as general as possible and allows interpolating
    lot of different fields.

    :param field :  Array{Number, dim}
    :param basis :  Function
    :param ip :  Array{Number, 1}
source:
-------
[JuliaFEM/src/elasticity_solver.jl:30](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L30)

 .. function:: solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights)

source:
-------
[JuliaFEM/src/elasticity_solver.jl:278](https://github.com/JuliaFEM/JuliaFEM.jl/tree/08077a783a6459336cf07b02e323df0e9977c74f/src/elasticity_solver.jl#L278)

