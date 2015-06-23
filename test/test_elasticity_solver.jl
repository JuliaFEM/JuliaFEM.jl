using FactCheck
using Logging
@Logging.configure(level=DEBUG)


using JuliaFEM.elasticity_solver: solve_elasticity_increment!
facts("test solve elasticity increment") do

  X = [0 0; 10 0; 10 1; 0 1]'
  elmap = [1; 2; 3; 4]
  nodalloads = [0 0; 0 0; 0 -2; 0 0]'
  @debug("nodal loads:\n", nodalloads)
  dirichletbc = [0 0; NaN NaN; NaN NaN; 0 0]'

  E = 90
  nu = 0.25
  mu = E/(2*(1+nu))
  la = E*nu/((1+nu)*(1-2*nu))
  la = 2*la*mu/(la + 2*mu)

  la = la*ones(1, 4)
  mu = mu*ones(1, 4)
  u = zeros(2, 4)
  du = zeros(2, 4)
  R = zeros(2,4)
  Kt = zeros(8,8)

  N(xi) = [
      (1-xi[1])*(1-xi[2])/4
      (1+xi[1])*(1-xi[2])/4
      (1+xi[1])*(1+xi[2])/4
      (1-xi[1])*(1+xi[2])/4
    ]

  dNdξ(ξ) = [-(1-ξ[2])/4.0    -(1-ξ[1])/4.0
              (1-ξ[2])/4.0    -(1+ξ[1])/4.0
              (1+ξ[2])/4.0     (1+ξ[1])/4.0
             -(1+ξ[2])/4.0     (1-ξ[1])/4.0]

  ipoints = 1/sqrt(3)*[-1 -1; 1 -1; 1 1; -1 1]
  iweights = [1 1 1 1]

  for i=1:10
    solve_elasticity_increment!(X, u, du, R, Kt,elmap, nodalloads, dirichletbc,
                                la, mu, N, dNdξ, ipoints, iweights)
    @debug("increment:\n",du)
    u += du
    if norm(du) < 1.0e-9
      break
    end
  end
  @debug("solution\n",u)
  @fact u[2, 3] => roughly(-2.222244754401764)  # Tested against Elmer solution
end


using JuliaFEM.elasticity_solver: assemble!

facts("test assembly of global matrix for 1 dim/node case") do
    I = Int64[]
    J = Int64[]
    V = Float64[]
    ke = [3 1; 1 1]
    eldofs = [1, 2]
    assemble!(ke, eldofs, I, J, V)
    ke = [4 2; 2 3]
    eldofs = [2, 4]
    assemble!(ke, eldofs, I, J, V)
    S = full(sparse(I, J, V))
    @fact S => [3.0 1.0 0.0 0.0
                1.0 5.0 0.0 2.0
                0.0 0.0 0.0 0.0
                0.0 2.0 0.0 3.0]
end

facts("test assembly of global vector for 1 dim/node case") do
    I = Int64[]
    V = Float64[]
    fe = [1;2]
    eldofs = [1, 2]
    assemble!(fe, eldofs, I, V)
    fe = [3;1]
    eldofs = [2, 4]
    assemble!(fe, eldofs, I, V)
    S = full(sparsevec(I, V))
    @fact S => [1.0 5.0 0.0 1.0]'
end

facts("test assembly of global matrix for 2 dim/node case") do
    # provide "convienence" function, if given only nodal connectivity
    # automatically find out dimension and "extend" matrix to full
    I = Int64[]
    J = Int64[]
    V = Float64[]
    ke = reshape(1:16, 4, 4)
    eldofs = [1, 2]
    assemble!(ke, eldofs, I, J, V)
    eldofs = [2, 3]
    assemble!(2*ke, eldofs, I, J, V)
    expected = zeros(6, 6)
    expected[1:4,1:4] += ke
    expected[3:6,3:6] += 2*ke
    S = full(sparse(I, J, V))
    @fact S => expected
end

facts("test assembly of global vector for 2 dim/node case") do
    # provide "convienence" function, if given only nodal connectivity
    # automatically find out dimension and "extend" matrix to full
    I = Int64[]
    J = Int64[]
    V = Float64[]
    fe = [1, 2, 3, 4]
    eldofs = [1, 2]
    assemble!(fe, eldofs, I, V)
    eldofs = [2, 3]
    assemble!(2*fe, eldofs, I, V)
    S = full(sparsevec(I, V))
    expected = [1.0 2.0 5.0 8.0 6.0 8.0]'
    @fact S => expected
end
