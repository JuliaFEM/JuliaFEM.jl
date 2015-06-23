using JuliaFEM.elasticity_solver

using FactCheck
using Logging
@Logging.configure(level=DEBUG)


facts("test solve elasticity increment") do
  X = [0 0; 10 0; 10 1; 0 1]'
  elmap = [1; 2; 3; 4]
  nodalloads = [0 0; 0 0; 0 -2; 0 0]'
  @debug("nodal loads:\n", nodalloads)
  dirichletbc = [0 0; NaN NaN; NaN NaN; 0 0]'

  E = 90
  ν = 0.25
  μ = E/(2*(1+ν))
  λ = E*ν/((1+ν)*(1-2*ν))
  λ = 2*λ*μ/(λ + 2*μ)

  #E = 90.0*ones(2, 4)
  #nu = 0.25*ones(2, 4)
  u = zeros(2, 4)
  du = zeros(2, 4)
  R = zeros(2,4)
  Kt = zeros(8,8)

  dNdξ(ξ) = [-(1-ξ[2])/4.0    -(1-ξ[1])/4.0
              (1-ξ[2])/4.0    -(1+ξ[1])/4.0
              (1+ξ[2])/4.0     (1+ξ[1])/4.0
             -(1+ξ[2])/4.0     (1-ξ[1])/4.0]

  ipoints = 1/sqrt(3)*[-1 -1; 1 -1; 1 1; -1 1]
  iweights = [1 1 1 1]

  for i=1:10
    JuliaFEM.elasticity_solver.solve_elasticity_increment!(X, u, du, R, Kt, elmap, nodalloads,
                                dirichletbc, λ, μ, dNdξ, ipoints,
                                iweights)
    @debug("increment:\n",du)
    u += du
    if norm(du) < 1.0e-9
      break
    end
  end
  @debug("solution\n",u)
  @fact u[2, 3] => roughly(-2.222244754401764)  # Tested against Elmer solution
end
