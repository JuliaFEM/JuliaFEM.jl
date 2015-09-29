# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM: Basis, Field, get_field, diff
using FactCheck

facts("test fields and interpolation") do

    # simple interpolation in domain [-1, 1]
    N = Basis((ξ) -> [0.5*(1.0-ξ[1]), 0.5*(1.0+ξ[1])])
    u = Field(0.0, [0.0, 1.0])
    @fact N([0.0])*u --> 0.5
    @fact (N*u)([0.0]) --> 0.5

    # multiply of field with constant
    u1 = Field(0.0, [0.0, 1.0])
    u2 = 3.0*u1
    @fact u1.time --> 0.0
    @fact u2.time --> 0.0
    @fact u2.values --> [0.0, 3.0]

    # addition of fields together
    u1 = Field(0.0, [0.0, 1.0])
    u2 = Field(0.0, [1.0, 2.0])
    u3 = u1 + u2
    @fact u3.values --> [1.0, 3.0]

    # interpolation between two fields in time domain
    u1 = Field(0.0, [0.0, 1.0])
    u2 = Field(0.0, [1.0, 2.0])
    t = Basis((t) -> [1-t, t])
    u = Field[u1, u2]
    u2 = (t*u)(0.5)
    @fact u2.values --> [0.5, 1.5]

    # interpolation in set of fields is defined for every time value
    u1 = Field(0.0, [0.0, 0.0])
    u2 = Field(1.0, [1.0, 2.0])
    u3 = Field(2.0, [0.5, 1.5])
    u = Field[u1, u2, u3]
    @fact u(-1.0).values --> [0.0, 0.0]  # "out of range -" -> first known value
    @fact u(0.0).values --> [0.0, 0.0]
    @fact u(1.0).values --> [1.0, 2.0]
    @fact u(2.0).values --> [0.5, 1.5]
    @fact u(3.0).values --> [0.5, 1.5]  # "out of range +" -> last known value
    @fact u(0.5).values --> [0.5, 1.0]
    @fact u(1.5).values --> [0.75, 1.75]
    # use Inf to get very first or last value of field
    @fact u(-Inf).values --> [0.0, 0.0]
    @fact u(+Inf).values --> [0.75, 1.75]

    # multidimensional interpolation with and without derivatives
    X = Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    h = Basis((xi) ->
    [(1-xi[1])*(1-xi[2])/4
     (1+xi[1])*(1-xi[2])/4
     (1+xi[1])*(1+xi[2])/4
     (1-xi[1])*(1+xi[2])/4])
   # midpoint of field
   @fact (h*X)([0.0, 0.0]) --> [0.5, 0.5]
   @fact h([0.0, 0.0])*X --> [0.5, 0.5]
   # derivatives of field at midpoint
   @fact diff(h)([0.0, 0.0])*X --> [0.5 0.0; 0.0 0.5]
   @fact (diff(h)*X)([0.0, 0.0]) --> [0.5 0.0; 0.0 0.5]

   # multiplying scalar field with a vector -> vector
   b = Basis((xi) -> [1/2*(1-xi[1]), 1/2*(1+xi[1])])
   f = Field(0.0, 100.0)
   @fact b(0.0) * f --> [50.0, 50.0]

end
