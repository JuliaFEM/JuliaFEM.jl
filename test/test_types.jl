# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

#= TODO: Fix test

facts("test interpolation of fields") do

    # interpolation of field in spatial domain
    N = Basis((xi) -> [0.5*(1.0-xi[1]), 0.5*(1.0+xi[1])])
    u = Field(0.0, [0.0, 1.0])
    @fact interpolate(N, u, [0.0]) --> 0.5

    # interpolation of fieldset in time domain
    u1 = Field(0.0, [0.0, 1.0])
    u2 = Field(1.0, [1.0, 2.0])
    u = FieldSet("displacement", [u1, u2])
    @fact interpolate(u, 0.5).values --> [0.5, 1.5]
    @fact interpolate(u, 0.5).time --> 0.5

    # interpolation of fieldset is defined for every time value:
    u1 = Field(0.0, [0.0, 0.0])
    u2 = Field(1.0, [1.0, 2.0])
    u3 = Field(2.0, [0.5, 1.5])
    u = FieldSet("displacement", [u1, u2, u3])
    @fact interpolate(u, -1.0).values --> [0.0, 0.0]  # "out of range -" -> first known value
    @fact interpolate(u, 0.0).values --> [0.0, 0.0]
    @fact interpolate(u, 1.0).values --> [1.0, 2.0]
    @fact interpolate(u, 2.0).values --> [0.5, 1.5]
    @fact interpolate(u, 3.0).values --> [0.5, 1.5]  # "out of range +" -> last known value
    @fact interpolate(u, 0.5).values --> [0.5, 1.0]
    @fact interpolate(u, 1.5).values --> [0.75, 1.75]
    # use Inf to get very first or last value of field
    @fact interpolate(u, -Inf).values --> [0.0, 0.0]
    @fact interpolate(u, +Inf).values --> [0.5, 1.5]

    # multidimensional interpolation with and without derivatives
    h(xi) = [
        (1-xi[1])*(1-xi[2])/4
        (1+xi[1])*(1-xi[2])/4
        (1+xi[1])*(1+xi[2])/4
        (1-xi[1])*(1+xi[2])/4]
    dh(xi) = [
        -(1-xi[2])/4.0   -(1-xi[1])/4.0
         (1-xi[2])/4.0   -(1+xi[1])/4.0
         (1+xi[2])/4.0    (1+xi[1])/4.0
        -(1+xi[2])/4.0    (1-xi[1])/4.0]
    N = Basis(h, dh)

    X = Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # get midpoint of field in spatial domain
    @fact interpolate(N, X, [0.0, 0.0]) --> [0.5, 0.5]

    # interpolate of scalar field -> scalar
    H = Field(0.0, 6.0)
    @fact interpolate(N, H, [0.0, 0.0]) --> 6.0

    # multiplying scalar field with a vector -> vector
    # this is actually not so good idea...
    #h(xi) = [1/2*(1-xi[1]), 1/2*(1+xi[1])]
    #dh(xi) = [-1/2 1/2]'
    #N = Basis(h, dh)
    #f = Field(0.0, 100.0)
    #@fact interpolate(N, f, [0.0]) --> [50.0, 50.0]

end

=#
