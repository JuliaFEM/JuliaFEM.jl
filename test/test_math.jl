# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module MathTests
using JuliaFEM.Test

function test_math()
    @test 1+1 == 2
end

#using JuliaFEM: interpolate
#=
facts("test interpolation of different field variables") do
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
    F1 = [36.0, 36.0, 36.0, 36.0]
    F2 = [36.0 36.0 36.0 36.0]
    F3 = F2'
    F4 = [0.0 0.0; 10.0 0.0; 10.0 1.0; 0.0 1.0]'
    F5 = F4'
    F6 = [36, 36, 36, 36]

    @fact interpolate(F1, N, [0.0, 0.0]) --> 36.0
    @fact interpolate(F2, N, [0.0, 0.0]) --> 36.0
    @fact interpolate(F3, N, [0.0, 0.0]) --> 36.0
    @fact interpolate(F4, N, [0.0, 0.0]) --> [5.0; 0.5]
    @fact interpolate(F5, N, [0.0, 0.0]) --> [5.0; 0.5]
    @fact interpolate(F5, dNdξ, [0.0, 0.0]) --> [5.0 0.0; 0.0 0.5]
    @fact interpolate(F6, N, [0.0, 0.0]) --> 36
end
=#

end
