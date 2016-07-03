# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test
importall Base
import JuliaFEM: get_basis, get_dbasis

type TestElement <: AbstractElement
end

function get_basis(element::Element{TestElement}, xi, time)
    1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
end

function get_dbasis(element::Element{TestElement}, xi, time)
    1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]
end

function length(element::Element{TestElement})
    return 4
end

function size(element::Element{TestElement})
    return (2, 4)
end

function get_element()
    element = Element(TestElement, [1, 2, 3, 4])
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    T = Dict{Int64, Float64}(
        1 => 1.0,
        2 => 2.0,
        3 => 3.0,
        4 => 4.0)
    u1 = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [1/4, 0.0],
        4 => [0.0, 0.0])
    u2 = Dict{Int64, Vector{Float64}}(
        1 => [0.0,  0.0],
        2 => [1.0, -1.0],
        3 => [2.0,  3.0],
        4 => [0.0,  0.0])
    update!(element, "geometry", X)
    update!(element, "temperature", T)
    update!(element, "displacement1", u1)
    update!(element, "displacement2", u2)
    return element
end

@testset "spatial interpolation in basis" begin
    element = get_element()
    @test isapprox(element([0.0, 0.0], 0.0), 1/4*[1 1 1 1])
    @test isapprox(element([0.0, 0.0], 1.0), 1/4*[1 1 1 1])
end

@testset "gradient of shape functions" begin
    element = get_element()
    grad = element([0.0, 0.0], 0.0, Val{:Grad})
    @test isapprox(grad, 1/2*[-1 1 1 -1; -1 -1 1 1])
end

@testset "interpolation of scalar field in spatial domain" begin
    # in unit square: T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    element = get_element()
    T_known(X) = 1 + X[1] + 3*X[2] - 2*X[1]*X[2]
    T_interpolated = element("temperature", [0.0, 0.0], 0.0)
    @test isapprox(T_interpolated, T_known([0.5, 0.5]))
end

@testset "interpolation of gradient of scalar field in spatial domain" begin
    # in unit square: grad(T)(X) = [1-2X[2], 3-2*X[1]]
    element = get_element()
    gradT = element("temperature", [0.0, 0.0], 0.0, Val{:Grad})
    gradT_expected(X) = [1-2*X[2] 3-2*X[1]]
    @test isapprox(gradT, gradT_expected([0.5, 0.5]))
end

@testset "test interpolation of vector field" begin
    # in unit square, u(X,t) = [1/4*t*X[1]*X[2], 0, 0]
    element = get_element()
    u = element("displacement1", [0.0, 0.0], 0.0)
#   x = X+u
    u_expected(X) = [1/4*X[1]*X[2], 0]
#   @test isapprox(x, [9/16, 1/2])
    @test isapprox(u, u_expected([0.5, 0.5]))
end

@testset "interpolation of gradient of vector_field" begin
    # in unit square, u(X) = t*[X[1]*(X[2]+1), X[1]*(4*X[2]-1)]
    # => u_i,j = t*[X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    element = get_element()
#    displacement = Field(
#        (0.5, Vector[[0.0, 0.0], [0.5, -0.5], [1.0, 1.5], [0.0, 0.0]]),
#        (1.5, Vector[[0.0, 0.0], [1.5, -1.5], [3.0, 4.5], [0.0, 0.0]]))
    gradu = element("displacement2", [0.0, 0.0], 0.0, Val{:Grad})
    gradu_expected(X) = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    @test isapprox(gradu, gradu_expected([0.5, 0.5]))
end

#= TODO: Fix test
@testset "linear time extrapolation of field" begin
    #T_known(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    T = DVTV()
    update!(T, 0.0 => [0.0, 0.0, 0.0, 0.0])
    update!(T, 1.0 => [1.0, 2.0, 3.0, 4.0])
    @test T(-1.0) == -1.0*[1.0, 2.0, 3.0, 4.0]
    @test T( 3.0) ==  3.0*[1.0, 2.0, 3.0, 4.0]
    # when going to \pm infinity, return the last one.
    @test T(-Inf) ==  0.0*[1.0, 2.0, 3.0, 4.0]
    @test T(+Inf) ==  1.0*[1.0, 2.0, 3.0, 4.0]
end
=#

#= TODO: Fix test
@testset "constant time extrapolation of field" begin
    #T_known(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    T = DVTV()
    update!(T, 0.0 => [0.0, 0.0, 0.0, 0.0])
    update!(T, 1.0 => [1.0, 2.0, 3.0, 4.0])
    @test isapprox(T(-1.0, Val{:constant}), [0.0, 0.0, 0.0, 0.0])
    @test isapprox(T( 3.0, Val{:constant}), [1.0, 2.0, 3.0, 4.0])
end
=#

#= TODO: Fix test
@testset "time extrapolation of field with only one timestep" begin
    T = DVTV()
    update!(T, 0.0 => [1.0, 2.0, 3.0, 4.0])
    @test isapprox(T(1.0), [1.0, 2.0, 3.0, 4.0])
end
=#

#= TODO: Fix test
@testset "interpolation in temporal direction" begin
    field = DCTV()
    update!(field, 0.0 => 0.0)
    update!(field, 2.0 => 1.0)
    update!(field, 4.0 => 2.0)
    @test isapprox(field(-Inf), 0.0)
    @test isapprox(field( 0.0), 0.0)
    @test isapprox(field( 1.0), 0.5)
    @test isapprox(field( 2.0), 1.0)
    @test isapprox(field( 3.0), 1.5)
    @test isapprox(field( 4.0), 2.0)
    @test isapprox(field(+Inf), 2.0)
end
=#

#= TODO: Fix test
@testset "time derivative interpolation in temporal basis in constant velocity" begin
    field = DCTV()
    update!(field, 0.0 => 0.0)
    update!(field, 2.0 => 1.0)
    update!(field, 4.0 => 2.0)
    @test isapprox(field(+Inf, Val{:diff}), 0.5)
    @test isapprox(field(-Inf, Val{:diff}), 0.5)
    @test isapprox(field( 0.0, Val{:diff}), 0.5)
    @test isapprox(field( 0.5, Val{:diff}), 0.5)
    @test isapprox(field( 1.0, Val{:diff}), 0.5)
    @test isapprox(field( 1.5, Val{:diff}), 0.5)
    @test isapprox(field( 2.0, Val{:diff}), 0.5)
end
=#

#= TODO: Fix test
@testset "time derivative interpolation in temporal basis in variable velocity" begin
    pos = DCTV()
    for ti in linspace(0, 2, 5)
        update!(pos, ti => 1/2*ti^2)
    end
    # => ((0.0,0.0),(0.5,0.125),(1.0,0.5),(1.5,1.125),(2.0,2.0))
    velocity = pos(1.0, Val{:diff})
    v1 = (0.500 - 0.125)/0.5
    v2 = (1.125 - 0.500)/0.5
    @test isapprox(velocity, mean([v1, v2])) # = 1.00
    velocity = pos(2.0, Val{:diff})
    @test isapprox(velocity, (2.0-1.125)/0.5) # = 1.75
end
=#

function test_time_derivative_gradient_interpolation_of_field()
    # in unit square, u(X) = t*[X[1]*(X[2]+1), X[1]*(4*X[2]-1)]
    # => u_i,j = t*[X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    # => d(u_i,j)/dt = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    u1 = Dict{Int64, Vector{Float64}}(
        1 => [0.0,  0.0],
        2 => [0.5, -0.5],
        3 => [1.0,  1.5],
        4 => [0.0,  0.0])
    u2 = Dict{Int64, Vector{Float64}}(
        1 => [0.0,  0.0],
        2 => [1.5, -1.5],
        3 => [3.0,  4.5],
        4 => [0.0,  0.0])
    element = Element(TestElement, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "displacement", 0.5 => u1)
    update!(element, "displacement", 1.5 => u2)

    xi = [0.0, 0.0]
    time = 1.2
    diffgradu = element("displacement", xi, time, Val{:diff}, Val{:Grad})
    diffgradu_expected(X, t) = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    @test diffgradu == diffgradu_expected([0.5, 0.5], 1.2)
end

@testset "some continuum mechanics interpolations" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [1.0, 1.0],
        4 => [0.0, 1.0])
    u1 = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [0.0, 0.0],
        4 => [0.0, 0.0])
    u2 = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [1/4, 0.0],
        4 => [0.0, 0.0])
    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "displacement", 0.0 => u1)
    update!(element, "displacement", 1.0 => u2)

    # from my old home works
    X = element("geometry", [0.0, 0.0], 1.0)
    u = element("displacement", [0.0, 0.0], 1.0)
    x = X + u
    x_expected = [9/16, 1/2]
    gradu = element("displacement", [0.0, 0.0], 1.0, Val{:Grad})
    epsilon = 1/2*(gradu + gradu')
    rotation = 1/2*(gradu - gradu')
    k = 0.25
    epsilon_expected = [
        X[2]*k      1/2*X[1]*k
        1/2*X[1]*k  0]
    rotation_expected = [
        0          k/2*X[1]
        -k/2*X[1]         0]
    F = I + gradu
    F_expected = [
        X[2]*k+1 X[1]*k
        0        1]
    C = F'*F
    C_expected = [
        (X[2]*k+1)^2       (X[2]*k+1)*X[1]*k
        (X[2]*k+1)*X[1]*k  X[1]^2*k^2+1]
    E = 1/2*(F'*F - I)
    E_expected = [
        1/2*(X[2]*k + 1)^2-1/2   1/2*(X[2]*k+1)*X[1]*k
        1/2*(X[2]*k + 1)*X[1]*k  1/2*X[1]^2*k^2]
    U = 1/sqrt(trace(C) + 2*sqrt(det(C)))*(C + sqrt(det(C))*I)
#   U_expected = [1.24235 0.13804; 0.13804 1.02149]

    @test isapprox(x, x_expected)
    @test isapprox(epsilon, epsilon_expected)
    @test isapprox(rotation, rotation_expected)
    @test isapprox(F, F_expected)
    @test isapprox(C, C_expected)
    @test isapprox(E, E_expected)
    # TODO: Fix test
#   @test isapprox(U, U_expected)
end



