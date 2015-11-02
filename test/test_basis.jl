# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module BasisTests

using JuliaFEM.Test

using JuliaFEM
using JuliaFEM: Basis, ElementGradientBasis, ElementFieldGradientBasis, Field
using JuliaFEM: Increment, TimeStep

function get_basis()

    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
    
    dbasis(xi) = 1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]

    return basis, dbasis
end


function test_basic_interpolation()
    basis, dbasis = get_basis()
    b = Basis(basis, dbasis)
    @test b([0.0, 0.0]) == 1/4*[1 1 1 1]
    @test b([0.0, 0.0], 1.0) == 1/4*[1 1 1 1]
end

function test_basic_interpolation_of_field()
    # in unit square: T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    temperature = Field(
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 2.0, 3.0, 4.0]))
    basis, dbasis = get_basis()
    b = Basis(basis, dbasis, temperature)
    T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    @test b([0.0, 0.0], 0.0) == T([0.5, 0.5], 0.0)
    @test b([0.0, 0.0], 0.6) == T([0.5, 0.5], 0.6)
    @test b([0.0, 0.0], 1.0) == T([0.5, 0.5], 1.0) 
end

function test_linear_time_extrapolation_of_field()
    temperature = Field(
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 2.0, 3.0, 4.0]))
    basis, dbasis = get_basis()
    b = Basis(basis, dbasis, temperature, :linear)
    T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    @test b([0.0, 0.0], -1.0) == T([0.5, 0.5], -1.0)
    @test b([0.0, 0.0],  3.0) == T([0.5, 0.5],  3.0)
    # when going to \pm infinity, return the last one.
    @test b([0.0, 0.0], -Inf) == T([0.5, 0.5],  0.0)
    @test b([0.0, 0.0], +Inf) == T([0.5, 0.5],  1.0)
end

function test_constant_time_extrapolation_of_field()
    temperature = Field(
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 2.0, 3.0, 4.0]))
    basis, dbasis = get_basis()
    b = Basis(basis, dbasis, temperature, :constant)
    T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    @test b([0.0, 0.0], -1.0) == T([0.5, 0.5], 0.0)
    @test b([0.0, 0.0],  3.0) == T([0.5, 0.5], 1.0)
end

function test_time_extrapolation_of_field_with_single_timestep()
    temperature = Field([1.0, 2.0, 3.0, 4.0])
    basis, dbasis = get_basis()
    b = Basis(basis, dbasis, temperature)
    @test b([0.0, 0.0], 1.0) == mean([1.0, 2.0, 3.0, 4.0])
end

function test_gradient_interpolation_empty_gradient()
    X = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]'
    geometry = Field(X)
#   P(X) = [1.0, X[1], X[2], X[1]*X[2]]
#   basis2, dbasis2 = JuliaFEM.calculate_lagrange_basis(P, X)
    basis, dbasis = get_basis()
    N = Basis(basis, dbasis)
    dN = ElementGradientBasis(N, geometry)
    @test dN([0.0, 0.0]) == 1/2*[-1 1 1 -1; -1 -1 1 1]
#   @test dN([0.0, 0.0]) == dbasis2([0.5, 0.5])
end

function test_gradient_interpolation_of_scalar_field()
    # in unit square: grad(T)(X) = [1-2X[2], 3-2*X[1]]
    geometry = Field([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')

    temperature = Field([1, 2, 3, 4])
    basis, dbasis = get_basis()
    N = Basis(basis, dbasis)

    dN = ElementGradientBasis(N, geometry)
    dT = ElementFieldGradientBasis(dN, temperature)
    dT_expected(X) = [1-2*X[2] 3-2*X[1]]
 
    @test dT([0.0, 0.0]) == dT_expected([0.5, 0.5])
end

function test_interpolation_of_vector_field()
    # in unit square, u(X,t) = [1/4*t*X[1]*X[2], 0, 0]
    geometry = Field([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    displacement = Field(
        (0.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        (1.0, Vector[[0.0, 0.0], [0.0, 0.0], [1/4, 0.0], [0.0, 0.0]]))

    basis, dbasis = get_basis()
    X = Basis(basis, dbasis, geometry)
    u = Basis(basis, dbasis, displacement)
    u_expected(X,t) = [1/4*t*X[1]*X[2], 0]
    # x = X + u
    x = X([0.0, 0.0], 1.0) + u([0.0, 0.0], 1.0)
    @test isapprox(x, [9/16, 1/2])
    @test isapprox(u([0.0, 0.0], 1.0), u_expected([0.5, 0.5], 1.0))
end

function test_interpolation_of_gradient_of_vector_field()
    # in unit square, u(X) = t*[X[1]*X[2]/4, X[1]*(X[1]+X[2])/2]
    # => u_i,j = t*[X[2]/4 X[1]/4; X[1]/2+(X[1]+X[2])/2 X[1]/2]
    geometry = Field([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    displacement = Field(
        (0.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.00, 0.0], [0.0, 0.0]]),
        (1.0, Vector[[0.0, 0.0], [0.0, 0.5], [0.25, 1.0], [0.0, 0.0]]))

    basis, dbasis = get_basis()
    N = Basis(basis, dbasis)
    dN = ElementGradientBasis(N, geometry)
    dU = ElementFieldGradientBasis(dN, displacement)
    dU_expected(X, t) = t*[X[2]/4 X[1]/4; X[1]/2+(X[1]+X[2])/2 X[1]/2]

    @test isapprox(dU([0.0, 0.0], 1.0), dU_expected([0.5, 0.5], 1.0))
end

# TODO: how on earth make this work without some serious spaghetti code
function test_time_derivative_gradient_interpolation_of_field()
    # in unit square, u(X) = t*[X[1]*X[2]/4, X[1]*(X[1]+X[2])/2]
    # => u_i,j = t*[X[2]/4 X[1]/4; X[1]/2+(X[1]+X[2])/2 X[1]/2]
    # => d(u_i,j)/dt = [X[2]/4 X[1]/4; X[1]/2+(X[1]+X[2])/2 X[1]/2]
    geometry = Field([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    displacement = Field(
        (0.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.00, 0.0], [0.0, 0.0]]),
        (1.0, Vector[[0.0, 0.0], [0.0, 0.5], [0.25, 1.0], [0.0, 0.0]]))

    # wanted
    #u = get_basis(element, "displacement")
    #L = grad(diff(u))
    #D = 1/2*(L + L')
    #@test isapprox(D([0.0, 0.0], 1.0), ...)

    basis, dbasis = get_basis()
    N = Basis(basis, dbasis)
    xi = [0.0, 0.0]
    time = 1.0
    grad = ElementGradientBasis(N, geometry)(xi, time)
    increment = displacement(time, Val{:derivative}, :linear, :linear)
    diffgradu = sum([grad[:,i]*increment[i]' for i=1:length(increment)])'
    diffgradu_expected(X, t) = [X[2]/4 X[1]/4; X[1]/2+(X[1]+X[2])/2 X[1]/2]
    @test diffgradu == diffgradu_expected([0.5, 0.5], 1.0)
end

#=
"""basic continuum interpolations"""
function test_basic_interpolations()

    element = Quad4([1, 2, 3, 4])

    element["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element["temperature"] = ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0])
    element["displacement"] = (
        Vector[[0.0, 0.0], [0.0, 0.0], [0.00, 0.0], [0.0, 0.0]],
        Vector[[0.0, 0.0], [0.0, 0.0], [0.25, 0.0], [0.0, 0.0]])

    # from my old home works
    basis = get_basis(element)
    dbasis = grad(basis)
    @test isapprox(basis("geometry", [0.0, 0.0], 1.0) + basis("displacement", [0.0, 0.0], 1.0),  [9/16, 1/2])
    gradu = dbasis("displacement", [0.0, 0.0], 1.0)
    epsilon = 1/2*(gradu + gradu')
    rotation = 1/2*(gradu - gradu')
    X = basis("geometry", [0.0, 0.0], 1.0)
    k = 0.25
    epsilon_wanted = [X[2]*k 1/2*X[1]*k; 1/2*X[1]*k 0]
    rotation_wanted = [0 k/2*X[1]; -k/2*X[1] 0]
    @test isapprox(epsilon, epsilon_wanted)
    @test isapprox(rotation, rotation_wanted)
    F = I + gradu
    @test isapprox(F, [X[2]*k+1 X[1]*k; 0 1])
    C = F'*F
    @test isapprox(C, [(X[2]*k+1)^2  (X[2]*k+1)*X[1]*k; (X[2]*k+1)*X[1]*k  X[1]^2*k^2+1])
    E = 1/2*(F'*F - I)
    @test isapprox(E, [1/2*(X[2]*k + 1)^2-1/2  1/2*(X[2]*k+1)*X[1]*k; 1/2*(X[2]*k + 1)*X[1]*k  1/2*X[1]^2*k^2])
    U = 1/sqrt(trace(C) + 2*sqrt(det(C)))*(C + sqrt(det(C))*I)
    @test isapprox(U, [1.24235 0.13804; 0.13804 1.02149])
end

=#

function test_interpolation_in_temporal_basis()
    i1 = Increment(0.0)
    i2 = Increment(1.0)
    i3 = Increment(2.0)
    t1 = TimeStep(0.0, Increment[i1])
    t2 = TimeStep(2.0, Increment[i2])
    t3 = TimeStep(4.0, Increment[i3])
    field = Field(TimeStep[t1, t2, t3])

    info("field(1.0) = $(field(1.0))")

    @test field(-Inf) == [0.0]
    @test field( 0.0) == [0.0]
    @test field( 1.0) == [0.5]
    @test field( 2.0) == [1.0]
    @test field( 3.0) == [1.5]
    @test field( 4.0) == [2.0]
    @test field(+Inf) == [2.0]
end

function test_derivative_interpolation_in_temporal_basis_in_constant_velocity()
    i1 = Increment(0.0)
    i2 = Increment(1.0)
    i3 = Increment(2.0)
    t1 = TimeStep(0.0, Increment[i1])
    t2 = TimeStep(2.0, Increment[i2])
    t3 = TimeStep(4.0, Increment[i3])
    field = Field(TimeStep[t1, t2, t3])
    @test field(+Inf, Val{:derivative}) == [0.5]
    @test field(-Inf, Val{:derivative}) == [0.5]
    @test field( 0.0, Val{:derivative}) == [0.5]
    @test field( 0.5, Val{:derivative}) == [0.5]
    @test field( 1.0, Val{:derivative}) == [0.5]
    @test field( 1.5, Val{:derivative}) == [0.5]
    @test field( 2.0, Val{:derivative}) == [0.5]
end

function test_derivative_interpolation_in_temporal_basis_in_variable_velocity()
    t = linspace(0, 2, 5)
    x = 1/2*t.^2
    timesteps = TimeStep[]
    for (ti, xi) in zip(t, x)
        increment = Increment(xi)
        push!(timesteps, TimeStep(ti, increment))
    end
    # => ((0.0,0.0),(0.5,0.125),(1.0,0.5),(1.5,1.125),(2.0,2.0))
    pos = Field(timesteps)

    velocity = pos(1.0, Val{:derivative})[1]
    v1 = (0.500 - 0.125)/0.5
    v2 = (1.125 - 0.500)/0.5
    @test isapprox(velocity, mean([v1, v2])) # = 1.00

    velocity = pos(2.0, Val{:derivative})[1]
    @test isapprox(velocity, (2.0-1.125)/0.5) # = 1.75
end

function test_derivative_interpolation_in_temporal_basis_in_variable_velocity_check_type()
    t = linspace(0, 2, 5)
    x = 1/2*t.^2
    timesteps = TimeStep[]
    for (ti, xi) in zip(t, x)
        increment = Increment(xi)
        push!(timesteps, TimeStep(ti, increment))
    end
    # => ((0.0,0.0),(0.5,0.125),(1.0,0.5),(1.5,1.125),(2.0,2.0))
    pos = Field(timesteps)
    velocity = pos(1.0, Val{:derivative})
    # after interpolation, we are expecting to have same type where we started
    @test isa(velocity, Increment) == true
end

end
