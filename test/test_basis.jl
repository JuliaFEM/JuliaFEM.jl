# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module BasisTests

using JuliaFEM.Test
using JuliaFEM.Core: AbstractElement, Element
import JuliaFEM.Core: get_basis, get_dbasis

abstract TestElement <: AbstractElement

function get_basis(::Type{TestElement}, xi::Vector{Float64})
    1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
end

function get_dbasis(::Type{TestElement}, xi::Vector{Float64})
    1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]
end

function get_element()
    element = Element{TestElement}([1, 2, 3, 4])
    element["geometry"] = Vector{Float64}[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element["temperature"] = Float64[1.0, 2.0, 3.0, 4.0]
    element["displacement1"] = Vector{Float64}[[0.0, 0.0], [0.0,  0.0], [1/4, 0.0], [0.0, 0.0]]
    element["displacement2"] = Vector{Float64}[[0.0, 0.0], [1.0, -1.0], [2.0, 3.0], [0.0, 0.0]]
    return element
end

### Test interpolation in spatial domain

function test_basis_interpolation()
    element = get_element()
    info(element([0.0, 0.0]))
    @test element([0.0, 0.0]) == 1/4*[1 1 1 1]
    @test element([0.0, 0.0], 1.0) == 1/4*[1 1 1 1]
end

function test_basis_gradient_interpolation()
    element = get_element()
    grad = element([0.0, 0.0], Val{:grad})
    info("grad = \n$grad")
    @test grad == 1/2*[-1 1 1 -1; -1 -1 1 1]
end

function test_interpolation_of_scalar_field_in_spatial_domain()
    # in unit square: T(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    element = get_element()
    T_known(X) = 1 + X[1] + 3*X[2] - 2*X[1]*X[2]
    T_interpolated = element("temperature", [0.0, 0.0])
    @test T_interpolated == T_known([0.5, 0.5])
end

function test_interpolation_of_gradient_of_scalar_field_in_spatial_domain()
    # in unit square: grad(T)(X) = [1-2X[2], 3-2*X[1]]
    element = get_element()
    gradT = element("temperature", [0.0, 0.0], Val{:grad})
    gradT_expected(X) = [1-2*X[2] 3-2*X[1]]
    @test gradT == gradT_expected([0.5, 0.5])
end

function test_interpolation_of_vector_field()
    # in unit square, u(X,t) = [1/4*t*X[1]*X[2], 0, 0]
    element = get_element()
    u = element("displacement1", [0.0, 0.0])
#   x = X+u
    u_expected(X) = [1/4*X[1]*X[2], 0]
#   @test isapprox(x, [9/16, 1/2])
    @test isapprox(u, u_expected([0.5, 0.5]))
end


function test_interpolation_of_gradient_of_vector_field()
    # in unit square, u(X) = t*[X[1]*(X[2]+1), X[1]*(4*X[2]-1)]
    # => u_i,j = t*[X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    element = get_element()
#    displacement = Field(
#        (0.5, Vector[[0.0, 0.0], [0.5, -0.5], [1.0, 1.5], [0.0, 0.0]]),
#        (1.5, Vector[[0.0, 0.0], [1.5, -1.5], [3.0, 4.5], [0.0, 0.0]]))
    gradu = element("displacement2", [0.0, 0.0], Val{:grad})
    gradu_expected(X) = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    @test isapprox(gradu, gradu_expected([0.5, 0.5]))
end

### Test interpolation in time domain

#=

function test_linear_time_extrapolation_of_field()
    #T_known(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    T = Field(
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 2.0, 3.0, 4.0]))
    @test T(-1.0) == -1.0*[1.0, 2.0, 3.0, 4.0]
    @test T( 3.0) ==  3.0*[1.0, 2.0, 3.0, 4.0]
    # when going to \pm infinity, return the last one.
    @test T(-Inf) ==  0.0*[1.0, 2.0, 3.0, 4.0]
    @test T(+Inf) ==  1.0*[1.0, 2.0, 3.0, 4.0]
end

function test_constant_time_extrapolation_of_field()
    #T_known(X,t) = t*(1 + X[1] + 3*X[2] - 2*X[1]*X[2])
    T = Field(
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (1.0, [1.0, 2.0, 3.0, 4.0]))
    @test T(-1.0, :constant) == [0.0, 0.0, 0.0, 0.0]
    @test T( 3.0, :constant) == [1.0, 2.0, 3.0, 4.0]
end

function test_time_extrapolation_of_field_with_single_timestep()
    T = Field([1.0, 2.0, 3.0, 4.0])
    @test T(1.0) == [1.0, 2.0, 3.0, 4.0]
end

function test_interpolation_in_temporal_basis()
    i1 = Increment(0.0)
    i2 = Increment(1.0)
    i3 = Increment(2.0)
    t1 = TimeStep(0.0, Increment[i1])
    t2 = TimeStep(2.0, Increment[i2])
    t3 = TimeStep(4.0, Increment[i3])
    field = Field(TimeStep[t1, t2, t3])
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
    @test field(+Inf, Val{:diff}) == [0.5]
    @test field(-Inf, Val{:diff}) == [0.5]
    @test field( 0.0, Val{:diff}) == [0.5]
    @test field( 0.5, Val{:diff}) == [0.5]
    @test field( 1.0, Val{:diff}) == [0.5]
    @test field( 1.5, Val{:diff}) == [0.5]
    @test field( 2.0, Val{:diff}) == [0.5]
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

    velocity = pos(1.0, Val{:diff})[1]
    v1 = (0.500 - 0.125)/0.5
    v2 = (1.125 - 0.500)/0.5
    @test isapprox(velocity, mean([v1, v2])) # = 1.00

    velocity = pos(2.0, Val{:diff})[1]
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
    velocity = pos(1.0, Val{:diff})
    # after interpolation, we are expecting to have same type where we started
    @test isa(velocity, Increment) == true
end

function test_time_derivative_gradient_interpolation_of_field()
    # in unit square, u(X) = t*[X[1]*(X[2]+1), X[1]*(4*X[2]-1)]
    # => u_i,j = t*[X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    # => d(u_i,j)/dt = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    geometry = Field([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    displacement = Field(
        (0.5, Vector[[0.0, 0.0], [0.5, -0.5], [1.0, 1.5], [0.0, 0.0]]),
        (1.5, Vector[[0.0, 0.0], [1.5, -1.5], [3.0, 4.5], [0.0, 0.0]]))

    # wanted
    #u = get_basis(element, "displacement")
    #L = grad(diff(u))
    #D = 1/2*(L + L')
    #@test isapprox(D([0.0, 0.0], 1.0), ...)

    basis, dbasis = get_basis()
    N = Basis(basis, dbasis)
    xi = [0.0, 0.0]
    time = 1.2
    grad = ElementGradientBasis(N, geometry)(xi, time)
    increment = displacement(time, Val{:derivative})
    diffgradu = sum([grad[:,i]*increment[i]' for i=1:length(increment)])'
    diffgradu_expected(X, t) = [X[2]+1 X[1]; 4*X[2]-1 4*X[1]]
    @test diffgradu == diffgradu_expected([0.5, 0.5], 1.2)
end

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


end
