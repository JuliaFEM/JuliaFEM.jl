# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module BasisTests

using JuliaFEM.Test

using JuliaFEM: get_basis, grad, FieldSet, Field, Quad4


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

function test_interpolation_in_temporal_basis()
    info("testing interpolation on temporal basis")
    temporalbasis = TemporalBasis((t) -> [1-t, t], (t) -> [-1, 1])
    @test temporalbasis(0.2) == [0.8, 0.2]
    i1 = Increment([0.0])
    i2 = Increment([1.0])
    i3 = Increment([2.0])
    t1 = TimeStep(0.0, Increment[i1])
    t2 = TimeStep(2.0, Increment[i2])
    t3 = TimeStep(4.0, Increment[i3])
    field = Field(TimeStep[t1, t2, t3])
    @test call(field, temporalbasis, -Inf) == [0.0]
    @test call(field, temporalbasis,  0.0) == [0.0]
    @test call(field, temporalbasis,  1.0) == [0.5]
    @test call(field, temporalbasis,  2.0) == [1.0]
    @test call(field, temporalbasis,  3.0) == [1.5]
    @test call(field, temporalbasis,  4.0) == [2.0]
    @test call(field, temporalbasis, +Inf) == [2.0]
    @test call(field, temporalbasis, +Inf, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis, -Inf, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis,  0.0, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis,  0.5, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis,  1.0, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis,  1.5, Val{:derivative}) == [0.5]
    @test call(field, temporalbasis,  2.0, Val{:derivative}) == [0.5]
    fs = FieldSet()

    t = collect(linspace(0, 2, 5))
    x = 1/2*t.^2
    x2 = tuple(collect(zip(t, x))...)
    # => ((0.0,0.0),(0.5,0.125),(1.0,0.5),(1.5,1.125),(2.0,2.0))
    fs["particle"] = x2
    position = call(fs["particle"], temporalbasis, 1.0)[1]
    @test isapprox(position, 0.50)
    velocity = call(fs["particle"], temporalbasis, 2.0, Val{:derivative})[1]
    @test isapprox(velocity, (2.0-1.125)/0.5) # = 1.75
    velocity = call(fs["particle"], temporalbasis, 1.0, Val{:derivative})[1]
    v1 = (0.500 - 0.125)/0.5
    v2 = (1.125 - 0.500)/0.5
    info("v1 = $v1, v2 = $v2")
    info(mean([v1, v2]))
    @test isapprox(velocity, mean([v1, v2])) # = 1.00

    # FIXME, returns wrong type.
    @test isa(position, Increment) == true
    @test isa(velocity, Increment) == true
end

function test_interpolation_in_spatial_basis()
    info("testing interpolation on spatial basis")
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
    dbasis(xi) = 1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]
    spatialbasis = SpatialBasis(basis, dbasis)
    @test spatialbasis.basis([0.0, 0.0]) == 1/4*[1 1 1 1]

    fs = FieldSet()
    fs["geometry"] = Vector{Float64}[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    fs["displacement"] = (0.0, zeros(2, 4)), (1.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.25, 0.0], [0.0, 0.0]])

    X = call(last(fs["geometry"]), spatialbasis, [0.0, 0.0])
    u = call(last(fs["displacement"]), spatialbasis, [0.0, 0.0])
    x = X+u
    @test X ≈ 1/2*[1, 1]
    @test x ≈ [9/16, 1/2]

    gradu = call(last(fs["displacement"]), spatialbasis, [0.0, 0.0], last(fs["geometry"]), Val{:gradient})
    @test gradu ≈ [0.125 0.125; 0.0 0.0]
end

end
