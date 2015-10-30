# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


module TypesTests

using JuliaFEM: Increment, TimeStep, AbstractField, DefaultDiscreteField, FieldSet
using JuliaFEM: TemporalBasis, SpatialBasis, ContinuousField, DiscreteField
using JuliaFEM: Field


using Base.Test

function test_increment()
    info("testing Increment")
    # testing Increment
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test dot(I1, I2) == 20
    @test dot([1,2,3], I2) == 20
    @test dot(I1, [2,3,4]) == 20
    @test 1/2*(I1+I2) == [1.5, 2.5, 3.5]
    @test I1 + 1 == [2, 3, 4]
    @test I1 - 1 == [0, 1, 2]
    @test I1*3 == [3, 6, 9]
    @test I1+I2 == [3, 5, 7]
    f = zeros(Increment, 2, 4)
    @test length(f) == 4

    g = similar(f, ones(8))
    @test typeof(f) == typeof(g)
    @test length(f) == length(g)

    # promotion of increment
    @test typeof(I1+1) == typeof(I1)
    @test typeof(I1-1) == typeof(I1)
    @test typeof(I1*3) == typeof(I1)

    # FIXME
    #@test typeof(I1) == typeof(I1+I2)
    #@test typeof(I1/2) == typeof(I1)
    #@test typeof(1/2*S1) == typeof(I1)
end
test_increment()

function test_timestep()
    info("testing TimeStep")
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    i3 = Increment([2, 3, 4])
    i4 = Increment([3, 4, 5])
    t1 = TimeStep(1.0, Increment[i1, i2])
    t2 = TimeStep(2.0, Increment[i3, i4])
    @test length(t1) == length(t2) == 2
    t3 = TimeStep(3.0, i1+1)
end
test_timestep()

function test_watta_fak()
    # TODO: this test will fail if Increment is typealiased to Vector 
    fs = FieldSet()
    fs["discrete field"] = [1, 2, 3, 4]
    T0 = last(fs["discrete field"])
    info("last discrete field: $T0, ", typeof(T0))
    T1 = T0 + 1
    info("adding 1 to discrete field: $T1, ", typeof(T1))
    ts = TimeStep(1.0, T1)
    info("creating time step: $ts, ", typeof(ts))
    push!(fs["discrete field"], ts)
    info("last discrete field = ", last(fs["discrete field"]))

    info("fieldset: $fs")

    @test last(fs["discrete field"]) == [2, 3, 4, 5]
end
test_watta_fak()

function test_default_discrete_field()
    info("testing DefaultDiscreteField")
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    i3 = Increment([2, 3, 4])
    i4 = Increment([3, 4, 5])
    t1 = TimeStep(1.0, Increment[i1, i2])
    t2 = TimeStep(2.0, Increment[i3, i4])
    timesteps = TimeStep[t1, t2]
    f1 = DefaultDiscreteField(timesteps)
    @test length(f1) == 2
    @test isa(f1, AbstractField) == true
end
test_default_discrete_field()

function test_fieldset()
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    i3 = Increment([2, 3, 4])
    i4 = Increment([3, 4, 5])
    t1 = TimeStep(1.0, Increment[i1, i2])
    t2 = TimeStep(2.0, Increment[i3, i4])
    timesteps = TimeStep[t1, t2]
    f1 = DefaultDiscreteField(timesteps)
    info("testing adding discrete field to FieldSet")
    fs = FieldSet()
    fs["temperature"] = f1
    @test length(fs) == 1

    info("testing adding discrete fields quickly")
    # the easy way
    fs2 = FieldSet()
    fs2["temperature"] = [1, 2, 3, 4]
    @test fs2["temperature"][end][end] == [1, 2, 3, 4]
    @test last(fs2["temperature"]) == [1, 2, 3, 4]

    fs2 = FieldSet()
    fs2["constant scalar field"] = 1
    fs2["scalar field"] = [1, 2, 3, 4]
    fs2["vector field"] = reshape(collect(1:8), 2, 4)
    fs2["second order tensor field"] = reshape(collect(1:3*3*4), 3, 3, 4)
    fs2["fourth order tensor field"] = reshape(collect(1:3*3*3*3*4), 3, 3, 3, 3, 4)
    timestep = fs2["vector field"][end]
    @test timestep.time == 0.0

    info("testing adding timesteps")
    # add another timestep
    fs = FieldSet()
    fs["temperature"] = [1, 2, 3, 4]
    T0 = last(fs["temperature"])  # last increment of last field
    info("last temperature = $T0")
    T1 = T0 + 1
    @test typeof(T0) == typeof(T1)
    timestep = TimeStep(1.0, Increment[T1])  # new list of increments for timestep
    push!(fs["temperature"], timestep)
    T2 = last(fs["temperature"])
    info("last temperature = $T2")
    @test last(fs["temperature"]) == [2, 3, 4, 5]
    # or more easily
    timestep = TimeStep(2.0, T1)
    push!(fs["temperature"], timestep)
    @test length(fs["temperature"].timesteps) == 3

    info("test adding several time steps at once")
    fs3 = FieldSet()
    fs3["time series 1"] = (0.0, [1, 2, 3, 4]), (0.5, [2, 3, 4, 5]), (1.0, [1, 1, 1, 1])
    @test fs3["time series 1"][end].time == 1.0
    fs3["time series 2"] = [1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 1, 1]
    @test fs3["time series 2"][end].time == 2.0
end
test_fieldset()

type MyFunnyContinuousField <: ContinuousField
    basis :: Function
    discretefield :: DiscreteField
end
function Base.call(field::MyFunnyContinuousField, xi::Vector, time::Number=1.0)
    data = last(field.discretefield) # get the last timestep last increment
    info("data = $data, typeof data = $(typeof(data))")
    basis = time*field.basis(xi) # evaluate basis at point ξ.
    sum([basis[i]*data[i] for i=1:length(data)]) # sum results
end
function test_continuous_field()
    info("testing continuous field")
    fs = FieldSet()
    fs["discrete field"] = [1, 2, 3, 4]
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2]),
        (1+xi[1])*(1-xi[2]),
        (1+xi[1])*(1+xi[2]),
        (1-xi[1])*(1+xi[2])]
    fs["continuous field"] = MyFunnyContinuousField(basis, fs["discrete field"])
    @test fs["continuous field"]([0.0, 0.0], 1.0) == 1/4*(1+2+3+4)
    T0 = last(fs["discrete field"])
    T1 = T0 + 1.0
    ts = TimeStep(1.0, T1)
    push!(fs["discrete field"], TimeStep(1.0, T0+1.0))
    @test fs["continuous field"]([0.0, 0.0], 1.0) == 1/4*(2+3+4+5)
end
test_continuous_field()

type MyFunnyDiscreteField <: DiscreteField
    discrete_points :: Vector
    continuousfield :: ContinuousField
end
Base.length(field::MyFunnyDiscreteField) = length(field.discrete_points)
Base.endof(field::MyFunnyDiscreteField) = endof(field.discrete_points)
Base.last(field::MyFunnyDiscreteField) = Float64[field[i] for i=1:length(field)]
function Base.getindex(field::MyFunnyDiscreteField, idx::Int64)
    field.continuousfield(field.discrete_points[idx])
end

function test_discrete_field()
    info("testing discrete field")
    fs = FieldSet()
    fs["discrete field"] = [1, 2, 3, 4]
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2]),
        (1+xi[1])*(1-xi[2]),
        (1+xi[1])*(1+xi[2]),
        (1-xi[1])*(1+xi[2])]
    fs["continuous field"] = MyFunnyContinuousField(basis, fs["discrete field"])
    discrete_points = 1.0/sqrt(3.0)*Vector[[-1, -1], [1, -1], [1, 1], [-1, 1]]
    fs["discrete field 2"] = MyFunnyDiscreteField(discrete_points, fs["continuous field"])
    @test last(fs["discrete field 2"]) ≈ [
        1.7559830641437073,
        2.0893163974770410,
        2.9106836025229590,
        3.2440169358562922]
end
test_discrete_field()


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
    @test position ≈ 0.50
    velocity = call(fs["particle"], temporalbasis, 2.0, Val{:derivative})[1]
    @test velocity ≈ (2.0-1.125)/0.5 # = 1.75
    velocity = call(fs["particle"], temporalbasis, 1.0, Val{:derivative})[1]
    v1 = (0.500 - 0.125)/0.5
    v2 = (1.125 - 0.500)/0.5
    info("v1 = $v1, v2 = $v2")
    info(mean([v1, v2]))
    @test velocity ≈ mean([v1, v2]) # = 1.00

    # FIXME, returns wrong type.
    #=
    @test isa(position, Increment) == true
    @test isa(velocity, Increment) == true
    =#
end
test_interpolation_in_temporal_basis()

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
test_interpolation_in_spatial_basis()


println("test_fields.jl: all test passing.")

end
