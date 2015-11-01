# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


module FieldTests

using JuliaFEM: Increment, TimeStep, Field, DefaultDiscreteField, FieldSet
using JuliaFEM: TemporalBasis, SpatialBasis, ContinuousField, DiscreteField
using JuliaFEM: DefaultContinuousField

using JuliaFEM.Test

""" testing Increment """
function test_increment()

    # constant increment
    I3 = Increment(1)
    @test isa(I3, Increment)
    @test length(I3) == 1
    # FIXME: i don't like 1-length arrays
    @test I3 == [1]

    # two increments with vector data
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test length(I1) == 3
    @test length(I2) == 3

    # basic math
    @test 1/2*(I1+I2) == [1.5, 2.5, 3.5]
    @test I1 + 1 == [2, 3, 4]
    @test I1 - 1 == [0, 1, 2]
    @test I1*3 == [3, 6, 9]
    @test I1+I2 == [3, 5, 7]

    # dot product
    @test dot(I1, I2) == 20
    @test dot([1,2,3], I2) == 20
    @test dot(I1, [2,3,4]) == 20
    @test dot([1, 2], Increment[I1, I2])

    # similarity
    f = zeros(Increment, 2, 4)
    @test length(f) == 4
    g = similar(f, ones(8))
    @test typeof(f) == typeof(g)
    @test length(f) == length(g)

    # vec
    @test vec(g) == ones(8)

    # promotion
    # FIXME: how to do promotion so that modified increment is still increment?
    @test isa(I1+1, Increment)
    @test isa(I1-1, Increment)
    @test isa(3*I1, Increment)
    @test isa(1/2*I1, Increment)
    @test isa(I1+I2, Increment)
    @test isa(I1-I2, Increment)

end

""" testing TimeStep """
function test_timestep()

    info("test_timestep(): create empty timestep")
    ts = TimeStep()
    @test length(ts) == 0
    @test ts.time == 0.0

    info("create timestep with two increments")
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    increments = Increment[i1, i2]
    ts = TimeStep(1.0, increments)
    @test length(ts) == 2

    info("create timestep with scalar value")
    ts = TimeStep(1)
    @test length(ts) == 1
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test ts[1] == [1]

    info("create timestep compactly for time t=0.0")
    ts = TimeStep([1, 2, 3])
    @test length(ts) == 1
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test ts[1] == [1, 2, 3]

    info("create timestep compactly, add three increments compactly for time t=0.0")
    ts = TimeStep(1, 2, 3)
    @test length(ts) == 3
    @test ts.time == 0.0
    @test isa(ts[1], Increment)

    info("create timestep compactly, add two increments compactly for time t=0.0")
    ts = TimeStep([1, 2, 3], [2, 3, 4])
    @test length(ts) == 2
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test ts[1] == [1, 2, 3]
    @test ts[2] == [2, 3, 4]

    info("create standard timesteps")
    info(TimeStep(0.5, Increment([1, 2])))

    @test TimeStep(0.5, [1, 2]).time == 0.5
    @test TimeStep(0.5, [1, 2]) == [1, 2]
    @test TimeStep(0.5, 1).time == 0.5
    @test TimeStep(0.5, 1) == [1]

end

""" testing DefaultDiscreteField """
function test_default_discrete_field()

    info("test_default_discrete_field(): the traditional way")
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    t1 = TimeStep(1.0, Increment[i1, i2])
    i3 = Increment([2, 3, 4])
    i4 = Increment([3, 4, 5])
    t2 = TimeStep(2.0, Increment[i3, i4])
    timesteps = TimeStep[t1, t2]
    f1 = DefaultDiscreteField(timesteps)
    @test length(f1) == 2
    @test isa(f1, Field)
    @test f1[1][1] == [1, 2, 3]
    @test f1[1][2] == [2, 3, 4]
    @test f1[2][1] == [2, 3, 4]
    @test f1[2][2] == [3, 4, 5]
    @test f1[1].time == 1.0
    @test f1[2].time == 2.0

    info("test_default_discrete_field(): quick way, this creates one timestep with vector value")
    f1 = DefaultDiscreteField([1, 2, 3])
    info("f1 = $f1")
    @test isa(f1[1], TimeStep)
    @test isa(f1[1][1], Increment)
    @test f1[1][1] == [1, 2, 3]
    @test f1[1].time == 0.0

    info("test_default_discrete_field(): quick way, two timesteps with constant value")
    f1 = DefaultDiscreteField(1, 2)
    @test length(f1) == 2
    @test isa(f1[1], TimeStep)
    @test isa(f1[2], TimeStep)
    @test isa(f1[1][1], Increment)
    @test isa(f1[2][1], Increment)
    @test f1[1][1] == [1]
    @test f1[2][1] == [2]
    @test f1[1].time == 0.0
    @test f1[2].time == 1.0

    info("test_default_discrete_field(): quick way, one timestep with scalar value")
    f1 = DefaultDiscreteField(1)
    @test length(f1) == 1
    @test isa(f1[1], TimeStep)
    @test isa(f1[1][1], Increment)
    @test f1[1][1] == [1]
    @test f1[1].time == 0.0

    info("test_default_discrete_field(): quick way, two timesteps with vector value")
    f1 = DefaultDiscreteField([1, 2, 3], [3, 4, 5])
    @test length(f1) == 2
    @test isa(f1[1], TimeStep)
    @test isa(f1[2], TimeStep)
    @test isa(f1[1][1], Increment)
    @test isa(f1[2][1], Increment)
    @test f1[1][1] == [1, 2, 3]
    @test f1[2][1] == [3, 4, 5]
    @test f1[1].time == 0.0
    @test f1[2].time == 1.0

    info("test_default_discrete_field(): quick way, set time vector also")
    f1 = DefaultDiscreteField( (0.5, [1, 2, 3]), (1.0, [3, 4, 5]) )
    @test isa(f1[1], TimeStep)
    @test isa(f1[2], TimeStep)
    @test isa(f1[1][1], Increment)
    @test isa(f1[2][1], Increment)
    @test f1[1][1] == [1, 2, 3]
    @test f1[2][1] == [3, 4, 5]
    @test f1[1].time == 0.5
    @test f1[2].time == 1.0

end

""" testing DefaultContinuousField """
function test_default_continuous_field()

    function myfield(xi::Vector, time::Float64)
        time/4*[
        (1-xi[1])*(1-xi[2]),
        (1+xi[1])*(1-xi[2]),
        (1+xi[1])*(1+xi[2]),
        (1-xi[1])*(1+xi[2])]'
    end

    f = DefaultContinuousField(myfield)
    @test f([0.0, 0.0], 1.0) == [0.25 0.25 0.25 0.25]
    
end

""" testing FieldSet """
function test_fieldset()

    info("test_fieldset(): testing adding discrete field to FieldSet")
    fs = FieldSet()
    fs["temperature"] = DefaultDiscreteField([1, 2, 3])
    @test length(fs) == 1

    info("test_fieldset(): testing adding discrete fields quickly")
    fs2 = FieldSet()
    fs2["temperature"] = [1, 2, 3, 4]
    @test fs2["temperature"][end][end] == [1, 2, 3, 4]
    @test last(fs2["temperature"]) == [1, 2, 3, 4]

    info("test_fieldset(): testing adding all kind of discrete fields")
    fs2 = FieldSet()
    fs2["constant scalar field"] = 1
    fs2["scalar field"] = [1, 2, 3, 4]
    fs2["vector field"] = reshape(collect(1:8), 2, 4)
    fs2["second order tensor field"] = reshape(collect(1:3*3*4), 3, 3, 4)
    fs2["fourth order tensor field"] = reshape(collect(1:3*3*3*3*4), 3, 3, 3, 3, 4)
    timestep = fs2["vector field"][end]
    @test timestep.time == 0.0

    info("test_fieldset(): testing adding timesteps")
    fs = FieldSet()
    fs["temperature"] = [1, 2, 3, 4]
    T0 = last(fs["temperature"])  # last increment of last field
    info("last temperature T0 = $T0")
    T1 = Increment(T0 + 1)
    info("typeof T1 = $(typeof(T1))")
    timestep = TimeStep(1.0, Increment[T1])  # new list of increments for timestep
    push!(fs["temperature"], timestep)
    T2 = last(fs["temperature"])
    info("last temperature T2 = $T2")
    @test last(fs["temperature"]) == [2, 3, 4, 5]

    info("test_fieldset(): testing adding timesteps compactly")
    timestep = TimeStep(2.0, T1)
    push!(fs["temperature"], timestep)
    @test length(fs["temperature"].timesteps) == 3

    info("test_fieldset(): test adding several time steps at once without time vector")
    fs3 = FieldSet()
    fs3["time series 2"] = [1, 2, 3, 4], [2, 3, 4, 5]
    info(fs3)
    @test fs3["time series 2"][1].time == 0.0
    @test fs3["time series 2"][2].time == 1.0
    @test fs3["time series 2"][1][end] == [1, 2, 3, 4]
    @test fs3["time series 2"][2][end] == [2, 3, 4, 5]

    info("test_fieldset(): test adding several time steps at once with time vector")
    fs3 = FieldSet()
    fs3["time series 1"] = (0.0, [1, 2, 3, 4]), (0.5, [2, 3, 4, 5])
    @test fs3["time series 1"][1].time == 0.0
    @test fs3["time series 1"][2].time == 0.5
    @test fs3["time series 1"][1][end] == [1, 2, 3, 4]
    @test fs3["time series 1"][2][end] == [2, 3, 4, 5]

    info("test_fieldset(): adding continuous field")
    fs = FieldSet()
    fs["continuous field"] = (xi, t) -> xi[1]*xi[2]*t
    @test fs["continuous field"]([1.0, 2.0], 3.0) == 6.0

end

type MyContinuousField <: ContinuousField
    basis :: Function
    discrete_field :: DiscreteField
end
function Base.call(field::MyContinuousField, xi::Vector, time::Number=1.0)
    data = last(field.discrete_field) # get the last timestep last increment
    info("data = $data, typeof data = $(typeof(data))")
    basis = time*field.basis(xi) # evaluate basis at point ξ.
    sum([basis[i]*data[i] for i=1:length(data)]) # sum results
end

""" testing ContinuousField """
function test_continuous_field()
    fs = FieldSet()
    fs["discrete field"] = [1, 2, 3, 4]
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2]),
        (1+xi[1])*(1-xi[2]),
        (1+xi[1])*(1+xi[2]),
        (1-xi[1])*(1+xi[2])]
    fs["continuous field"] = MyContinuousField(basis, fs["discrete field"])
    @test fs["continuous field"]([0.0, 0.0], 1.0) == 1/4*(1+2+3+4)
    T0 = last(fs["discrete field"])
    T1 = T0 + 1.0
    ts = TimeStep(1.0, T1)
    push!(fs["discrete field"], TimeStep(1.0, T0+1.0))
    @test fs["continuous field"]([0.0, 0.0], 1.0) == 1/4*(2+3+4+5)
end

type MyDiscreteField <: DiscreteField
    discrete_points :: Vector
    continuous_field :: ContinuousField
end
Base.length(field::MyDiscreteField) = length(field.discrete_points)
Base.endof(field::MyDiscreteField) = endof(field.discrete_points)
Base.last(field::MyDiscreteField) = Float64[field[i] for i=1:length(field)]
function Base.getindex(field::MyDiscreteField, idx::Int64)
    field.continuous_field(field.discrete_points[idx])
end

""" testing DiscreteField """
function test_discrete_field()
    fs = FieldSet()
    fs["discrete field"] = [1, 2, 3, 4]
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2]),
        (1+xi[1])*(1-xi[2]),
        (1+xi[1])*(1+xi[2]),
        (1-xi[1])*(1+xi[2])]
    fs["continuous field"] = MyContinuousField(basis, fs["discrete field"])
    discrete_points = 1.0/sqrt(3.0)*Vector[[-1, -1], [1, -1], [1, 1], [-1, 1]]
    fs["discrete field 2"] = MyDiscreteField(discrete_points, fs["continuous field"])
    @test last(fs["discrete field 2"]) ≈ [
        1.7559830641437073,
        2.0893163974770410,
        2.9106836025229590,
        3.2440169358562922]
end

function test_field_conversion()
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    t1 = TimeStep(1.0, Increment[i1, i2])
    i3 = Increment([2, 3, 4])
    i4 = Increment([3, 4, 5])
    t2 = TimeStep(2.0, Increment[i3, i4])
    timesteps = TimeStep[t1, t2]
    info("timesteps = $timesteps")
    f1 = Field(timesteps)
    info("field = $f1")
    @test length(f1) == 2
    @test isa(f1, Field)
    @test f1[1][1] == [1, 2, 3]
    @test f1[1][2] == [2, 3, 4]
    @test f1[2][1] == [2, 3, 4]
    @test f1[2][2] == [3, 4, 5]
    @test f1[1].time == 1.0
    @test f1[2].time == 2.0
end

end
