# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


module FieldTests

using JuliaFEM
using JuliaFEM: Increment, TimeStep, Field, DefaultDiscreteField, FieldSet
using JuliaFEM: TemporalBasis, SpatialBasis, ContinuousField, DiscreteField
using JuliaFEM: DefaultContinuousField

using JuliaFEM.Test

function test_increment_constant_increment()
    I = Increment(1)
    @test isa(I, Increment)
    @test length(I) == 1
    @test I == 1
end

function test_increments_with_vector_data()
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test length(I1) == 3
    @test length(I2) == 3
    @test I1 == [1, 2, 3]
    @test I2 == [2, 3, 4]
end

function test_increments_basic_math()
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test 1/2*(I1+I2) == [1.5, 2.5, 3.5]
    @test I1 + 1 == [2, 3, 4]
    @test I1 - 1 == [0, 1, 2]
    @test I1*3 == [3, 6, 9]
    @test I1+I2 == [3, 5, 7]
end

function test_increment_dot_product()
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test dot(I1, I2) == 20
    @test dot([1,2,3], I2) == 20
    @test dot(I1, [2,3,4]) == 20
    @test dot([1, 2], Increment[I1, I2])
end

function test_increment_similarity()
    f = zeros(Increment, Int, 2, 4)
    @test length(f) == 4
    g = similar(f, ones(Int, 8))
    @test typeof(f) == typeof(g)
    @test length(f) == length(g)
    @test size(g) == (2, 4)
end

function test_increment_vec()
    g = zeros(Increment, Int, 2, 4)
    @test vec(g) == ones(Int, 8)
end

function test_increment_promotion()
    I1 = Increment([1, 2, 3])
    I2 = Increment([2, 3, 4])
    @test isa(I1+1, Increment)
    @test isa(I1-1, Increment)
    @test isa(3*I1, Increment)
    @test isa(1/2*I1, Increment)
    @test isa(I1+I2, Increment)
    @test isa(I1-I2, Increment)
end

function test_timestep_empty_timestep()
    ts = TimeStep()
    @test length(ts) == 0
    @test ts.time == 0.0
end

function test_timestep_with_two_increments()
    i1 = Increment([1, 2, 3])
    i2 = Increment([2, 3, 4])
    increments = Increment[i1, i2]
    ts = TimeStep(1.0, increments)
    @test length(ts) == 2
end

function test_create_timestep_with_scalar_value()
    ts = TimeStep(1)
    @test length(ts) == 1
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test ts[1] == [1]
end

function test_create_timestep_compactly_for_time_t0()
    ts = TimeStep([1, 2, 3])
    @test length(ts) == 1
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test ts[1] == [1, 2, 3]
end

function test_create_timestep_compactly_add_three_increments_compactly_for_time_t0()
    ts = TimeStep(1, 2, 3)
    @test length(ts) == 3
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
end

function test_create_timestep_compactly_add_two_increments()
    ts = TimeStep([1, 2, 3], [2, 3, 4])
    @test length(ts) == 2
    @test ts.time == 0.0
    @test isa(ts[1], Increment)
    @test isa(ts[2], Increment)
    @test ts[1] == [1, 2, 3]
    @test ts[2] == [2, 3, 4]
end

function test_create_timesteps_for_different_times()
    @test TimeStep(0.5, [1, 2]).time == 0.5
    @test TimeStep(0.5, [1, 2]) == [1, 2]
    @test TimeStep(0.5, 1).time == 0.5
    @test TimeStep(0.5, 1) == [1]
end

function test_default_discrete_field_quick_way_vector()
    f1 = DefaultDiscreteField([1, 2, 3])
    @debug("f1 = $f1")
    @test isa(f1[1], TimeStep)
    @test isa(f1[1][1], Increment)
    @test f1[1][1] == [1, 2, 3]
    @test f1[1].time == 0.0
end

function test_default_discrete_field_quick_way_scalar()
    f1 = DefaultDiscreteField(1)
    @test length(f1) == 1
    @test isa(f1[1], TimeStep)
    @test isa(f1[1][1], Increment)
    @test f1[1][1] == [1]
    @test f1[1].time == 0.0
end

function test_default_discrete_field_traditional_way()
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
end

function test_default_discrete_field_quick_way_two_timesteps_with_constant_value()
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
end

function test_default_discrete_field_quick_way_two_timesteps_with_vector_value()
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
end

function test_default_discrete_field_quick_way_set_time_vector_also()
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

function test_add_discrete_field_to_fieldset()
    fs = FieldSet()
    fs["temperature"] = DefaultDiscreteField([1, 2, 3])
    @test length(fs) == 1
    @test fs["temperature"] == [1, 2, 3]
end

function test_adding_discrete_fields_to_fieldset_quickly()
    fs = FieldSet()
    fs["temperature"] = [1, 2, 3, 4]
    @test fs["temperature"][end][end] == [1, 2, 3, 4]
    @test last(fs["temperature"]) == [1, 2, 3, 4]
end

function test_adding_all_kind_of_fields_to_fieldset()
    fs = FieldSet()
    fs["constant scalar field"] = 1
    fs["scalar field"] = [1, 2, 3, 4]
    fs["vector field"] = reshape(collect(1:8), 2, 4)
    fs["second order tensor field"] = reshape(collect(1:3*3*4), 3, 3, 4)
    fs["fourth order tensor field"] = reshape(collect(1:3*3*3*3*4), 3, 3, 3, 3, 4)
    timestep = fs["vector field"][end]
    @test fs["vector field"][end].time == 0.0
end

function test_adding_timesteps()
    fs = FieldSet()
    fs["temperature"] = [1, 2, 3, 4]
    T0 = last(fs["temperature"])  # last increment of last field
    @debug("last temperature T0 = $T0")
    T1 = Increment(T0 + 1)
    @debug("typeof T1 = $(typeof(T1))")
    timestep = TimeStep(1.0, Increment[T1])  # new list of increments for timestep
    push!(fs["temperature"], timestep)
    T2 = last(fs["temperature"])
    @debug("last temperature T2 = $T2")
    @test length(fs["temperature"]) == 2
    @test last(fs["temperature"]) == [2, 3, 4, 5]
    @test fs["temperature"][end].time == 1.0
end

function test_adding_timesteps_compactly()
    fs = FieldSet()
    fs["temperature"] = [1, 2, 3, 4]
    T0 = last(fs["temperature"])
    T1 = Increment(T0 + 1)
    push!(fs["temperature"], TimeStep(1.0, T1))
    @test length(fs["temperature"]) == 2
    @test last(fs["temperature"]) == [2, 3, 4, 5]
    @test fs["temperature"][end].time == 1.0
end

function test_add_several_timesteps_without_time_vector()
    fs = FieldSet()
    fs["time series"] = [1, 2, 3, 4], [2, 3, 4, 5]
    @debug("fieldset = $fs")
    @test fs["time series"][1].time == 0.0
    @test fs["time series"][2].time == 1.0
    @test fs["time series"][1][end] == [1, 2, 3, 4]
    @test fs["time series"][2][end] == [2, 3, 4, 5]
end

function test_adding_several_timesteps_at_once_with_time_vector()
    fs = FieldSet()
    fs["time series"] = (0.0, [1, 2, 3, 4]), (0.5, [2, 3, 4, 5])
    @test fs3["time series"][1].time == 0.0
    @test fs3["time series"][2].time == 0.5
    @test fs3["time series"][1][end] == [1, 2, 3, 4]
    @test fs3["time series"][2][end] == [2, 3, 4, 5]
end

function test_adding_continuous_field_to_fieldset()
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
    @debug("data = $data, typeof data = $(typeof(data))")
    basis = time*field.basis(xi) # evaluate basis at point ξ.
    sum([basis[i]*data[i] for i=1:length(data)]) # sum results
end
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
    T1 = Increment(T0 + 1)
    push!(fs["discrete field"], TimeStep(1.0, T1))
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
