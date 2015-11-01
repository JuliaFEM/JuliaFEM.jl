# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractBasis

""" Defined to dimensionless coordinate ξ∈[-1,1]^n. """
type SpatialBasis <: AbstractBasis
    basis :: Function
    dbasisdxi :: Function
end

typealias Basis SpatialBasis

""" Defined to to interval t∈[0, 1]. """
type TemporalBasis <: AbstractBasis
    basis :: Function
    dbasisdt :: Function
end
function TemporalBasis()
    basis(t) = [1-t, t]
    dbasis(t) = [-1, 1]
    return TemporalBasis(basis, dbasis)
end

function call(b::TemporalBasis, value::Number)
    b.basis(value)
end

function call(b::SpatialBasis, value::Vector)
    b.basis(value)
end

### INTERPOLATION IN TIME DOMAIN ###

function Base.call(field::Field, basis::TemporalBasis, time)
    # FieldSet -> Field -> TimeStep -> Increment -> data
    # special cases, -Inf, +Inf and ~0.0
    if time > field[end].time
        return field[end][end]
    end
    if (time < field[1].time) || abs(time-field[1].time) < 1.0e-12
        return field[1][end]
    end
    i = length(field)
    while field[i].time >= time
        i -= 1
    end
    field[i].time == time && return field[i][end]
    t1 = field[i].time
    t2 = field[i+1].time
    inc1 = field[i][end]
    inc2 = field[i+1][end]
    # TODO: may there be some reasons for "unphysical" jumps in
    # fields w.r.t time which should be taken account in some way?
    # i.e. dt between two fields → 0
    dt = t2 - t1
    b = basis.basis((time-t1)/dt)
    r = Increment[inc1, inc2]
    return dot(b, r)
end
function Base.call(field::DiscreteField, time)
    return Base.call(field, TemporalBasis(), time)
end

function Base.call(field::Field, basis::TemporalBasis, time,
                   derivative::Type{Val{:derivative}})
    # FieldSet -> Field -> TimeStep -> Increment -> data

    if length(field) == 1
        # just one timestep, time derivative cannot be evaluated.
        error("Field length = $(length(field)), cannot evaluate time derivative")
    end

    function eval_field(i, j)
        timesteps = TimeStep[field[i], field[j]]
        increments = Increment[timesteps[1][end], timesteps[2][end]]
        J = norm(timesteps[2].time - timesteps[1].time)
        dbasisdt = basis.dbasisdt( (time-timesteps[1].time)/J )
        return dot(dbasisdt, increments)/J
    end

    # special cases, +Inf, -Inf, ~0.0
    if (time > field[end].time) || isapprox(time, field[end].time)
        return eval_field(endof(field)-1, endof(field))
    end
    if (time < field[1].time) || isapprox(time, field[1].time)
        return eval_field(1, 2)
    end

    # search for a correct "bin" between time steps
    i = length(field)
    while (field[i].time > time) && !isapprox(field[i].time, time)
        i -= 1
    end

    if isapprox(field[i].time, time)
        # This is the hard case, maybe discontinuous time
        # derivative if linear approximation.
        # we are on the "mid node" in time axis
        field1 = eval_field(i-1,i)
        field2 = eval_field(i,i+1)
        return 1/2*(field1 + field2)
    end

    return eval_field(i, i+1)

end

### INTERPOLATION IN SPATIAL DOMAIN ###

function Base.call(increment::Increment, basis::SpatialBasis, xi::Vector)
    basis = basis.basis(xi)
    sum([basis[i]*increment[i] for i=1:length(increment)])
end

function Base.call(increment::Increment, basis::SpatialBasis, xi::Vector,
                   geometry::Increment, gradient::Type{Val{:gradient}})
    dbasis = basis.dbasisdxi(xi)
    J = sum([dbasis[:,i]*geometry[i]' for i=1:length(geometry)])
    grad = inv(J)*dbasis
    gradf = sum([grad[:,i]*increment[i]' for i=1:length(increment)])'
    return gradf
end

