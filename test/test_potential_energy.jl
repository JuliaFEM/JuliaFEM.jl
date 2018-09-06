# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

abstract type HeatProblem <: AbstractProblem 

end

function HeatProblem(dim::Int=1, elements=[])
    return Problem{HeatProblem}(dim, elements)
end

function get_unknown_field_name(::Type{P}) where P<:HeatProblem
    return "temperature"
end

function get_unknown_field_type(::Type{P}) where P<:HeatProblem
    return Float64
end

""" Calculate a potential Π = Wint - Wext of system. """
function get_potential_energy(problem::Problem{HeatProblem}, element::Element{Quad4}, ip::IntegrationPoint, time::Number; variation=nothing)
    k = element("temperature thermal conductivity", ip, time)
    f = element("temperature load", ip, time)
    T = element("temperature", ip, time, variation)
    c = element("temperature nonlinearity coefficient", ip, time)
    gradT = element("temperature", ip, time, Val{:grad}, variation)
    Wint = (k + c*T) * 1/2*vecdot(gradT, gradT)
    Wext = f*T
    W = Wint - Wext
    J = get_jacobian(element, ip, time)
    return W*det(J)
end

function get_potential_energy(problem::Problem{HeatProblem}, element::Element{Seg2}, ip::IntegrationPoint, time::Number; variation=nothing)
    T = element("temperature", ip, time, variation)[1]
    T_ext = element("temperature external", ip, time)[1]
    coeff = element("temperature coefficient", ip, time)[1]
    q0 = coeff*(T_ext^4 - T^4)
    Wint = 0.0
    Wext = q0*T
    W = Wint - Wext
    J = get_jacobian(element, ip, time)
    return W*norm(J)
end

function test_potential_energy_method()

    # create model -- start
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    element["temperature thermal conductivity"] = 6.0
    element["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element["temperature nodal load"] = [3.0, 3.0, 0.0, 0.0]
    element["temperature nonlinearity coefficient"] = 6.0
    element["temperature"] = (0.0 => zeros(Float64, 4))
    problem = HeatProblem()
    push!(problem, element)
    # create model -- end

    solve!(problem, [1, 2], 0.0)
    temp = element("temperature", [0.0, -1.0], 0.0)
    err = temp - 2/3
    @info("error: $err")
    @test isapprox(err, 0.0)
end


function test_potential_energy_method_2()

    # create model -- start
    N = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element1 = Quad4([1, 2, 3, 4])
    element1["geometry"] = Vector[N[1], N[2], N[3], N[4]]
    element1["temperature thermal conductivity"] = 6.0
    element1["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element1["temperature nonlinearity coefficient"] = [0.0, 0.0, 0.0, 0.0]
    element1["temperature"] = (0.0 => zeros(Float64, 4))

    element2 = Seg2([1, 2])
    element2["geometry"] = Vector[N[1], N[2]]
    element2["temperature coefficient"] = 3.0e-8 # ~ 5.7e-8 * 0.5
    element2["temperature external"] = 100.0
    element2["temperature"] = (0.0 => zeros(Float64, 2))
    # create model -- end

    problem = HeatProblem()
    push!(problem, element1)
    push!(problem, element2)
    solve!(problem, [1, 2], 0.0)

    temp = element1("temperature", [0.0, -1.0], 0.0)
    err = temp - 0.5
    @info("error: $err")
    @test isapprox(err, 0.0, atol=1.0e-6)
    
    # @test isapprox(temp, 2.93509690572300E+00)  # tested using Code Aster
end

