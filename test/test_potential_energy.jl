# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElementTests

using JuliaFEM.Test

using JuliaFEM
using JuliaFEM: Equation, Quad4, IntegrationPoint, Assembly, assemble!,
                get_element, get_basis, grad, get_unknown_field_name,
                PlaneHeatProblem, Seg2, Problem, solve!,
                get_default_integration_points, Equation

abstract MyEquation <: Equation

function JuliaFEM.get_unknown_field_name(equation::MyEquation)
    return "temperature"
end

""" Diffusive heat transfer for 4-node bilinear element, with a nonlinear source term. """
type DC2D4NL <: MyEquation
    element :: Quad4
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DC2D4NL)
    return (1, 4)
end

""" Nonlinear flux term. """
type DC2D2NL <: MyEquation
    element :: Seg2
    integration_points :: Vector{IntegrationPoint}
end

function Base.size(equation::DC2D2NL)
    return (1, 2)
end

function Base.convert(::Type{MyEquation}, element::Quad4)
    integration_points = get_default_integration_points(element)
    haskey(element, "temperature") || (element["temperature"] = 0.0 => zeros(4))
    DC2D4NL(element, integration_points)
end

function Base.convert(::Type{MyEquation}, element::Seg2)
    integration_points = JuliaFEM.line5()
    haskey(element, "temperature") || (element["temperature"] = 0.0 => zeros(2))
    DC2D2NL(element, integration_points)
end


""" Calculate a potential Π = Wint - Wext of system. """
function JuliaFEM.get_potential_energy(equation::DC2D4NL, ip, time; variation=nothing)
    element = get_element(equation)
    basis = get_basis(element)
    k = basis("temperature thermal conductivity", ip, time)
    f = basis("temperature load", ip, time)
    T = basis("temperature", ip, time, variation)
    c = basis("temperature nonlinearity coefficient", ip, time)
    gradT = grad(basis)("temperature", ip, time, variation)
    Wint = (k + c*T) * 1/2*vecdot(gradT, gradT)
    Wext = f*T
    return Wint - Wext
end

function JuliaFEM.get_potential_energy(equation::DC2D2NL, ip, time; variation=nothing)
    element = get_element(equation)
    basis = get_basis(element)
    T = basis("temperature", ip, time, variation)[1]
    T_ext = basis("temperature external", ip, time)[1]
    coeff = basis("temperature coefficient", ip, time)[1]
    q0 = coeff*(T_ext^4 - T^4)
    Wint = 0.0
    Wext = q0*T
    W = Wint - Wext
    return W
end

function test_potential_energy_method()

    # create model -- start
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    element["temperature thermal conductivity"] = 6.0
    element["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element["temperature nodal load"] = [3.0, 3.0, 0.0, 0.0]
    element["temperature nonlinearity coefficient"] = 6.0
    equation = convert(MyEquation, element)
    # create model -- end

    solve!(equation, [1, 2], 0.0)
    basis = get_basis(element)
    temp = basis("temperature", [0.0, -1.0], 0.0)
    err = temp - 2/3
    info("error: $err")
    @test isapprox(err, 0.0)
end


type TestProblem <: Problem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Vector{MyEquation}
end

function TestProblem(equations=[])
    TestProblem("temperature", 1, equations)
end

function test_potential_energy_method_2()

    # create model -- start
    N = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element1 = Quad4([1, 2, 3, 4])
    element1["geometry"] = Vector[N[1], N[2], N[3], N[4]]
    element1["temperature thermal conductivity"] = 6.0
    element1["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element1["temperature nonlinearity coefficient"] = [0.0, 0.0, 0.0, 0.0]

    element2 = Seg2([1, 2])
    element2["geometry"] = Vector[N[1], N[2]]
    element2["temperature coefficient"] = 3.0e-8 # ~ 5.7e-8 * 0.5
    element2["temperature external"] = 100.0
    # create model -- end

    problem = TestProblem()
    push!(problem, element1)
    push!(problem, element2)
    solve!(problem, [1, 2], 0.0)

    basis = get_basis(element1)
    temp = basis("temperature", [0.0, -1.0], 0.0)
    err = temp - 0.5
    info("error: $err")
    @test isapprox(err, 0.0, atol=1.0e-6)
    
    # @test isapprox(temp, 2.93509690572300E+00)  # tested using Code Aster
end

end
