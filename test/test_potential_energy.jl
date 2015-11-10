# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElementTests

using JuliaFEM.Test

using JuliaFEM
using JuliaFEM: Equation, Quad4, IntegrationPoint, initialize_local_assembly,
                get_element, get_basis, grad, calculate_local_assembly!,
                PlaneHeatProblem, Seg2, HeatEquation, Problem, solve!


""" Diffusive heat transfer for 4-node bilinear element, with a nonlinear source term. """
type DC2D4NL <: Equation
    element :: Quad4
    integration_points :: Array{IntegrationPoint, 1}
end

function DC2D4NL(element::Quad4, initial_temperature=zeros(4))
    integration_points = [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)]
    if !haskey(element, "temperature")
        element["temperature"] = initial_temperature
    end
    DC2D4NL(element, integration_points)
end

function Base.size(equation::DC2D4NL)
    return (1, 4)
end

""" Nonlinear flux term. """
type DC2D2NL <: Equation
    element :: Seg2
    integration_points :: Array{IntegrationPoint, 1}
end

function DC2D2NL(element::Seg2, initial_temperature=zeros(2))
    #integration_points = [
    #    IntegrationPoint([0.0], 2.0)]
    integration_points = JuliaFEM.line5()
    if !haskey(element, "temperature")
        element["temperature"] = initial_temperature
    end
    DC2D2NL(element, integration_points)
end

function Base.size(equation::DC2D2NL)
    return (1, 2)
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
    #Wint = k*1/2*vecdot(gradT, gradT)
    Wext = f*T
    #Wext = 0.0
    return Wint - Wext
end

function JuliaFEM.has_potential_energy(eq::DC2D4NL)
    return true
end

function JuliaFEM.get_potential_energy(equation::DC2D2NL, ip, time; variation=nothing)
    element = get_element(equation)
    basis = get_basis(element)
    T = basis("temperature", ip, time, variation)[1]
    Wint = 0.0
    sig = 5.7e-8
    eps = basis("emissivity", ip, time)[1]
    T_ext = basis("temperature external", ip, time)[1]
    q0 = eps*sig*((T_ext+273.15)^4 - (T+273.15)^4)
    Wext = q0*T
    W = Wint - Wext
    return W
end

function JuliaFEM.has_potential_energy(eq::DC2D2NL)
    return true
end

function test_potential_energy_method()

    # create model -- start
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    element["temperature thermal conductivity"] = 6.0
    element["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element["temperature nodal load"] = [3.0, 3.0, 0.0, 0.0]
    element["temperature nonlinearity coefficient"] = [6.0, 6.0, 6.0, 6.0]
    equation = DC2D4NL(element)
    # create model -- end

    la = initialize_local_assembly() # create workspace for local matrices
    T = zeros(4) # create workspace for solution vector
    dT = zeros(4) # 
    fd = [1, 2] # free dofs
    tic()
    # start loops, in principle solve ∂r(u)/∂uΔu = -r(u) and update.
    for i=1:10
        calculate_local_assembly!(la, equation, "temperature")  # calculate local matrices
        dT[fd] = la.stiffness_matrix[fd,fd] \ la.force_vector[fd]
        T += dT
        push!(element["temperature"], T) # add new increment to model
        info("T = $T")
        @printf("increment %2d, |du| = %8.5f\n", i, norm(dT))
        err = last(element["temperature"])[1] - 2/3
        isapprox(err, 0.0) && break
    end
    toc()
    err = last(element["temperature"])[1] - 2/3
    info("error: $err")
    @test isapprox(err, 0.0)
end

type TestProblem <: Problem
    unknown_field_name :: ASCIIString
    unknown_field_dimension :: Int
    equations :: Array{Equation, 1}
    element_mapping :: Dict{DataType, DataType}
end

function TestProblem(equations=[])
    element_mapping = Dict(
        Quad4 => DC2D4NL,
        Seg2 => DC2D2NL)
    TestProblem("temperature", 1, equations, element_mapping)
end

function test_potential_energy_method_2()

    # create model -- start
    N = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[N[1], N[2], N[3], N[4]]
    element["temperature thermal conductivity"] = 6.0
    element["temperature load"] = [0.0, 0.0, 0.0, 0.0]
    element["temperature nonlinearity coefficient"] = [0.0, 0.0, 0.0, 0.0]
    #equation1 = DC2D4NL(element, initial_temperature=ones(4))
    equation1 = DC2D4NL(element)

    boundary_element = Seg2([1, 2])
    boundary_element["geometry"] = Vector[N[1], N[2]]
    boundary_element["emissivity"] = 0.5
    boundary_element["temperature external"] = 10.0
    #equation2 = DC2D2NL(boundary_element, initial_temperature=ones(4))
    equation2 = DC2D2NL(boundary_element)
    # create model -- end

    element["temperature"] = ones(4)
    boundary_element["temperature"] = ones(2)

    equations = [equation1, equation2]
    la = initialize_local_assembly() # create workspace for local matrices
    T = zeros(4) # create workspace for solution vector
    dT = zeros(4) # 
    fd = [1, 2] # free dofs
    info("equation 1")
    calculate_local_assembly!(la, equation1, "temperature")
    info("stiffness matrix: $(la.stiffness_matrix)")
#   info("force vector: $(la.force_vector)")
    info("equation 2")
    calculate_local_assembly!(la, equation2, "temperature")
#   info("stiffness matrix: $(la.stiffness_matrix)")
    info("force vector: $(la.force_vector)")

    info("Creating problem")
    #problem = PlaneHeatProblem("temperature", 1, equations, Dict())
    problem = TestProblem(equations)

    free_dofs = [1, 2]
    tic()
    solve!(problem, free_dofs; max_iterations=10)
    toc()
    temp = get_basis(boundary_element)("temperature", [0.0])[1]
    info("temperature = $temp")
    #err = last(element["temperature"])[1] - 2/3
    #info("error: $err")
    # 0.3888756709834147 tulee jostakin syysta...
    # tai -0.39411350336960116
    info(boundary_element["temperature"])
    @test isapprox(temp, 2.93509690572300E+00)  # tested using Code Aster
end

end
