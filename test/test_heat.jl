# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# unit tests for heat equations

module HeatTests  # always wrap tests to module ending with "Tests"

using JuliaFEM.Test  # always use JuliaFEM.Test, not Base.Test

using JuliaFEM: HeatEquation
using JuliaFEM: Seg2, Quad4, DC2D4, DC2D2, Assembly, assemble!

function test_one_element()  # always start test function with name test_

    # volume element
    element = Quad4([1, 2, 3, 4])
    element["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    element["temperature thermal conductivity"] = 6.0
    element["temperature load"] = [12.0, 12.0, 12.0, 12.0]
    element["density"] = 36.0

    # boundary element 
    boundary_element = Seg2([1, 2])
    boundary_element["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0]]
    # linear ramp from 1 to 6 in time 0 to 1
    boundary_element["temperature flux"] = (0.0, 0.0), (1.0, 6.0)

    # Set constant source f=12 with k=6. Accurate solution is
    # T=1 on free boundary, u(x,y) = -1/6*(1/2*f*x^2 - f*x)
    equation = convert(HeatEquation, element)
    #la = initialize_local_assembly()
    #calculate_local_assembly!(la, equation, "temperature")
    assembly = Assembly()
    assemble!(assembly, equation)
    fdofs = [1, 2]
    A = full(assembly.stiffness_matrix)
    b = full(assembly.force_vector)
    @test isapprox(A[fdofs, fdofs] \ b[fdofs], [1.0, 1.0])

    # Set constant flux g=6 on boundary. Accurate solution is
    # u(x,y) = x which equals T=1 on boundary.
    boundary_equation = convert(HeatEquation, boundary_element)
    empty!(assembly)

    time = 1.0
    assemble!(assembly, equation, time)
    info("after first element: $(length(assembly.force_vector.V))")
    info(full(assembly.force_vector)')
    assemble!(assembly, boundary_equation, time)
    info("after second element: $(length(assembly.force_vector.V))")
    info(full(assembly.force_vector)')
    #calculate_local_assembly!(la, boundary_equation, "temperature")
    #b = la.force_vector
    A = full(assembly.stiffness_matrix)
    b = full(assembly.force_vector)
    T = A[fdofs, fdofs] \ b[fdofs]
    info("T = $T")
    @test isapprox(T, [2.0, 2.0])  # always use @test to test things.

end

end
