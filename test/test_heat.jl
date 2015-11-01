# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# unit tests for heat equations

module HeatTests  # always wrap tests to module ending with "Tests"

using JuliaFEM.Test  # always use JuliaFEM.Test, not Base.Test

using JuliaFEM: Seg2, Quad4, Field, FieldSet, DC2D4,
                initialize_local_assembly, calculate_local_assembly!,
                DC2D2


"tests on [0x1]x[0x1] domain"
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
    equation = DC2D4(element)
    la = initialize_local_assembly()
    calculate_local_assembly!(la, equation, "temperature")
    fdofs = [1, 2]
    A = la.stiffness_matrix
    b = la.force_vector
    @test isapprox(A[fdofs, fdofs] \ b[fdofs], [1.0, 1.0])

    # Set constant flux g=6 on boundary. Accurate solution is
    # u(x,y) = x which equals T=1 on boundary.
    boundary_equation = DC2D2(boundary_element);

    calculate_local_assembly!(la, boundary_equation, "temperature")
    b = la.force_vector
    @test isapprox(A[fdofs, fdofs] \ b[fdofs], [1.0, 1.0])  # always use @test to test things.

end

end
