# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# 2d poisson problem with known analytical solution
# from FENiCS tutorial, u(x,y) = 1 + x² + 2y² on [0x1]×[0,1]
# and u₀(x,y) = 1 + x² + 2y², f(x,y) = -6

mesh_file = @__DIR__()*"/testdata/primitives.med"
mesh = aster_read_mesh(mesh_file, "UNITSQUARE_6X4")

square = Problem(PlaneHeat, "unit square, 6x4 triangular mesh", 1)
square_elements = create_elements(mesh, "UNITSQUARE")
update!(square_elements, "thermal conductivity", 1.0)
update!(square_elements, "heat source", -6.0)
add_elements!(square, square_elements)

bc = Problem(Dirichlet, "u₀(x,y) = 1 + x² + 2y²", 1, "temperature")
bc_elements = create_elements(mesh, "FACE1", "FACE2", "FACE3", "FACE4")
function u0(element, ip, time)
    x, y = element("geometry", ip, time)
    return 1 + x^2 + 2*y^2
end
update!(bc_elements, "temperature 1", u0)
add_elements!(bc, bc_elements)

analysis = Analysis(Linear)
add_problems!(analysis, square, bc)
run!(analysis)

T_fem = Float64[]
T_acc = Float64[]
for (nid, Xi) in square("geometry", 0.0)
    push!(T_fem, square("temperature", Xi, 0.0)[1])
    push!(T_acc, 1.0 + Xi[1]^2 + 2*Xi[2]^2)
end

@test maximum(abs.(T_fem-T_acc)) < 1.0e-12

# gradient of field is
X = [0.5, 0.5]
gradT1 = [2*X[1] 4*X[2]]
gradT2 = square("temperature", X, 0.0, Val{:Grad})
@debug("gradT1 = $gradT1, gradT2 = $gradT2")
# [1.1666666666666625 1.5000000000000018] quite big difference ..?
@test isapprox(gradT1, gradT2; rtol=25.0e-2)
