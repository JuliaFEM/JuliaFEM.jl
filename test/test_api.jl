# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module APITests

using JuliaFEM
using JuliaFEM.Test

function test_basic()
    # basic workflow, copied from test_solver.jl
    model = Model("my calculation model")

    # create nodes
    n1 = Node(1, [0.0, 0.0])
    n2 = Node(2, [1.0, 0.0])
    n3 = Node(3, [1.0, 1.0])
    n4 = Node(4, [0.0, 1.0])

    # create elements
    e1 = Element{Quad4}([n1, n2, n3, n4])
    e2 = Element{Seg2}([n1, n2])

    # material properties
    material = Material("my material")
    material["temperature thermal conductivity"] = 6.0
    material["density"] = 36.0

    # create element set body
    body = ElementSet("body")
    push!(body, e1, e2)

    # add element set to model
    add_element_set(model, body)

    # assign material to elements in element set body
    set_material(model, material, body)

    # boundary conditions


end

end
