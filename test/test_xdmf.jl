using JuliaFEM
using Base.Test


function test_write_to_xdfm()
    m = new_model()
    nodes = Dict(1 => [0.0, 0.0, 0.0],
                 2 => [1.0, 0.0, 0.0],
                 3 => [0.0, 1.0, 0.0],
                 4 => [0.0, 0.0, 1.0])
    add_nodes(m, nodes)
    el1 = Dict("element_type" => 0x6, "node_ids" => [1, 2, 3, 4])
    elements = [el1]
    add_elements(elements)

    # create new field "temperature" for nodal points
    f = new_field(m.nodes, "temperature")
    f[1] = 0.1
    f[2] = 0.2
    f[3] = 0.3
    # do NOT write to fourth node.

    # export model to xdmf
    fn = tempname()
    println("temp file: ", fn)
    time = 123
    write_xdmf(m, fn, time; fields = ["temperature"])

    d = h5open(fn)
    @test d[123]["temperature"][:] = [0.1, 0.2, 0.3]

end
test_write_to_xdfm()