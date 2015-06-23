using JuliaFEM
using Base.Test

function test_one_element_solution()
    model = new_model()

    # 1. create mesh structure
    # add nodes to model
    nodes = Dict(1 => [0.0, 0.0], 2 => [10.0, 0.0], 3 => [10.0, 1.0], 4 => [0.0, 1.0])
    add_nodes(model, nodes)

    # 2. add elements to model
    const QUAD4 = 0x4
    element = Dict("element_type" => QUAD4, "node_ids" => [1, 2, 3, 4])
    elements = Dict()
    elements[1] = element
    add_elements(model, elements)

    # 3. add boundary conditions
    # add dirichlet boundary condition u=0
    boundaries = Dict("SUPPORT" => [1, 4])
    add_dirichlet_boundary_condition(model, boundaries)

    # add nodal load to node 3
    nodal_loads = Dict(3 => [-2, 0])
    add_nodal_loads(mode, nodal_loads)

    # 4. model is defined, solve it
    JuliaFEM.solvers.solve_elasticity!(model; parallel=false)

    # 5. extract results
    #write_xdmf(m, "results", ["displacement"])
    disp = get_field(model, "displacement")
    @assert_eq disp[2, 4] == -2.22224475
end

test_one_element_solution()
