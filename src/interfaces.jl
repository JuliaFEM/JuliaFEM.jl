# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using JuliaFEM
using JuliaFEM.Core
using JuliaFEM.API: Model

"""
Function for creating solver and all the necessary components
for the calculation
"""
function get_solver(model::Model, case_name::ASCIIString, time::Float64)
    case = model.load_cases[case_name]
    
    # Create core elements
    core_elements = create_core_elements(model)
    
    # Add Neumann boundary conditions to elements
    add_neumann_bcs!(model, case, core_elements)

    # Create Dirichlet boundary conditions
    dirichlet_arr = create_dirichlet_bcs(model, case, core_elements)

    # Create solver 
    solver = create_solver(model, case,
                            core_elements, dirichlet_arr)
	return solver
end

"""
"""
function create_solver(model, case, core_elements, dirichlet_arr)
    field_problem = JuliaFEM.Core.(case.problem)()
    all_elsets = model.elsets
    
    # Pushing all the defined element sets into the calculation 
    for element_set in case.sets
        el_ids = all_elsets[element_set].elements
        for each in el_ids
            push!(field_problem, core_elements[each])
        end
    end

    # Creating the solver and pushing problems and 
    # boundary conditions
#    if case.solver == :LinearSolver
#        solver = JuliaFEM.Core.(case.solver)(field_problem,
#                                             dirichlet_arr...) 
#    else
	    solver = JuliaFEM.Core.(case.solver)()
	    push!(solver, field_problem)
	    push!(solver, dirichlet_arr...)
#     end
    return solver
end

"""
"""
function create_dirichlet_bcs(model, case, core_elements)
    dirichlet_arr = Any[]
    dirichlet_bcs = case.dirichlet_boundary_conditions
    elsets = model.elsets
    field_problem = JuliaFEM.Core.(case.problem)()
    for each in dirichlet_bcs
        set_name = each.set_name 
        value = each.value
        problem = JuliaFEM.Core.DirichletProblem(
           JuliaFEM.Core.get_unknown_field_name(field_problem),
           JuliaFEM.Core.get_unknown_field_dimension(field_problem))
        set_for_bc = each.set_name
        set_ids = elsets[set_for_bc]
        bc = each.value
        for el_id in set_ids.elements
            core_element = core_elements[el_id]
            core_element[bc[1]] = bc[2]
            push!(problem, core_element)
        end
        push!(dirichlet_arr, problem)
    end
    dirichlet_arr
end

"""
"""
function create_core_elements(model)
    core_elements = Dict()
    nodes = model.nodes
    all_elements = model.elements
    element_ids = keys(all_elements)
    for el_id in element_ids
        element = all_elements[el_id]
        el_type = element.element_type
        el_id = element.id
        mat = element.material
        conn = element.connectivity
        core_element = JuliaFEM.Core.(el_type)(conn)
        core_element["geometry"] = map(x->nodes[x].coords, conn)
        for each in keys(mat.scalar_data)
            core_element[each] = mat.scalar_data[each]
        end
        core_elements[el_id] = core_element
        model.elements[el_id].results = core_element
    end
    core_elements
end

"""
"""
function add_neumann_bcs!(model, case, core_elements)
    neumann_bcs = case.neumann_boundary_conditions
    elsets = model.elsets
    for each in neumann_bcs
        set_for_bc = each.set_name
        set_ids = elsets[set_for_bc]
        bc = each.value
        for el_id in set_ids.elements
            core_element = core_elements[el_id]
            core_element[bc[1]] = bc[2]
        end
        if !(set_for_bc in case.sets)
            push!(case.sets, set_for_bc)
        end
    end
end

"""
"""
function solve!(model::Model, case_name::ASCIIString, time::Float64)
    # Create solver
	solver = get_solver(model, case_name, time)

    # Solve problem at given time 
    solver(time)
end
