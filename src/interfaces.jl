# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using JuliaFEM
using JuliaFEM.Core
using JuliaFEM.API: Model

"""
This needs this a bit honing ... 
"""
function solve!(model::Model, case_name::ASCIIString, time::Float64)
    element_ids = keys(model.elements)
    all_elements = model.elements
    case = model.load_cases[case_name]
    neumann_bcs = case.neumann_boundary_conditions
    dirichlet_bcs = case.dirichlet_boundary_conditions
    nodes = model.nodes
    field_problem = JuliaFEM.Core.(case.problem)()
    core_elements = Dict()

    # luodaan core elementit
    for el_id in element_ids
        element = all_elements[el_id]
        el_type = element.element_type
        el_id = element.id
        mat = element.material
        conn = element.connectivity
        core_element = JuliaFEM.Core.(el_type)(conn)
        core_element["geometry"] = map(x->nodes[x], conn)
        for each in keys(mat.scalar_data)
            core_element[each] = mat.scalar_data[each]
        end
        core_elements[el_id] = core_element
        model.elements[el_id].results = core_element
    end
        
    # Add Neumann boundary conditions to elements
    for each in neumann_bcs
        set_for_bc = each.set_name
        set_ids = model.elsets[set_for_bc]
        bc = each.value
        for el_id in set_ids.elements
            core_element = core_elements[el_id]
            core_element[bc[1]] = bc[2]
        end
    end
    dirile_arr = Any[]
    # Create Dirichlet boundary conditions
    for each in dirichlet_bcs
        set_name = each.set_name 
        value = each.value
        problem = JuliaFEM.Core.DirichletProblem(
           JuliaFEM.Core.get_unknown_field_name(field_problem),
           JuliaFEM.Core.get_unknown_field_dimension(field_problem))
        set_for_bc = each.set_name
        set_ids = model.elsets[set_for_bc]
        bc = each.value
        for el_id in set_ids.elements
            core_element = core_elements[el_id]
            core_element[bc[1]] = bc[2]
            push!(problem, core_element)
        end
        push!(dirile_arr, problem)
    end

    # Push all the elements, where the field problem is solved into the field_problem
    element_set = case.sets
    el_ids = model.elsets[element_set].elements
    for each in el_ids
        push!(field_problem, core_elements[each])
    end

    # Creating the solver
    solver = JuliaFEM.Core.(case.solver)(field_problem, dirile_arr...) 

    # Solving 
    solver(time)
 end

