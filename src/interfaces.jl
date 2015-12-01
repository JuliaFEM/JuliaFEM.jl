# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using JuliaFEM
using JuliaFEM.API: Model

function solve!(model::Model, case_name::ASCIIString, time::Float64)
    element_ids = keys(model.elements)
    all_eles = model.elements
    case = model.load_cases[case_name]
    bcs = case.boundary_conditions
    nodes = model.nodes
    field_problem = JuliaFEM.Core.(case.problem)()
    # luodaan core elementit
    # for each in elements
    # end

    # Lisätään Neumann:nin reunaehdot ja listään field probleemaan
    # for each in neumann
    # end

    # lisätään Dirichlet:in reunaehdot ja lisätään reunaehtoon
    # for each in dirichlet
    # end
    #
    # SOLVE !

    # Adding elements to field problem
 #   for element in all_eles
 #       el_type = element.element_type
 #       el_id = element.id
 #       mat = element.material
 #       conn = element.connectivity
 #       core_element = JuliaFEM.Core.(el_type)(conn)
 #       core_element["geometry"] = map(x->nodes[x], conn)
 #       for each in keys(mat)
 #           core_element[each] = mat[each]
 #       end
 #       # TODO ! Lisätään Neumann:nin reunaehdot ennen kuin
 #       # lisätään probleemaan
 #       push!(field_problem, core_element)
 #       model.elements[el_id].fields = core_element.fields
 #   end
 #   # käydaan läpi Dirichlet:in reunaehdot  
 #   for bc in bcs
 #       elset_name = bc.set_name
 #       elset = model.elsets[elset_name]
 #       for element in elset.elements
 #           element_id = element.id
 #           el_type = element.element_type
 #           core_element = JuliaFEM.Core.(el_type)(element.connectivity)
 #           
 #           model.elements[element_id].fields = core_element.fields
 #       println(core_element)
    end
end

function foo()
    return "bar"
end
