using JuliaFEM
using FactCheck
using Logging
@Logging.configure(level=DEBUG)


facts("Testing if somebody used print, println(), @sprint in src directory") do
  # TODO: make better reqular expression. Currently it will match all print words
  lines_with_print = Dict()
  src_dir = "../src"
  src = readdir(src_dir)
  for file_name in src
    fil = open(joinpath(src_dir,file_name),"r")
    for (line_number,line) in enumerate(readlines(fil))
      if ismatch(r"print",line)
        lines_with_print[file_name * ":$line_number"] = "print found in line: $line_number"
      end
    end
    close(fil)
  end
  @fact lines_with_print => isempty "Instead of println() use Logging.jl package"
end

facts("One test to test code coverage") do
   m = new_model()
    #field = new_field(m.elements, "color")
    #field[1] = "red"  # set element 1 field value to "red"
    #field2 = get_field(m.elements, "color") # get color field
    @fact "red" => "red"
end


#function test_get_element()
#    m = new_model()
#    nodes = Dict(1 => [0.0, 0.0, 0.0],
#                 2 => [1.0, 0.0, 0.0],
#                 3 => [0.0, 1.0, 0.0],
#                 4 => [0.0, 0.0, 1.0])
#    add_nodes(m, nodes)
#    el1 = Dict("element_type" => 0x6, "node_ids" => [1, 2, 3, 4])
#    el2 = Dict("element_type" => 0x6, "node_ids" => [4, 3, 2, 1])
#    elements = Dict(1 => el1, 2 => el2)
#    #println(elements)
#    add_elements(m, elements)
#    #println(m)
#    # get elements by element id
#    els = get_elements(m, [1])
#    #println(els)
#    @test length(els) == 1
#    @test 1 in keys(els)
#    @test els[1]["element_type"] == 0x6
#    @test els[1]["node_ids"] == [1, 2, 3, 4]
#end
#test_get_element()



# include("test_xdmf.jl")
# include("solver_tests/test_elasticity_solver.jl")
# include("test_model.jl")

include("test_elasticity_solver.jl")
