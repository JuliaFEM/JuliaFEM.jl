using JuliaFEM
using Base.Test


function test_create_model()
    m = new_model()
    fn = fieldnames(m)
    @test :model in fn
    @test :nodes in fn
    @test :elements in fn
    @test :element_nodes in fn
    @test :element_gauss_points in fn
end
test_create_model()


function test_add_field()
    m = new_model()
    field = new_field(m.elements, "color")
    @test "color" in keys(m.elements)
end
test_add_field()


function test_get_field()
    m = new_model()
    field = new_field(m.elements, "color")
    field[1] = "red"  # set element 1 field value to "red"
    field2 = get_field(m.elements, "color") # get color field
    @test field2[1] == "red"
end
test_get_field()


function test_get_field_if_it_doesnt_exist()
    m = new_model()
    # FIXME: how to test exception?
    # @assert "throw error when" field = get_field(m.elements, "temperature")
    field = get_field(m.elements, "temperature"; create_if_doesnt_exist=true)
    field[1] = 173
    field2 = get_field(m.elements, "temperature")
    @test field2[1] == 173
end
test_get_field_if_it_doesnt_exist()


function test_add_and_get_nodes()
    m = new_model()
    nodes = Dict(1 => [0.0, 0.0, 0.0],
                 2 => [1.0, 0.0, 0.0],
                 3 => [0.0, 1.0, 0.0],
                 4 => [0.0, 0.0, 1.0])
    add_nodes(m, nodes)
    subset = get_nodes(m, [2, 3])
    @test length(subset) == 2
    @test subset[2] == [1.0, 0.0, 0.0]
    @test subset[3] == [0.0, 1.0, 0.0]
end
test_add_and_get_nodes()



function test_print()
  lines_with_print = Dict()
  #src = readdir("/src")
  src_dir = "/home/travis/build/ovainola/JuliaFEM"
  src = readdir(src_dir)
  for file_name in src
    fil = open(joinpath("src",file_name),"r")
    for line in readlines(fil)
      if ismatch(r"print",line)
        lines_with_print[file_name] = "print"
      end
    end
    close(fil)
  end
  @test lines_with_print == Dict()
end
test_print()



function test_add_element()
    m = new_model()
    nodes = Dict(1 => [0.0, 0.0, 0.0],
                 2 => [1.0, 0.0, 0.0],
                 3 => [0.0, 1.0, 0.0],
                 4 => [0.0, 0.0, 1.0])
    add_nodes(m, nodes)
    el1 = Dict("element_type" => 0x6, "node_ids" => [1, 2, 3, 4])
    el2 = Dict("element_type" => 0x6, "node_ids" => [4, 3, 2, 1])
    elements = Dict(1 => el1, 2 => el2)
    add_elements(m, elements)
    println(m)
    @test m.elements["element_type"][1] == 0x6
    @test m.elements["connectivity"][1] == [1, 2, 3, 4]
end
test_add_element()


function test_get_element()
    m = new_model()
    nodes = Dict(1 => [0.0, 0.0, 0.0],
                 2 => [1.0, 0.0, 0.0],
                 3 => [0.0, 1.0, 0.0],
                 4 => [0.0, 0.0, 1.0])
    add_nodes(m, nodes)
    el1 = Dict("element_type" => 0x6, "node_ids" => [1, 2, 3, 4])
    el2 = Dict("element_type" => 0x6, "node_ids" => [4, 3, 2, 1])
    elements = Dict(1 => el1, 2 => el2)
    #println(elements)
    add_elements(m, elements)
    #println(m)
    # get elements by element id
    els = get_elements(m, [1])
    #println(els)
    @test length(els) == 1
    @test 1 in keys(els)
    @test els[1]["element_type"] == 0x6
    @test els[1]["node_ids"] == [1, 2, 3, 4]
end
test_get_element()



# write your own tests here
# @test 1 == JuliaFEM.test()
# @test_approx_eq 1.0 1.0
