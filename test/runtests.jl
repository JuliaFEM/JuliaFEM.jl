using JuliaFEM
using Base.Test

function test_create_model()
    m = new_model()
    fn = fieldnames(m)
    @assert :model in fn
    @assert :nodes in fn
    @assert :elements in fn
    @assert :element_nodes in fn
    @assert :element_gauss_points in fn
end

function test_add_field()
    m = new_model()
    field = new_field(m.elements, "color")
    @assert "color" in keys(m.elements)
end

function test_get_field()
    m = new_model()
    field = new_field(m.elements, "color")
    field[1] = "red"  # set element 1 field value to "red"
    field2 = get_field(m.elements, "color") # get color field
    @assert field2[1] == "red"
end

function test_get_field_if_it_doesnt_exist()
    m = new_model()
    # FIXME: how to test exception?
    # @assert "throw error when" field = get_field(m.elements, "temperature")
    field = get_field(m.elements, "temperature"; create_if_doesnt_exist=true)
    field[1] = 173
    field2 = get_field(m.elements, "temperature")
    @assert field2[1] == 173
end


# write your own tests here
# @test 1 == JuliaFEM.test()
# @test_approx_eq 1.0 1.0
