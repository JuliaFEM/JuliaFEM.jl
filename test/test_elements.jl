# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElementTests

using JuliaFEM.Test

using JuliaFEM: Element, Basis, Field, FieldSet, FunctionSpace, test_element

""" Prototype element

This should always pass test_element if everything is ok.
"""
type MockElement <: Element
    connectivity :: Vector{Int}
    basis :: Basis
    fields :: FieldSet
end

function MockElement(connectivity)

    h(xi) = 1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'

    dh(xi) = 1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]

    basis = Basis(h, dh)
    MockElement(connectivity, basis, Dict())
end

Base.size(element::Type{MockElement}) = (2, 4)

"""test test_element against mock element"""
function test_mockelement()
    test_element(MockElement)
end

""" test adding fieldsets and fields to element"""
function test_add_fields_to_element()
    el = MockElement([1, 2, 3, 4])
    #geometry = Field([0.0, 0.0, 0.0, 0.0])
    el["geometry"] = Field([0.0, 0.0, 0.0, 0.0])
    @test el["geometry"][1].time == 0.0
    @test last(el["geometry"]) == [0.0, 0.0, 0.0, 0.0]
    el["geometry"] = Field(Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    @test last(el["geometry"])[3] == [1.0, 1.0]
    el["geometry"] = Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    @test last(el["geometry"])[3] == [1.0, 1.0]
    el["geometry"] = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]'
    @test last(el["geometry"])[3] == [1.0, 1.0]
    el["geometry"] = (0.0, [0.0, 0.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0, 1.0])
    field = el["geometry"]
    @test length(field) == 2  # two time steps
    el["boundary flux"] = (0.0, 0.0), (1.0, 6.0)
end

function test_add_fields_to_element_2()
    el = MockElement([1, 2, 3, 4])
    el["data"] = (0.0 => [1, 2], 1.0 => [2, 3])
    @test length(el["data"]) == 2
    @test el["data"][1].time == 0.0
    @test el["data"][2].time == 1.0
    @test last(el["data"][1]) == [1, 2]
    @test last(el["data"][2]) == [2, 3]
end

function test_add_data_to_element_using_push()
    el = MockElement([1, 2, 3, 4])
    el["data"] = [0, 0, 0, 0]
   
    push!(el["data"], [1, 2, 3, 4])
    @test length(el["data"]) == 1
    @test length(el["data"][1]) == 2
    @test el["data"][1].time == 0.0

    push!(el["data"], 1.0 => [2, 3, 4, 5])  # creates new timestep at t=1.0
    push!(el["data"], [3, 4, 5, 6])  # adds new increment data to last timestep
    @test length(el["data"]) == 2
    @test length(el["data"][2]) == 2
    @test el["data"][2].time == 1.0
    
end

#=
facts("interpolation of fields in some function space") do

    el = MockElement([1, 2, 3, 4])
    fieldset1 = FieldSet("geometry", [Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])])
    fieldset2 = FieldSet("constant scalar field", [Field(0.0, 1.0)])
    fieldset3 = FieldSet("scalar field", [Field(0.0, [1.0, 2.0, 3.0, 4.0])])
    fieldset4 = FieldSet("vector field 1", [Field(0.0, Vector[[1.0], [2.0], [3.0], [4.0]])])
    fieldset5 = FieldSet("vector field 2", [Field(0.0, Vector[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])])
    fieldset6 = FieldSet("vector field 3", [Field(0.0, Vector[[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]])])
    fieldset7 = FieldSet("tensor field 1", [Field(0.0, Matrix[[1.0 5.0; 9.0 13.0], [2.0 6.0; 10.0 14.0], [3.0 7.0; 11.0 15.0], [4.0 8.0; 12.0 16.0]])])

    element["geometry"] = fieldset1
    element["constant scalar field"] = fieldset2
    element["scalar field"] = fieldset3
    element["vector field 1"] = fieldset4
    element["vector field 2"] = fieldset5
    element["vector field 3"] = fieldset6
    element["tensor field 1"] = fieldset7

    xi = [0.0, 0.0]
    t = 0.0
    u = FunctionSpace(element)
    v = FunctionSpace(element)

    @fact v("constant scalar field", xi, t) --> 1.0
    @fact v("scalar field", xi, t) --> 1/4*(1+2+3+4)
    @fact v("vector field 1", xi, t) --> [1/4*(1+2+3+4)]
    @fact v("vector field 2", xi, t) --> 1/4*[1+2+3+4, 5+6+7+8]
    @fact v("vector field 3", xi, t) --> 1/4*[1+2+3+4, 5+6+7+8, 9+10+11+12]
    @fact v("tensor field 1", xi, t) --> 1/4*[1+2+3+4 5+6+7+8; 9+10+11+12 13+14+15+16]
end
=#

end
