# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using JuliaFEM: Element, Basis, Field, FieldSet, FunctionSpace

""" Prototype element

This should always pass test_element if everything is ok.
"""
type MockElement <: Element
    connectivity :: Array{Int, 1}
    basis :: Basis
    fields :: Dict{ASCIIString, FieldSet}
end
function MockElement(connectivity)

    h(xi) = 1/4*[(1-xi[1])*(1-xi[2])   (1+xi[1])*(1-xi[2])   (1+xi[1])*(1+xi[2])   (1-xi[1])*(1+xi[2])]

    dh(xi) = 1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]

    basis = Basis(h, dh)
    MockElement(connectivity, basis, Dict())
end
Base.size(element::Type{MockElement}) = (2, 4)

using JuliaFEM: test_element
facts("test test_element against mock element") do
    test_element(MockElement)
end


facts("test adding fieldsets and fields to element") do
    el = MockElement([1, 2, 3, 4])
    fieldset = JuliaFEM.FieldSet("geometry")
    field1 = JuliaFEM.Field(0.0, [0.0, 0.0, 0.0, 0.0])
    push!(fieldset, field1)
    field2 = JuliaFEM.Field(1.0, [1.0, 1.0, 1.0, 1.0])
    push!(fieldset, field2)
    push!(el, fieldset)
    fields = el["geometry"]
    @fact length(fields) --> 2
    @fact fields[1] --> field1
    @fact fields[2] --> field2
end

facts("interpolation of fields in some function space") do

    element = MockElement([1, 2, 3, 4])
    fieldset1 = FieldSet("geometry", [Field(0.0, Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])])
    fieldset2 = FieldSet("constant scalar field", [Field(0.0, 1.0)])
    fieldset3 = FieldSet("scalar field", [Field(0.0, [1.0, 2.0, 3.0, 4.0])])
    fieldset4 = FieldSet("vector field 1", [Field(0.0, Vector[[1.0], [2.0], [3.0], [4.0]])])
    fieldset5 = FieldSet("vector field 2", [Field(0.0, Vector[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])])
    fieldset6 = FieldSet("vector field 3", [Field(0.0, Vector[[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]])])
    fieldset7 = FieldSet("tensor field 1", [Field(0.0, Matrix[[1.0 5.0; 9.0 13.0], [2.0 6.0; 10.0 14.0], [3.0 7.0; 11.0 15.0], [4.0 8.0; 12.0 16.0]])])

    push!(element, fieldset1)
    push!(element, fieldset2)
    push!(element, fieldset3)
    push!(element, fieldset4)
    push!(element, fieldset5)
    push!(element, fieldset6)
    push!(element, fieldset7)

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

