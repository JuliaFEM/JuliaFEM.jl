# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using JuliaFEM: Element, Basis, FieldSet

""" Prototype element

This should always pass test_element if everything is ok.
"""
type MockElement <: Element
    connectivity :: Array{Int, 1}
    basis :: Basis
    fields :: Dict{Symbol, FieldSet}
end
function MockElement(connectivity)
    h(xi) = [
        (1-xi[1])*(1-xi[2])/4
        (1+xi[1])*(1-xi[2])/4
        (1+xi[1])*(1+xi[2])/4
        (1-xi[1])*(1+xi[2])/4]
    dh(xi) = [
        -(1-xi[2])/4.0   -(1-xi[1])/4.0
         (1-xi[2])/4.0   -(1+xi[1])/4.0
         (1+xi[2])/4.0    (1+xi[1])/4.0
        -(1+xi[2])/4.0    (1-xi[1])/4.0]
    basis = Basis(h, dh)
    MockElement(connectivity, basis, Dict())
end
JuliaFEM.get_number_of_basis_functions(el::Type{MockElement}) = 4
JuliaFEM.get_element_dimension(el::Type{MockElement}) = 2


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

