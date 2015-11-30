# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module ElementTests

using JuliaFEM.Test

using JuliaFEM: AbstractElement, Element, Field, FieldSet, test_element
import JuliaFEM: get_basis, get_dbasis
import Base: size

""" Prototype element

This should always pass test_element if everything is ok.
"""
abstract TestElement <: AbstractElement

function get_basis(::Type{TestElement}, xi::Vector{Float64})
    1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
end

function get_dbasis(::Type{TestElement}, xi::Vector{Float64})
    1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]
end

function size(::Type{TestElement})
    return (2, 4)
end

""" Return test element with some fields. """
function get_element()
    el = Element{TestElement}([1, 2, 3, 4])
    el["geometry"] = Vector{Float64}[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    el["temperature"] = (
        0.0 => [0.0, 0.0, 0.0, 0.0],
        1.0 => [1.0, 2.0, 3.0, 4.0])
    el["displacement"] = (
        0.0 => Vector{Float64}[[0.0,0.0], [0.0, 0.0], [0.0,0.0], [0.0,0.0]],
        1.0 => Vector{Float64}[[0.0,0.0], [1.0,-1.0], [2.0,3.0], [0.0,0.0]])
    return el
end

function test_mock_element()
    test_element(TestElement)
end

function test_add_fields_to_element()
    el = get_element()
    info(el.fields)
end

function test_interpolate()
    el = get_element()
    @test isapprox(el("geometry", [0.0, 0.0]), [0.5, 0.5])
    @test isapprox(el("geometry", [0.0, 0.0], 0.0), [0.5, 0.5])
    @test isapprox(el([0.0, 0.0]), [0.25 0.25 0.25 0.25])
    @test isapprox(el([0.0, 0.0], Val{:grad}), [-0.5 0.5 0.5 -0.5; -0.5 -0.5 0.5 0.5])
    gradT = el("temperature", [0.0, 0.0], 1.0, Val{:grad})
    info("gradT = $gradT")
    X = [0.5, 0.5]
    gradT_expected = [1-2*X[2] 3-2*X[1]]
    info("gradT(expected) = $gradT_expected")
    @test isapprox(gradT, gradT_expected)

#   @test isapprox(el("temperature", [0.0, 0.0], 0.5), 1/2*gradT_expected)

#   gradT = el("temperature", [0.0, 0.0], 0.5, Val{:grad})
#   info("gradT = $gradT")
#   @test isapprox(gradT, 1/2*gradT_expected)
end

end
