# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Test


function get_basis()
    basis(xi) = 1/4*[
        (1-xi[1])*(1-xi[2])
        (1+xi[1])*(1-xi[2])
        (1+xi[1])*(1+xi[2])
        (1-xi[1])*(1+xi[2])]'
    dbasis(xi) = 1/4*[
        -(1-xi[2])    (1-xi[2])   (1+xi[2])  -(1+xi[2])
        -(1-xi[1])   -(1+xi[1])   (1+xi[1])   (1-xi[1])]
    return Basis(basis, dbasis)
end

function get_fieldset()
    X = Field(Vector[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    T = Field(
        (0.0, [0, 0, 0, 0]),
        (1.0, [1, 2, 3, 4]))
    u = Field(
        (0.5, Vector[[0.0, 0.0], [0.5, -0.5], [1.0, 1.5], [0.0, 0.0]]),
        (1.5, Vector[[0.0, 0.0], [1.5, -1.5], [3.0, 4.5], [0.0, 0.0]]))
    # FIXME: this is not working
    #fieldset = FieldSet("geometry" => X, "temperature" => T, "displacement" => u)
    fieldset = FieldSet()
    fieldset["geometry"] = X
    fieldset["temperature"] = T
    fieldset["displacement"] = u
    return fieldset
end

function test_create_symbolic_field()
    f = Field("temperature")
    @test isa(f, Field)
end

function test_evaluate_symbolic_field()
    T = Field("temperature")
    expr = Symbol(T)
    @test expr == :(temperature)
end

function test_simple_math()
    T = Field("temperature")
    eq = 1/2*T
    @test eq.expr == :(0.5*temperature)
end

function test_evaluate_expression()
    basis = get_basis()
    fieldset = get_fieldset()
    T = Field("temperature")
    expr = 1/2*T
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), 1/2*mean([1, 2, 3, 4]))
end

function test_evaluate_gradient_of_scalar_field()
    basis = get_basis()
    fieldset = get_fieldset()
    T = Field("temperature")
    expr = grad(T)
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), [0.0 2.0])
end

function test_evaluate_gradient_of_vector_field()
    basis = get_basis()
    fieldset = get_fieldset()
    u = Field("displacement")
    expr = grad(u)
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), [1.5 0.5; 1.0 2.0])
end

function test_evaluate_strain_rate()
    basis = get_basis()
    fieldset = get_fieldset()
    u = Field("displacement")
    #expr = grad(u)
    expr = Expression("1/2*(grad(diff(displacement)) + grad(diff(displacement))')")
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), [1.5 0.75; 0.75 2.0])
end

function test_evaluate_grad_diff()
    basis = get_basis()
    fieldset = get_fieldset()
    u = Field("displacement")
    #expr = grad(u)
    expr = Expression("grad(diff(displacement))")
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), [1.5 0.5; 1.0 2.0])
end

function test_grad_diff_simplification()
    basis = get_basis()
    fieldset = get_fieldset()
    u = Field("displacement")
    expr1 = diff(grad(u))
    expr2 = Expression("grad(diff(displacement))")
    info("expr1 = $expr1")
    info("expr2 = $expr2")
    @test expr1 == expr2
end

function test_evaluate_time_derivative()
    basis = get_basis()
    fieldset = get_fieldset()
    T = Field("temperature")
    expr = diff(T)
    result = basis(expr, fieldset, [0.0, 0.0], 1.0)
    @test isapprox(eval(result), mean([1.0, 2.0, 3.0, 4.0]))
end

