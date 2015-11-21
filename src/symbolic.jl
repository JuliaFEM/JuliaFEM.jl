# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/notebooks/2015-06-14-data-structures.ipynb

immutable SymbolicField <: AbstractField
    name :: ASCIIString
end

immutable Expression
    expr :: Expr
end

function Expression(expression::ASCIIString)
    return Expression(parse(expression))
end

function Base.convert(::Type{Field}, name::ASCIIString)
    return SymbolicField(name)
end

function Base.convert(::Type{Symbol}, field::SymbolicField)
    return Symbol(field.name)
end

function Base.(:*)(n::Number, field::SymbolicField)
    expr = Expr(:call, :*, n, Symbol(field.name))
    return Expression(expr)
end

function grad(field::SymbolicField)
    expr = Expr(:call, :grad, Symbol(field.name))
    return Expression(expr)
end

function diff(field::SymbolicField)
    expr = Expr(:call, :diff, Symbol(field.name))
    return Expression(expr)
end

function Base.call(basis::Basis, expr::Expression, fieldset::FieldSet,
                   xi::Vector, time::Number)
    return replace(expr.expr, basis, fieldset, xi, time)
end

# Unbelievable code. I don't know why or how this works.
""" Replace symbolic fields with real arrays. """
function Base.replace(expression::Expr, basis::Basis, fieldset::FieldSet,
                      xi::Vector, time::Number, data=Dict())

    info("expression = $expression")
    if expression.head == symbol("'")
        info("transpose")
        expr = Expr(:call, :transpose, expression.args...)
        return replace(expr, basis, fieldset, xi, time, data)
    end
    operator = expression.args[1]

    if operator == :diff
        info("inside diff operator")
        if !haskey(data, expression)
            data[expression] = fieldset[string(expression.args[2])](time, Val{:diff})
        end
        info("data = $(data[expression])")
        #return basis(data[expression], xi)
        return data[expression]
    end
    
    if operator == :grad
        info("inside gradient operator")
        info("gradient args: $(expression.args)")
        field_name = expression.args[2]
        if isa(field_name, Expr)
            info("expression inside gradient")
        end
        #if startswith(string(field_name), "diff")
        if !haskey(data, field_name)
            info("grad: evaluate field $field_name")
            if isa(field_name, Symbol)
                data[field_name] = fieldset[string(field_name)](time)
            elseif isa(field_name, Expr) && (field_name.args[1] == :diff)
                info("taking time derivative of $(field_name.args[2])")
                #data[field_name] = fieldset[string(field_name.args[2])](time, Val{:diff})
                data[field_name] = replace(field_name, basis, fieldset, xi, time, data)
            end
        end
        if !haskey(data, :geometry)
            info("evaluate geometry")
            data[:geometry] = fieldset["geometry"](time)
        end
        #return Expr(:call, basis, data[:geometry], data[field_name], xi, Val{:gradient})
        #return :(basis($(data[:geometry]), $(data[field_name]), $xi, Val{:gradient}))
        info("evaluate gradient")
        return basis(data[:geometry], data[field_name], xi, Val{:grad})
    end

    for i in 2:length(expression.args)
        arg = expression.args[i]
        if isa(arg, Expr)
            expression.args[i] = replace(arg, basis, fieldset, xi, time, data)
        end
        if isa(arg, Symbol) && haskey(fieldset, string(arg))
            if !haskey(data, arg)
                info("evaluate field $arg")
                data[arg] = fieldset[string(arg)](time)
            end
            #expression.args[i] = Expr(:call, basis, data[arg], xi)
            #return :(basis($(data[arg]), $xi))
            expression.args[i] = basis(data[arg], xi)
        end
    end

    info("new expression: $expression")
    return expression
end

