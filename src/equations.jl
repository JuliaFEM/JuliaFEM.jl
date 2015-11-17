# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle element level things -- integration, assembly, ...

abstract Equation

type Assembly
    mass_matrix :: SparseMatrixIJV
    stiffness_matrix :: SparseMatrixIJV
    force_vector :: SparseMatrixIJV
    lhs :: SparseMatrixIJV
    rhs :: SparseMatrixIJV
end

function Assembly()
    return Assembly(
        SparseMatrixIJV(),
        SparseMatrixIJV(),
        SparseMatrixIJV(),
        SparseMatrixIJV(),
        SparseMatrixIJV())
end

function Base.empty!(assembly::Assembly)
    empty!(assembly.mass_matrix)
    empty!(assembly.stiffness_matrix)
    empty!(assembly.force_vector)
    empty!(assembly.lhs)
    empty!(assembly.rhs)
end

function get_mass_matrix
end

function get_stiffness_matrix
end

function get_force_vector
end

function get_potential_energy
end

function get_residual_vector
end

function has_mass_matrix(equation::Equation)
    default_args = Tuple{typeof(equation), IntegrationPoint, Float64}
    return method_exists(get_mass_matrix, default_args)
end

function has_stiffness_matrix(equation::Equation)
    default_args = Tuple{typeof(equation), IntegrationPoint, Float64}
    return method_exists(get_stiffness_matrix, default_args)
end

function has_force_vector(equation::Equation)
    default_args = Tuple{typeof(equation), IntegrationPoint, Float64}
    return method_exists(get_force_vector, default_args)
end

function has_potential_energy(equation::Equation)
    default_args = Tuple{typeof(equation), IntegrationPoint, Float64}
    return method_exists(get_potential_energy, default_args)
end

function has_residual_vector(equation::Equation)
    default_args = Tuple{typeof(equation), IntegrationPoint, Float64}
    return method_exists(get_residual_vector, default_args)
end

function get_element(equation::Equation)
    return equation.element
end

function get_integration_points(equation::Equation)
    return equation.integration_points
end

function Base.size(equation::Equation, i::Int)
    return size(equation)[i]
end

""" Return global degrees of freedom of element in matrix level.

Notes
-----
This is calculated from connectivity and equation dimension.
"""

function get_gdofs(equation::Equation)
    element = get_element(equation)
    conn = get_connectivity(element)
    dim = size(equation, 1)
    gdofs = vec(vcat([dim*conn'-i for i=dim-1:-1:0]...))
    return gdofs
end

function get_gdofs(element::Element, dim::Int)
    conn = get_connectivity(element)
    gdofs = vec(vcat([dim*conn'-i for i=dim-1:-1:0]...))
    return gdofs
end

""" Assemble element. """
function assemble!(assembly::Assembly, equation::Equation, time::Number=0.0, problem=nothing)

    element = get_element(equation)
    gdofs = get_gdofs(equation)
    basis = get_basis(element)
    detJ = det(basis)
    unknown_field_name = get_unknown_field_name(equation)

    # 1. if equations are defined we just integrate them, without caring how they are done
    if has_mass_matrix(equation) || has_stiffness_matrix(equation) || has_force_vector(equation)
        for ip in get_integration_points(equation)
            s = ip.weight*detJ(ip)
            if has_mass_matrix(equation)
                add!(assembly.mass_matrix, gdofs, gdofs, s*get_mass_matrix(equation, ip, time))
            end
            if has_stiffness_matrix(equation)
                add!(assembly.stiffness_matrix, gdofs, gdofs, s*get_stiffness_matrix(equation, ip, time))
            end
            if has_force_vector(equation)
                add!(assembly.force_vector, gdofs, s*get_force_vector(equation, ip, time))
            end
        end
        # external loads -- if any nodal loads is defined add to force vector
        if haskey(element, "$unknown_field_name nodal load")
            add!(assembly.force_vector, gdofs, vec(element["$unknown_field_name nodal load"](time)))
        end
    end

    # 2. energy form -- user has defined potential energy W -> min!
    if has_potential_energy(equation)
        field = element[unknown_field_name](time)

        """ Wrapper for potential energy for ForwardDiff. """
        function calc_W(data::Vector)
            W = 0.0
            df = similar(field, data)
            # integrate potential energy
            for ip in get_integration_points(equation)
                s = ip.weight*detJ(ip)
                dw = get_potential_energy(equation, ip, time; variation=df)
                W += s*dw
            end
            # external energy -- if any nodal loads is defined, decrease from potential energy
            if haskey(element, "$unknown_field_name nodal load")
                P = element["$unknown_field_name nodal load"](time)
                W -= dot(vec(P), vec(df))
            end
            return isa(W, Array) ? W[1] : W
        end

        hessian, allresults = ForwardDiff.hessian(calc_W, vec(field), AllResults, cache=autodiffcache)
        add!(assembly.stiffness_matrix, gdofs, gdofs, hessian)
        add!(assembly.force_vector, gdofs, -ForwardDiff.gradient(allresults))
    end

    # 3. virtual work -- user has defined some residual r = p - f = 0
    if has_residual_vector(equation)
        field = element[unknown_field_name](time)

        """ Wrapper for virtual work for ForwardDiff. """
        function calc_R(data::Vector)
            R = zeros(length(data))
            df = similar(field, data)
            # integrate residual vector
            for ip in get_integration_points(equation)
                s = ip.weight*detJ(ip)
                dr = get_residual_vector(equation, ip, time; variation=df)
                R += s*dr
            end
            # external loads -- if any nodal loads is defined, decrease from residual
            if haskey(element, "$unknown_field_name nodal load")
                R -= vec(element["$unknown_field_name nodal load"](time))
            end
            return R
        end
       
        jacobian, allresults = ForwardDiff.jacobian(calc_R, vec(field), AllResults, cache=autodiffcache)
        add!(assembly.stiffness_matrix, gdofs, gdofs, jacobian)
        add!(assembly.force_vector, gdofs, -ForwardDiff.value(allresults))
    end
end
