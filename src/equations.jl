# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle element level things -- integration, assembly, ...

type FieldAssembly
    mass_matrix :: SparseMatrixCOO
    stiffness_matrix :: SparseMatrixCOO
    force_vector :: SparseMatrixCOO
end

function FieldAssembly()
    return FieldAssembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO())
end

typealias Assembly FieldAssembly

"""
"Boundary" matrices C₁, C₂, D, g for general problem type
Au + C₁'λ = f
C₂u + Dλ  = g
"""
type BoundaryAssembly
    C1 :: SparseMatrixCOO
    C2 :: SparseMatrixCOO
    D :: SparseMatrixCOO
    g :: SparseMatrixCOO
end

function BoundaryAssembly()
    return BoundaryAssembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO())
end

function Base.empty!(assembly::Assembly)
    empty!(assembly.mass_matrix)
    empty!(assembly.stiffness_matrix)
    empty!(assembly.force_vector)
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

function has_mass_matrix(problem::Problem, element::Element)
    default_args = Tuple{typeof(problem), typeof(element), IntegrationPoint, Float64}
    return method_exists(get_mass_matrix, default_args)
end

function has_stiffness_matrix(problem::Problem, element::Element)
    default_args = Tuple{typeof(problem), typeof(element), IntegrationPoint, Float64}
    return method_exists(get_stiffness_matrix, default_args)
end

function has_force_vector(problem::Problem, element::Element)
    default_args = Tuple{typeof(problem), typeof(element), IntegrationPoint, Float64}
    return method_exists(get_force_vector, default_args)
end

function has_potential_energy(problem::Problem, element::Element)
    default_args = Tuple{typeof(problem), typeof(element), IntegrationPoint, Float64}
    return method_exists(get_potential_energy, default_args)
end

function has_residual_vector(problem::Problem, element::Element)
    default_args = Tuple{typeof(problem), typeof(element), IntegrationPoint, Float64}
    return method_exists(get_residual_vector, default_args)
end

function get_gdofs(element::Element, dim::Int)
    conn = get_connectivity(element)
    gdofs = vec(vcat([dim*conn'-i for i=dim-1:-1:0]...))
    return gdofs
end

""" Assemble element. """
function assemble!(assembly::Assembly, problem::Problem, element::Element, time::Number)

    gdofs = get_gdofs(element, problem.dim)
    unknown_field_name = get_unknown_field_name(problem)

    # 1. if equations are defined we just integrate them, without caring how they are done
    if has_mass_matrix(problem, element) || has_stiffness_matrix(problem, element) || has_force_vector(problem, element)
        for ip in get_integration_points(element)
            w = ip.weight*det(J)
            if has_mass_matrix(element)
                add!(assembly.mass_matrix, gdofs, gdofs, w*get_mass_matrix(problem, element, ip, time))
            end
            if has_stiffness_matrix(element)
                add!(assembly.stiffness_matrix, gdofs, gdofs, w*get_stiffness_matrix(problem, element, ip, time))
            end
            if has_force_vector(element)
                add!(assembly.force_vector, gdofs, w*get_force_vector(problem, element, ip, time))
            end
        end
        # external loads -- if any nodal loads is defined add to force vector
        if haskey(element, "$unknown_field_name nodal load")
            add!(assembly.force_vector, gdofs, vec(element["$unknown_field_name nodal load"](time)))
        end
    end

    # 2. energy form -- user has defined potential energy W -> min!
    if has_potential_energy(problem, element) && haskey(element, unknown_field_name)
        field = element[unknown_field_name](time)

        """ Wrapper for potential energy for ForwardDiff. """
        function calc_W(data::Vector)
            W = 0.0
            df = similar(field, data)
            # integrate potential energy
            for ip in get_integration_points(element)
                dw = get_potential_energy(problem, element, ip, time; variation=df)
                W += ip.weight*dw
            end
            # external energy -- if any nodal loads is defined, decrease from potential energy
            if haskey(element, "$unknown_field_name nodal load")
                P = element["$unknown_field_name nodal load"](time)
                W -= dot(vec(P), vec(df))
            end
            return W[1]
        end

        hessian, allresults = ForwardDiff.hessian(calc_W, vec(field), AllResults, cache=autodiffcache)
        add!(assembly.stiffness_matrix, gdofs, gdofs, hessian)
        add!(assembly.force_vector, gdofs, -ForwardDiff.gradient(allresults))
    end

    # 3. virtual work -- user has defined some residual r = p - f = 0
    if has_residual_vector(problem, element) && haskey(element, unknown_field_name)

        field = DVTI(last(element[unknown_field_name]).data)

        """ Wrapper for virtual work for ForwardDiff. """
        function calc_R(data::Vector)
            R = zeros(length(data))
            df = similar(field, data)
            gauss_fields = IntegrationPoint[]
            # integrate residual vector
            for ip in get_integration_points(element)
                dr = get_residual_vector(problem, element, ip, time; variation=df)
                R += ip.weight*dr
                if ip.changed
                    push!(gauss_fields, ip)
                end
            end
            # external loads -- if any nodal loads is defined, decrease from residual
            if haskey(element, "$unknown_field_name nodal load")
                R -= vec(element["$unknown_field_name nodal load"](time))
            end
            #info("return = $R")
            if length(gauss_fields) != 0
                update_gauss_fields!(element, gauss_fields, time)
            end
            return R
        end

        jacobian, allresults = ForwardDiff.jacobian(calc_R, vec(field), AllResults, cache=autodiffcache)
        residual_vector = -ForwardDiff.value(allresults)
        add!(assembly.stiffness_matrix, gdofs, gdofs, jacobian)
        add!(assembly.force_vector, gdofs, residual_vector)
    end
end

