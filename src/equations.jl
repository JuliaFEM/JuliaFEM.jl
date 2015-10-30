# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle element level things -- integration, assembly, ...

abstract Equation

abstract Assembly

""" Local element assembly. """
type LocalAssembly <: Assembly
    ndofs :: Int
    mass_matrix :: Matrix
    stiffness_matrix :: Matrix
    force_vector :: Matrix
    potential_energy
    residual_vector :: Vector
end

""" Initialize workspace for local matrices for dimension ndofs. """
function initialize_local_assembly(ndofs::Int=1)
    mass_matrix = zeros(ndofs, ndofs)
    stiffness_matrix = zeros(ndofs, ndofs)
    force_vector = zeros(ndofs, 1)
    potential_energy = 0.0
    residual_vector = zeros(ndofs)
    return LocalAssembly(ndofs, mass_matrix, stiffness_matrix, force_vector,
                         potential_energy, residual_vector)
end

""" Initialize workspace for local matrices, get dimension from equation. """
function initialize_local_assembly(equation::Equation)
    ndofs = prod(size(equation))
    return initialize_local_assembly(ndofs)
end

""" Initialize or zero workspace. """
function initialize_local_assembly!(assembly::LocalAssembly, equation::Equation)
    ndofs = prod(size(equation))
    if ndofs != assembly.ndofs
        # if problem size changes, automatically initialize new work space
        assembly.ndofs = ndofs
        assembly.mass_matrix = zeros(ndofs, ndofs)
        assembly.stiffness_matrix = zeros(ndofs, ndofs)
        assembly.force_vector = zeros(ndofs, 1)
        assembly.potential_energy = 0.0
        assembly.residual_vector = zeros(ndofs)
        return
    end
    # otherwise, empty workspace ready for next iteration
    fill!(assembly.mass_matrix, 0.0)
    fill!(assembly.stiffness_matrix, 0.0)
    fill!(assembly.force_vector, 0.0)
    assembly.potential_energy = 0.0
    fill!(assembly.residual_vector, 0.0)
    return
end

has_mass_matrix(equation::Equation) = false
function get_mass_matrix(equation::Equation, ip, time=Inf, problem=nothing)
    get_mass_matrix(equation, ip, time)
end
function get_mass_matrix(equation::Equation, ip, time=Inf)
    get_mass_matrix(equation, ip)
end
function get_mass_matrix(equation::Equation, ip)
    nothing
end

has_stiffness_matrix(equation::Equation) = false
function get_stiffness_matrix(equation::Equation, ip, time=Inf, problem=nothing)
    get_stiffness_matrix(equation, ip, time)
end
function get_stiffness_matrix(equation::Equation, ip, time=Inf)
    get_stiffness_matrix(equation, ip)
end
function get_stiffness_matrix(equation::Equation, ip)
    nothing
end

has_force_vector(equation::Equation) = false
function get_force_vector(equation::Equation, ip, time=Inf, problem=nothing)
    get_force_vector(equation, ip, time)
end
function get_force_vector(equation::Equation, ip, time=Inf)
    get_force_vector(equation, ip)
end
function get_force_vector(equation::Equation, ip)
    nothing
end

has_residual_vector(equation::Equation) = false
function get_residual_vector(equation::Equation, ip, time=Inf, problem=nothing)
    get_residual_vector(equation, ip, time)
end
function get_residual_vector(equation::Equation, ip, time=Inf)
    get_residual_vector(equation, ip)
end
function get_residual_vector(equation::Equation, ip)
    nothing
end

has_potential_energy(equation::Equation) = false
function get_potential_energy(equation::Equation, ip, time=Inf, problem=nothing)
    get_potential_energy(equation, ip, time)
end
function get_potential_energy(equation::Equation, ip, time=Inf)
    get_potential_energy(equation, ip)
end
function get_potential_energy(equation::Equation, ip)
    nothing
end

get_element(equation::Equation) = equation.element
get_integration_points(equation::Equation) = equation.integration_points


""" Return a local assembly for element. """
function calculate_local_assembly!(assembly::LocalAssembly, equation::Equation,
                                   unknown_field_name::ASCIIString, time::Number=Inf,
                                   problem=nothing)

    initialize_local_assembly!(assembly, equation) # zero all

    element = get_element(equation)
    basis = get_basis(element)
    detJ = det(basis)

    # 1. if equations are defined we just integrate them
    if has_mass_matrix(equation) || has_stiffness_matrix(equation) || has_force_vector(equation)
        for ip in get_integration_points(equation)
            s = ip.weight*detJ(ip)
            if has_mass_matrix(equation)
                assembly.mass_matrix += s*get_mass_matrix(equation, ip, time, problem)
            end
            if has_stiffness_matrix(equation)
                assembly.stiffness_matrix += s*get_stiffness_matrix(equation, ip, time, problem)
            end
            if has_force_vector(equation)
                assembly.force_vector += s*get_force_vector(equation, ip, time, problem)
            end
        end
        # external loads -- if any nodal loads is defined add to force vector
        if haskey(element, "$unknown_field_name nodal load")
            assembly.force_vector += vec(element["$unknown_field_name nodal load"](time))
        end
    end

    # 2. variational / energy form - user has defined some potential energy / variational form
    if has_potential_energy(equation)
        element = get_element(equation)
        field = element[unknown_field_name](time)
        function potential_energy(data::Vector)
            # calculate potential energy for some setting. this is needed by forwarddiff
            assembly.potential_energy = 0.0
            df = similar(field, data)
            # integrate potential energy
            for ip in get_integration_points(equation)
                dw = get_potential_energy(equation, ip, time; variation=df)
                assembly.potential_energy += ip.weight * dw * detJ(ip)
            end
            # external energy -- if any nodal loads is defined, decrease from potential energy
            if haskey(element, "$unknown_field_name nodal load")
                P = element["$unknown_field_name nodal load"](time)
                assembly.potential_energy -= dot(vec(P), vec(df))
            end
            if isa(assembly.potential_energy, Array)
                return assembly.potential_energy[1]
            end
            return assembly.potential_energy
        end
        hessian, allresults = ForwardDiff.hessian(potential_energy, vec(field),
                                                  AllResults, cache=autodiffcache)
        assembly.stiffness_matrix += hessian
        assembly.force_vector -= ForwardDiff.gradient(allresults)  # <--- minus explained in tutorial
        assembly.potential_energy = ForwardDiff.value(allresults)
    end

    # 3. virtual work form - user has defined residual vector δW_int(u,δu) + δW_ext(u,δu) = 0 ∀ v
    if has_residual_vector(equation)
        element = get_element(equation)
        field = element[unknown_field_name](time)
        function residual_vector(data::Vector)
            fill!(assembly.residual_vector, 0.0)
            #@debug("field: $field, length = $(size(field))")
            #@debug("data: $data, size = $(size(data))")
            #df = similar(field, data)
            df = Increment(reshape(data, size(equation)...))
            # integrate W
            for ip in get_integration_points(equation)
                dr = get_residual_vector(equation, ip, time; variation=df)
                assembly.residual_vector += ip.weight*dr*detJ(ip)
            end
            # external loads -- if any nodal loads is defined, remove from residual
            if haskey(element, "$unknown_field_name nodal load")
                assembly.residual_vector -= vec(element["$unknown_field_name nodal load"](time))
            end
            return assembly.residual_vector
        end
        jacobian, allresults = ForwardDiff.jacobian(residual_vector, vec(field),
                                                    AllResults, cache=autodiffcache)
        assembly.stiffness_matrix += jacobian
        assembly.force_vector -= ForwardDiff.value(allresults)  # <-- minus explained in tutorial
    end

end
