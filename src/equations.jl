# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract Equation

abstract Assembly

""" Local element assembly. """
type LocalAssembly <: Assembly
    ndofs :: Int
    mass_matrix :: Matrix
    stiffness_matrix :: Matrix
    force_vector :: Matrix
    potential_energy# :: Union{Array, Float64}
    residual_vector :: Vector
end

function LocalAssembly(ndofs, mass_matrix, stiffness_matrix, force_vector::Matrix)
    LocalAssembly(ndofs, mass_matrix, stiffness_matrix, force_vector[:])
end

""" Initialize workspace for local assembly. """
function LocalAssembly(equation::Equation)
    ndofs = size(equation)
    mass_matrix = zeros(ndofs, ndofs)
    stiffness_matrix = zeros(ndofs, ndofs)
    force_vector = zeros(ndofs, 1)
    potential_energy = 0.0
    residual_vector = zeros(ndofs)
    return LocalAssembly(ndofs, mass_matrix, stiffness_matrix, force_vector,
                         potential_energy, residual_vector)
end

function initialize_local_assembly(equation::Equation)
    LocalAssembly(equation)
end

function initialize_local_assembly(equation::Equation, assembly::LocalAssembly)
    if size(equation) != assembly.ndofs
        # if problem size changes, automatically initialize new work space
        return initialize_local_assembly(equation)
    end
    # otherwise, empty workspace ready for next iteration
    fill!(assembly.mass_matrix, 0.0)
    fill!(assembly.stiffness_matrix, 0.0)
    fill!(assembly.force_vector, 0.0)
    assembly.potential_energy = 0.0
    fill!(assembly.residual_vector, 0.0)
    return assembly
end
function initialize_local_assembly(assembly::LocalAssembly, equation::Equation)
    initialize_local_assembly(equation, assembly)
end

function get_unknown_field_name(equation::Equation)
    eqtype = typeof(equation)
    error("define get_unknown_field_name for this equation type $eqtype")
end

has_mass_matrix(equation::Equation) = false
get_mass_matrix(equation::Equation, ip, time) = nothing
has_stiffness_matrix(equation::Equation) = false
get_stiffness_matrix(equation::Equation, ip, time) = nothing
has_force_vector(equation::Equation) = false
get_force_vector(equation::Equation, ip, time) = nothing
has_residual_vector(equation::Equation) = false
get_residual_vector(equation::Equation, ip, time) = nothing
has_potential_energy(equation::Equation) = false
get_potential_energy(equation::Equation, ip, time) = nothing
get_element(equation::Equation) = equation.element
get_number_of_dofs(equation::Equation) = nothing
get_integration_points(equation::Equation) = equation.integration_points


""" Return a local assembly for element. """
function calculate_local_assembly!(assembly::LocalAssembly, equation::Equation, time::Number=Inf)

    initialize_local_assembly(assembly, equation) # zero all

    element = get_element(equation)
    basis = get_basis(element)
    detJ = det(basis)
    field_name = get_unknown_field_name(equation)

    # 1. if equations are defined we just integrate them
    if has_mass_matrix(equation) || has_stiffness_matrix(equation) || has_force_vector(equation)
        for ip in get_integration_points(equation)
            s = ip.weight*detJ(ip)
            if has_mass_matrix(equation)
                assembly.mass_matrix += s*get_mass_matrix(equation, ip, time)
            end
            if has_stiffness_matrix(equation)
                assembly.stiffness_matrix += s*get_stiffness_matrix(equation, ip, time)
            end
            if has_force_vector(equation)
                assembly.force_vector += s*get_force_vector(equation, ip, time)[:]
            end
            # external loads -- if any nodal loads is defined add to force vector
            if haskey(element, "$field_name nodal load")
                assembly.force_vector += element["$field_name nodal load"](time)[:]
            end
        end
    end

    # 2. variational / energy form - user has defined some potential energy / variational form
    if has_potential_energy(equation)
        field_name = get_unknown_field_name(equation)
        element = get_element(equation)
        field = element[field_name](time)
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
            if haskey(element, "$field_name nodal load")
                P = element["$field_name nodal load"](time)
                assembly.potential_energy -= dot(P[:], df[:])
            end
            if isa(assembly.potential_energy, Array)
                return assembly.potential_energy[1]
            end
            return assembly.potential_energy
        end
        hessian, allresults = ForwardDiff.hessian(potential_energy, field[:],
                                                  AllResults, cache=autodiffcache)
        assembly.stiffness_matrix += hessian
        assembly.force_vector -= ForwardDiff.gradient(allresults)  # <--- minus explained in tutorial
        assembly.potential_energy = ForwardDiff.value(allresults)
    end

    # 3. virtual work form - user has defined residual vector δW_int(u,δu) + δW_ext(u,δu) = 0 ∀ v
    if has_residual_vector(equation)
        field_name = get_unknown_field_name(equation)
        element = get_element(equation)
        field = element[field_name](time)
        function residual_vector(data::Vector)
            fill!(assembly.residual_vector, 0.0)
            df = similar(field, data)
            # integrate W
            for ip in get_integration_points(equation)
                dr = get_residual_vector(equation, ip, time; variation=df)
                assembly.residual_vector += ip.weight*dr*detJ(ip)
            end
            # external loads -- if any nodal loads is defined, remove from residual
            if haskey(element, "$field_name nodal load")
                assembly.residual_vector -= element["$field_name nodal load"](time)[:]
            end
            return assembly.residual_vector
        end
        jacobian, allresults = ForwardDiff.jacobian(residual_vector, field[:],
                                                    AllResults, cache=autodiffcache)
        assembly.stiffness_matrix += jacobian
        assembly.force_vector -= ForwardDiff.value(allresults)  # <-- minus explained in tutorial
    end

end

function calculate_local_assembly!(equation::Equation, assembly::LocalAssembly, time::Number=Inf)
    calculate_local_assembly!(assembly, equation)
end


""" Get global degrees of freedom for this element. """
function get_global_dofs(eq::Equation)
    eq.global_dofs
end

""" Set global degrees of freedom for this element. """
function set_global_dofs!(eq::Equation, dofs)
    eq.global_dofs = dofs
end

