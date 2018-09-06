# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using ForwardDiff

"""
Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
"""
function find_root!(f, df, x; max_iter=50, norm_acc=1e-9)
    converged = false
    for i=1:max_iter
        dx = -df(x) \ f(x)
        x += dx
        norm(dx) < norm_acc && (converged = true; break)
    end
    converged || error("No convergence in radial return!")
    return x
end

"""
Equivalent tensile stress.

More info can be found from: https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
    Section: Reduced von Mises equation for different stress conditions
"""
function equivalent_stress(stress, ::Type{Val{:type_3d}})
    stress_ten = [stress[1] stress[6] stress[5];
                  stress[6] stress[2] stress[4];
                  stress[5] stress[4] stress[3]]
    stress_dev = stress_ten - 1/3 * tr(stress_ten) * eye(3)
    s = vec(stress_dev)
    return sqrt(3/2 * dot(s, s))
end

"""
http://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_plane_stress.cfm

von mises: plane stress
https://andriandriyana.files.wordpress.com/2008/03/yield_criteria.pdf
"""
function equivalent_stress(stress, ::Type{Val{:type_2d}})
    s1, s2, t12  = stress
    # Calculating principal stresses
    # http://www.engineersedge.com/material_science/principal_vonmises_stress__13418.htm
    se1 = (s1 + s2)/2 + sqrt(((s1 - s2)/2)^2 + t12^2)
    se2 = (s1 + s2)/2 - sqrt(((s1 - s2)/2)^2 + t12^2)

    return sqrt(se1^2 -se1*se2 + se2^2)
end

"""
https://andriandriyana.files.wordpress.com/2008/03/yield_criteria.pdf
"""
function yield_function(stress, stress_y, ::Type{Val{:von_mises}}, type_)
    equivalent_stress(stress, type_) - stress_y
end

function radial_return(params, dstrain, D, stress_y, stress_base, yield_surface_, type_)

    # Creating wrapper for gradient
    vm_wrap(stress_) = yield_function(stress_, stress_y, yield_surface_, type_)
    dfds = x -> ForwardDiff.gradient(vm_wrap, x)

    # Stress rate and total strain
    dstress = params[1:end-1]
    stress_tot = stress_base + dstress

    # Calculating plastic strain rate
    dstrain_p = params[end] * dfds(stress_tot)

    # Calculating equations
    function_1 = dstress - D * (dstrain - dstrain_p)
    function_2 = vm_wrap(stress_tot)
    [vec(function_1); function_2]
end

function ideal_plasticity!(stress_new, stress_last, dstrain_vec, pstrain, D, params, Dtan, yield_surface_, time, dt, type_)
    # Test stress
    dstress = vec(D * dstrain_vec)
    stress_trial = stress_last + dstress
    stress_y = params["yield_stress"]

    yield_curr = x -> yield_function(x, stress_y, yield_surface_, type_)

    # Calculating and checking for yield
    yield = yield_curr(stress_trial)
    if isless(yield, 0.0)

        stress_new[:] = stress_trial[:]
        Dtan[:,:] = D[:,:]
    else
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ \ f and initial values
        f = stress_ -> radial_return(stress_, dstrain_vec, D, stress_y, stress_last, yield_surface_, type_)
        df = x -> ForwardDiff.jacobian(f, x)

        # Calculating root (two options)
        vals = [vec(stress_trial - stress_last); 0.0]

        #results = nlsolve(not_in_place(f), vals).zero
        results = find_root!(f, df, vals)

        # extracting results
        dstress = results[1:end-1]
        plastic_multiplier = results[end]

        # Updating stress
        stress_new[:] = stress_last + dstress


        # Calculating plastic strain
        dfds_ = x -> ForwardDiff.gradient(yield_curr, x)
        dep = plastic_multiplier * dfds_(vec(stress_new))

        # Equations for consistent tangent matrix can be found from:
        # http://homes.civil.aau.dk/lda/continuum/plast.pdf
        # equations: 152 & 153
        D2g = x -> ForwardDiff.hessian(yield_curr, x)
        Dc = (D^-1 + plastic_multiplier * D2g(stress_new))^-1
        dfds = dfds_(stress_new)
        Dtan[:,:] = Dc - (Dc * dfds * dfds' * Dc) / (dfds' * Dc * dfds)[1]
        pstrain[:] = plastic_multiplier * dfds
    end
end
