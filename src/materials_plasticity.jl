using ForwardDiff
using NLsolve

# Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
function find_root!(f, df, x; max_iter=50, norm_acc=1e-9)
    converged = false
    for i=1:max_iter
        dx = -df(x) \ f(x)
        norm(dx) < norm_acc && (converged = true; break)

        x += dx

    end
    converged || error("no convergence!")
    return x
end

type State
    C :: Array{Float64, 2}
    stress_y :: Float64
    stress :: Array{Float64, 1}
    strain :: Array{Float64, 1}
end

"""
Equivalent tensile stress.

More info can be found from: https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
    Section: Reduced von Mises equation for different stress conditions

Parameters
----------
    σ: Array{Float64, 6}
        Stress in Voigt notation

Returns
-------
    Float
"""
function stress_eq(stress)
    stress_ten = [stress[1] stress[6] stress[5];
                  stress[6] stress[2] stress[4];
                  stress[5] stress[4] stress[3]]
    stress_dev = stress_ten - 1/3 * trace(stress_ten) * eye(3)
    s = vec(stress_dev)
    return sqrt(3/2 * dot(s, s))
end


"""
Von Mises Yield criterion

More info can be found from: http://csm.mech.utah.edu/content/wp-content/uploads/2011/10/9tutorialOnJ2Plasticity.pdf

Parameters
----------
    σ: Array{Float64, 6}
        Stress in Voigt notation
    k: Float64
        Material constant, Yield limit

Returns
-------
    Float
"""
function vonMisesYield(stress, stress_y)
    stress_eq(stress) - stress_y
end

"""
Function for NLsolve. Inside this function are the equations which we want to find root.
Ψ is the yield function below. Functions defined here:

    dσ - C (dϵ - dλ*dΨ/dσ) = 0
                 σₑ(σ) - k = 0

Parameters
----------
    params: Array{Float64, 7}
        Array containing values from solver
    dϵ: Array{Float64, 6}
        Strain rate vector in Voigt notation
    C: Array{Float64, (6, 6)}
        Material tensor
    k: Float
        Material constant, yield limit
    Δt: Float
        time increment
    σ_begin:Array{Float64, 6}
        Stress vector in Voigt notation

Returns
-------
    Array{Float64, 7}, return values for solver
"""
function vonMisesRoot(params, dstrain, C, stress_y, stress_base)

    # Creating wrapper for gradient
    vm_wrap(stress_) = vonMisesYield(stress_, stress_y)
    dfds = ForwardDiff.gradient(vm_wrap)

    # Stress rate and total strain
    dstress = params[1:6]
    stress_tot = vec(stress_base) + params[1:6]

    # Calculating plastic strain rate
    dstrain_p = params[end] * dfds(stress_tot)

    # Calculating equations
    function_1 = dstress - C * (dstrain - dstrain_p)
    function_2 = vm_wrap(stress_tot)
    [vec(function_1); function_2]
end



"""
Stress for ideal plastic von Mises material model

Parameters
----------
    dϵ: Array{Float64, 6}
        Strain rate vector in Voigt notation
    Δt: Float
        time increment
    σ: Array{Float64, 6}
        Last stress vector in Voigt notation
    C: Array{Float64, (6, 6)}
        Material tensor
    k: Float
        Material constant, yield limit

Returns
-------
    Tuple
    Plastic strain rate dϵᵖ and new stress vector σ
"""
function calculate_stress!(dstrain, mat::State, ::Type{Val{:vonMises}})
    stress = mat.stress
    C = mat.C
    stress_y = mat.stress_y
    # Test stress
    stress_tria = stress + C * dstrain

    # Calculating and checking for yield
    yield = vonMisesYield(stress_tria, stress_y)
    if isless(yield, 0.0)
        mat.stress = vec(stress_tria)
    else
        # Yielding happened
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
        initial_guess = Float64[vec(stress_tria - stress); 0.1]
        f(stress_)  = vonMisesRoot(stress_, dstrain, C, stress_y, stress)
        df     = ForwardDiff.jacobian(f)

        # Calculating root
        result = nlsolve(not_in_place(f, df), initial_guess).zero
        mat.stress  += result[1:6]
    end
end

function calculate_stress(dstrain, stress, C, stress_y,
     ::Type{Val{:vonMises}},
     ::Type{Val{:ElasticPlasticProblem}})
    # Test stress
    stress_tria = stress + C * dstrain

    # Calculating and checking for yield
    yield = vonMisesYield(stress_tria, stress_y)
    if isless(yield, 0.0)
        # stress[i] = stress_tria[i]
        return 0.0
    else
        # Yielding happened
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
        x = [vec(stress_tria - stress); 0.0]
        f(stress_)  = vonMisesRoot(stress_, dstrain, C, stress_y, stress)
        df     = ForwardDiff.jacobian(f)

        # Calculating root
        # result = nlsolve(not_in_place(f, df), initial_guess).zero
        max_iter = 10
        converged = false
        for i=1:5
            dx = df(x) \ -f(x)
            x += dx
         #   println(x)
            norm(dx) < 1e-10 && (converged = true; break)
        end
        converged || error("no convergence!")
        # stress[:] += x[1:6]
        return x[end]
    end
end

##################################################################################
# ----- AFTER THIS POINT: VON MISES : PLANE STRESS IMPLEMENTATION          ----- #
##################################################################################

"""
http://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_plane_stress.cfm
"""
# von mises: plane stress
# https://andriandriyana.files.wordpress.com/2008/03/yield_criteria.pdf
function stress_eq_plane_stress(stress)
    s1, s2, t12  = stress
    # Calculating principal stresses
    # http://www.engineersedge.com/material_science/principal_vonmises_stress__13418.htm
    se1 = (s1 + s2)/2 + sqrt(((s1 - s2)/2)^2 + t12^2)
    se2 = (s1 + s2)/2 - sqrt(((s1 - s2)/2)^2 + t12^2)

    return sqrt(se1^2 -se1*se2 + se2^2)
end

# https://andriandriyana.files.wordpress.com/2008/03/yield_criteria.pdf
function vonMisesYieldPlaneStress(stress, stress_y)
    stress_eq_plane_stress(stress) - stress_y
end

function vonMisesRootPlaneStress(params, dstrain, D, stress_y, stress_base)

    # Creating wrapper for gradient
    vm_wrap(stress_) = vonMisesYieldPlaneStress(stress_, stress_y)
    dfds = x -> ForwardDiff.gradient(vm_wrap, x)

    # Stress rate and total strain
    dstress = params[1:3]
    stress_tot = stress_base + params[1:3]

    # Calculating plastic strain rate
    dstrain_p = params[end] * dfds(stress_tot)

    # Calculating equations
    function_1 = dstress - D * (dstrain - dstrain_p)
    function_2 = vm_wrap(stress_tot)
    [vec(function_1); function_2]
end

# http://homes.civil.aau.dk/lda/continuum/plast.pdf
function plastic_von_mises!(stress, dstrain_vec, D, params, Dtan)
    # Test stress
    dstress = vec(D * dstrain_vec)
    stress_tria = stress + dstress
    stress_y = params["yield_stress"]

    # Calculating and checking for yield
    yield = vonMisesYieldPlaneStress(stress_tria, stress_y)
    if isless(yield, 0.0)
        stress[:] = stress_tria[:]
        Dtan[:,:] = D[:,:]
    else
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ \ f and initial values
        x = [vec(stress_tria - stress); 0.0]
        f = stress_ -> vonMisesRootPlaneStress(stress_, dstrain_vec, D, stress_y, stress)
        df = x -> ForwardDiff.jacobian(f, x)

        # Calculating root
        results = nlsolve(not_in_place(f), x).zero
        dstress = results[1:3]

        stress_new = stress + dstress
        plastic_multiplier = results[end]
        f_ = stress_ -> vonMisesYieldPlaneStress(stress_, stress_y)
        dfds_ = x -> ForwardDiff.gradient(f_, x)
        dep = plastic_multiplier * dfds_(vec(stress_new))

        D2g = x -> ForwardDiff.hessian(f_, x)
        Dc = (D^-1 + plastic_multiplier * D2g(stress_new))^-1
        dfds = dfds_(stress_new)
        Dtan = Dc - (Dc * dfds * dfds' * Dc) / (dfds' * Dc * dfds)[1]
        println("plastic stress")
        println(stress_new)
        println(Dtan)
        stress[:] = stress_new[:]
        # stress[:] = D * dstrain_vec
        println("elastic stress")
        println(stress)
        println(D)
        Dtan[:,:] = D[:,:]
    end
end
