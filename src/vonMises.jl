# imports



using ForwardDiff
using NLsolve


"""
Create a isotropic Hooke material matrix C

More information: http://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_isotropic.cfm
                  https://en.wikipedia.org/wiki/Hooke's_law
                  http://www.ce.berkeley.edu/~sanjay/ce231mse211/symidentity.pdf
Parameters
----------
    E: Float
        Elastic modulus
    ν: Float
        Poisson constant

Returns
-------
    Array{Float64, (6,6)}
"""
#=
my_kron(i,j) = i == j ? 1 : 0
II = zeros(Float64, (3, 3, 3, 3))
for i=1:3
    for j=1:3
        for k=1:3
            for l=1:3
                A[i,j,k,l] = my_kron(i, j) * my_kron(k, l)
            end
        end
    end
end
=#
#function outer_product(a, b)
#
#end
function hookeStiffnessTensor(E, ν)
    a = 1 - ν
    b = 1 - 2*ν
    c = 1 + ν
    multiplier = E / (b * c)
    return Float64[a ν ν 0 0 0;
                   ν a ν 0 0 0;
                   ν ν a 0 0 0;
                   0 0 0 b 0 0;
                   0 0 0 0 b 0;
                   0 0 0 0 0 b].*multiplier
end

# Pick material values
#E = 200.0e3
#mu =  0.3
#C = hookeStiffnessTensor(E, ν)
#lambda =
#nu =
#C = λ * I ⊗ I + 2 * μ * II

type State
    C :: Array{Float64, 2}
    σ_y :: Float64
    σ :: Array{Float64, 1}
    ϵ :: Array{Float64, 1}
end

# using vectors with double contradiction
# http://www-2.unipv.it/compmech/teaching/available/const_mod/const_mod_mat-review_notation.pdf
M = [1 0 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 2 0 0;
     0 0 0 0 2 0;
     0 0 0 0 0 2;]

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
function σₑ(σ)
    s = σ[1:6] - 1/3 * sum([σ[1], σ[2], σ[3]]) * [1 1 1 0 0 0]'
    return sqrt(3/2 * s' * M * s)[1]
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
function vonMisesYield(σ, k)
    σₑ(σ) - k
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
function vonMisesRoot(params, dϵ, C, σ_y, σ_begin)

    # Creating wrapper for gradient
    yield_wrap(pars) = vonMisesYield(pars, σ_y)
    dfdσ = ForwardDiff.gradient(yield_wrap)

    # Stress rate
    dσ = params[1:6]

    σ_tot = [vec(σ_begin); 0.0] + params

    # Calculating plastic strain rate
    dϵp = params[end] * dfdσ(σ_tot)

    # Calculating equations
    function_1 = dσ - C * (dϵ - dϵp[1:6])
    function_2 = yield_wrap(σ_tot)
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
function calculate_stress!(dϵ, mat::State, ::Type{Val{:vonMises}})
    σ = mat.σ
    C = mat.C
    σ_y = mat.σ_y
    # Test stress
    σ_tria = σ + C * dϵ

    # Calculating and checking for yield
    yield = vonMisesYield(σ_tria, σ_y)
    if yield > 0
        # Yielding happened
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
        initial_guess = [vec(σ_tria - σ); 0.1]
        f(σ_)  = vonMisesRoot(σ_, dϵ, C, σ_y, σ)
        df     = ForwardDiff.jacobian(f)

        # Calculating root
        result = nlsolve(not_in_place(f, df), initial_guess).zero
        mat.σ  += result[1:6]
    else
        mat.σ = vec(σ_tria)
    end
end
