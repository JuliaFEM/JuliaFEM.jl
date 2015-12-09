# imports
using ForwardDiff
using NLsolve
using PyPlot

"""
Create a isotropic Hooke material matrix C

More information: # http://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_isotropic.cfm

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
E = 200.0e3
ν =  0.3
C = hookeStiffnessTensor(E, ν)


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
function G(params, dϵ, C, k, σ_begin)

    # Creating wrapper for gradient
    yield(pars) = vonMisesYield(pars, k)
    dfdσ = ForwardDiff.gradient(yield)

    # Stress rate
    dσ = params[1:6]

    σ_tot = [vec(σ_begin); 0.0] + params

    # Calculating plastic strain rate
    dϵp = params[end] * dfdσ(σ_tot)

    # Calculating equations
    function_1 = dσ - C * (dϵ - dϵp[1:6])
    function_2 = yield(σ_tot)
    [vec(function_1); function_2]
end



"""
Function which calculates the stress. Also handles if any yielding happens

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
function calculate_stress!(dϵ, mat::State)
    σ = mat.σ
    C = mat.C
    σ_y = mat.σ_y
    # Test stress
    σ_tria = σ + C * dϵ

    # Calculating yield
    yield = vonMisesYield(σ_tria, σ_y)

    if yield > 0
        # Yielding happened
        # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
        initial_guess = [vec(σ_tria - σ); 0.1]
        f(σ_)  = G(σ_, dϵ, C, σ_y, σ)
        df     = ForwardDiff.jacobian(f)

        # Calculating root
        result = nlsolve(not_in_place(f, df), initial_guess).zero

        mat.σ  += result[1:6]
    else
        mat.σ = vec(σ_tria)
    end
end

steps = 1000
strain_max = 0.003
num_cycles = 3

ϵ_tot = zeros(Float64, (steps, 6))
ϵ_tot2 = zeros(Float64, (steps, 6))
ϵ_tot3 = zeros(Float64, (steps, 6))

# Adding only strain in x-axis and counting for the poisson effect
ϵ_tot[:, 1] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps))
ϵ_tot[:, 2] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν
ϵ_tot[:, 3] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν
ϵ_tot[:, 4] = strain_max / 10 * sin(2 * pi * linspace(0, num_cycles, steps))

ϵ_last = zeros(Float64, (6))
ϵᵖ = zeros(Float64, (6))
σ = zeros(Float64, (6, 1))
σy =  200.0
ss = Float64[]
ee = Float64[]

eig_stress = zeros(Float64, (3, 3))
eig_vals = zeros(Float64, (steps, 3))
function fill_tensor(a, b)
    a[1, 1] = b[1]
    a[2, 2] = b[2]
    a[3, 3] = b[3]

    a[1, 2] = b[6]
    a[1, 3] = b[5]
    a[2, 3] = b[4]

    a[2, 1] = b[6]
    a[3, 1] = b[5]
    a[3, 2] = b[4]
end
mat = State(C, σy, zeros(Float64, 6), zeros(Float64, 6))

info("Starting calculation")
for i=1:steps
    ϵ_new = reshape(ϵ_tot[i, :, :], (6, 1))
    dϵ = ϵ_new - mat.ϵ
    calculate_stress!(dϵ, mat)
    mat.ϵ += vec(dϵ)
    push!(ss, mat.σ[1])
    push!(ee, mat.ϵ[1])

    fill_tensor(eig_stress, mat.σ)
    eig_vals[i, :] = sort(eigvals(eig_stress))
end

# ================ Plotting =================== #
n(θ, ϕ) = [sin(θ)*cos(ϕ)
           sin(θ)*sin(ϕ)
           cos(θ)]
m(θ, ϕ, χ) = [-sin(ϕ)*cos(χ)-cos(θ)*cos(ϕ)*sin(χ)
               cos(ϕ)*cos(χ)-cos(θ)*sin(ϕ)*sin(χ)
               sin(θ)*sin(χ)]

w = [sqrt(2/3) * 200 * m(54.735 * pi / 180, 45 * pi/180, x) for x=0:0.2:(2*pi+0.1)]
base_vec =  [1 1 1] / sqrt(3)

for i=-7:7
    tt = [w[x] + vec(base_vec) + 50 * i for x=1:length(w)]
    x = map(x->tt[x][1], collect(1:length(w)))
    y = map(x->tt[x][2], collect(1:length(w)))
    z = map(x->tt[x][3], collect(1:length(w)))
    plot3D(x, y, z, color="blue")
end


#n(54.735 * pi / 180, 45 * pi/180)
info("Calculation finished")
#PyPlot.plot(ee, ss)
plot3D(eig_vals[:, 1], eig_vals[:, 2], eig_vals[:, 3], color="red")
PyPlot.title("Stress-Strain curve")
PyPlot.xlabel("Strain")
PyPlot.ylabel("Stress")
PyPlot.grid()
# PyPlot.plot(ee, ss)
PyPlot.show()
