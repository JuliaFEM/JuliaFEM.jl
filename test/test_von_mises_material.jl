module VonMisesTests

using PyPlot
using JuliaFEM.Test
using JuliaFEM.MaterialModels: stiffnessTensor, calculate_stress, State
using JuliaFEM.MaterialModels: stiffnessTensorPlaneStress


function test_von_mises_3D_basic()

    steps = 1000
    strain_max = 0.003
    num_cycles = 3
    E = 200.0e3
    nu =  0.3
    ν = 0.3
    C = stiffnessTensor(E, ν)

    strain_tot = zeros(Float64, (steps, 6))
    strain_tot2 = zeros(Float64, (steps, 6))
    strain_tot3 = zeros(Float64, (steps, 6))

    # Adding only strain in x-axis and counting for the poisson effect
    strain_tot[:, 1] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps))
    strain_tot[:, 2] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν
    strain_tot[:, 3] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν
    strain_tot[:, 4] = strain_max / 10 * sin(2 * pi * linspace(0, num_cycles, steps))

    strain_last = zeros(Float64, (6))
    strain_p = zeros(Float64, (6))
    stress = zeros(Float64, (6, 1))
    stress_y =  200.0
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

    mat = State(C, stress_y, zeros(Float64, 6), zeros(Float64, 6))

    info("Starting calculation")
    tic()
    #=
    for i=1:steps
        strain_new = reshape(strain_tot[i, :, :], (6, 1))
        dstrain = strain_new - mat.strain
        calculate_stress!(dstrain, mat, Val{:vonMises})
        mat.strain += vec(dstrain)
        push!(ss, mat.stress[1])
        push!(ee, mat.strain[1])

        fill_tensor(eig_stress, mat.stress)
        eig_vals[i, :] = sort(eigvals(eig_stress))
    end
    =#
    stress = zeros(Float64, 6)
    strain = zeros(Float64, 6)
    for i=1:steps
        strain_new = reshape(strain_tot[i, :, :], (6, 1))
        dstrain = strain_new - strain
        calculate_stress!(dstrain, stress, C, stress_y, Val{:vonMises})
        strain = vec(strain_new)
        push!(ss, stress[1])
        push!(ee, strain[1])
        fill_tensor(eig_stress, stress)
        eig_vals[i, :] = sort(eigvals(eig_stress))
    end

    toc()
    # ================ Plotting =================== #
    n(θ, ϕ) = [sin(θ)*cos(ϕ)
               sin(θ)*sin(ϕ)
               cos(θ)]
    m(θ, ϕ, χ) = [-sin(ϕ)*cos(χ)-cos(θ)*cos(ϕ)*sin(χ)
                   cos(ϕ)*cos(χ)-cos(θ)*sin(ϕ)*sin(χ)
                   sin(θ)*sin(χ)]

    w = [sqrt(2/3) * 200 * m(54.735 * pi / 180, 45 * pi/180, x) for x=0:0.15:(2*pi+0.1)]
    base_vec =  [1 1 1] / sqrt(3)

    for i=-5:5
        tt = [w[x] + vec(base_vec) + 50 * i for x=1:length(w)]
        x = map(x->tt[x][1], collect(1:length(w)))
        y = map(x->tt[x][2], collect(1:length(w)))
        z = map(x->tt[x][3], collect(1:length(w)))
        plot3D(x, y, z, color="blue")
    end

    tt = [w[x] + vec(base_vec) + 50 * -5 for x=1:length(w)]
    x_start = map(x->tt[x][1], collect(1:length(w)))[1:5:end]
    y_start = map(x->tt[x][2], collect(1:length(w)))[1:5:end]
    z_start = map(x->tt[x][3], collect(1:length(w)))[1:5:end]


    tt = [w[x] + vec(base_vec) + 50 * 5 for x=1:length(w)]
    x_end = map(x->tt[x][1], collect(1:length(w)))[1:5:end]
    y_end = map(x->tt[x][2], collect(1:length(w)))[1:5:end]
    z_end = map(x->tt[x][3], collect(1:length(w)))[1:5:end]

    for i=1:length(x_start)
        x = [x_start[i], x_end[i]]
        y = [y_start[i], y_end[i]]
        z = [z_start[i], z_end[i]]
        plot3D(x, y, z, color="blue")
    end


    info("Calculation finished")
    #PyPlot.plot(ee, ss)
    plot3D(eig_vals[:, 1], eig_vals[:, 2], eig_vals[:, 3], color="red")
    PyPlot.title("Stress path and von Mises yield surface")
    PyPlot.xlabel("Eig Stress 1")
    PyPlot.ylabel("Eig Stress 2")
    PyPlot.zlabel("Eig Stress 3")
    PyPlot.grid()
    PyPlot.show()
end

function test_von_mises_planestress_basic()

    steps = 1000
    strain_max = 0.003
    num_cycles = 5
    E = 200.0e3
    nu =  0.3
    ν = 0.3
    C = stiffnessTensorPlaneStress(E, ν)

    strain_tot = zeros(Float64, (steps, 3))

    # Adding only strain in x-axis and counting for the poisson effect
    strain_tot[:, 1] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps))
    strain_tot[:, 2] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν
    strain_tot[:, 3] = strain_max * sin(2 * pi * linspace(0, num_cycles, steps)).*-ν

    strain_last = zeros(Float64, (3))
    strain_p = zeros(Float64, (3))
    stress = zeros(Float64, (3, 1))
    stress_y =  200.0
    ss = Float64[]
    ee = Float64[]


    ss2 = Float64[]
    ee2 = Float64[]

    eig_stress = zeros(Float64, (3, 3))
    eig_vals = zeros(Float64, (steps, 3))
    #mat = State(C, stress_y, zeros(Float64, 6), zeros(Float64, 6))

    info("Starting calculation")
    tic()
    #=
    for i=1:steps
        strain_new = reshape(strain_tot[i, :, :], (6, 1))
        dstrain = strain_new - mat.strain
        calculate_stress!(dstrain, mat, Val{:vonMises})
        mat.strain += vec(dstrain)
        push!(ss, mat.stress[1])
        push!(ee, mat.strain[1])

        fill_tensor(eig_stress, mat.stress)
        eig_vals[i, :] = sort(eigvals(eig_stress))
    end
    =#
    stress = zeros(Float64, 3)
    strain = zeros(Float64, 3)
    for i=1:steps
        strain_new = reshape(strain_tot[i, :, :], (3, 1))
        dstrain = strain_new - strain
        stress_inc, lambda = calculate_stress(dstrain,
                                              stress,
                                              C,
                                              stress_y,
                                              Val{:vonMises},
                                              Val{:PlaneStressElasticPlasticProblem})
        stress += stress_inc
        strain = vec(strain_new)
        s1, s2, t12  = stress
        se1 = (s1 + s2)/2 + sqrt(((s1 - s2)/2)^2 + t12^2)
        se2 = (s1 + s2)/2 - sqrt(((s1 - s2)/2)^2 + t12^2)
        push!(ss, se1)
        push!(ee, se2)
    end

    toc()

    function vm_upper(a, c)
        vals = f(a[1], a[2], c)
        vm(vals[1], vals[2], 200)
    end
    vm(a,b) = sqrt(a^2 - a*b + b^2) - 200
    f(m,c) = [600*cos(c) 600*sin(c)].*m
    x_vals = []
    max_iter = 100
    y_vals = []
    for i=0:0.1:(2*pi+0.3)
        wf(x) = f(x, i)
        t = 0.01
        step = 2
        merkki = -1
        s11, s22 = wf(t)
        ii = 0
        while (abs(vm(s11, s22)) > 1e-7) && ii < max_iter
            val = vm(s11, s22)
            if sign(val) != merkki
                merkki *= -1
                step *= -0.5
            end
            t += step
            s11, s22 = wf(t)
            ii += 1
        end
        push!(x_vals, s11)
        push!(y_vals, s22)
    end
    PyPlot.plot(x_vals, y_vals)
    PyPlot.plot(ee, ss)
    PyPlot.grid()
    PyPlot.show()
end

# test_von_mises_3D_basic()

test_von_mises_planestress_basic()

end
