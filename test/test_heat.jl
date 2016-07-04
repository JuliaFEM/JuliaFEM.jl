# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess

@testset "Tet10 + convection" begin
    # For some reason Tet10 fails, maybe because of convection.
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "Tet10")
    prob = Problem(Heat, "tet", 1)
    face = Problem(Heat, "face 4", 1)
    fixed = Problem(Dirichlet, "fixed face 3", 1, "temperature")
    prob.elements = create_elements(mesh, "TET")
    update!(prob, "temperature thermal conductivity", 50.0)
    face.elements = create_elements(mesh, "FACE4") 
    update!(face, "temperature external temperature", 20.0)
    update!(face, "temperature heat transfer coefficient", 60.0)
    fixed.elements = create_elements(mesh, "FACE2")
    info("# of elements in fixed set: $(length(fixed))")
    update!(fixed, "temperature 1", 0.0)
    solver = LinearSolver(prob, face, fixed)
    solver()
    T = prob.assembly.u
    info("Solution: $T")
    T_expected = [ # using code aster
        1.45606533688540E+01
        5.01315339269860E-17
        3.02236827927507E-17
       -2.01049663215778E-16
        1.05228712963739E+01
        0.00000000000000E+00
        9.44202309239159E+00
        1.05228712963739E+01
        4.44089209850063E-16
        0.00000000000000E+00]
    @test isapprox(T, T_expected; rtol=1.0e-6)
end

@testset "2d heat problem (one element)" begin

    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])

    # define volume element
    el1 = Element(Quad4, [1, 2, 3, 4])

    update!(el1, "geometry", X)
    update!(el1, "temperature thermal conductivity", 6.0)
    update!(el1, "temperature load", 12.0)

    # define boundary element for flux
    el2 = Element(Seg2, [1, 2])
    update!(el2, "geometry", X)
    # linear ramp from 0 -> 6 in time 0 -> 1
    update!(el2, "temperature flux", 0.0 => 0.0, 1.0 => 6.0)

    # define heat problem and push elements to problem
    problem = Problem(Heat, "one element heat problem", 1)
    problem.properties.formulation = "2D"
    push!(problem, el1, el2)

    # Set constant source f=12 with k=6. Accurate solution is
    # T=1 on free boundary, u(x,y) = -1/6*(1/2*f*x^2 - f*x)
    # when boundary flux not active (at t=0)
    assemble!(problem, 0.0)
    A = full(problem.assembly.K)
    b = full(problem.assembly.f)
    A_expected = [
         4.0 -1.0 -2.0 -1.0
        -1.0  4.0 -1.0 -2.0
        -2.0 -1.0  4.0 -1.0
        -1.0 -2.0 -1.0  4.0]
    free_dofs = [1, 2]
    @test isapprox(A, A_expected)
    @test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [1.0, 1.0])

    # Set constant flux g=6 on boundary. Accurate solution is
    # u(x,y) = x which equals T=1 on boundary.
    # at time t=1.0 all loads should be on.
    empty!(problem)
    assemble!(problem, 1.0)
    A = full(problem.assembly.K)
    b = full(problem.assembly.f)
    @test isapprox(A[free_dofs, free_dofs] \ b[free_dofs], [2.0, 2.0])
end

function T_acc(x)
    # accurate solution
    a = 0.01
    L = 0.20
    k = 50.0
    Tᵤ = 20.0
    h = 10.0
    P = 4*a
    A = a^2
    α = h
    β = sqrt((h*P)/(k*A))
    T̂ = 100.0
    C = [1.0 1.0; (α+k*β)*exp(β*L) (α-k*β)*exp(-β*L)] \ [T̂-Tᵤ, 0.0]
    return dot(C, [exp(β*x), exp(-β*x)]) + Tᵤ
end

#=
@testset "test 1d heat problem" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [0.1, 0.0, 0.0],
        3 => [0.2, 0.0, 0.0])
    e1 = Element(Seg2, [1, 2])
    e2 = Element(Seg2, [2, 3])
    e3 = Element(Poi1, [3])

    p1 = Problem(Heat, "1d heat problem", 1)
    p1.properties.formulation = "1D"
    push!(p1, e1, e2, e3)
    update!(p1, "geometry", X)
    a = 0.010
    update!(p1, "cross-section area", a^2)
    update!(p1, "cross-section perimeter", 4*a)
    update!(p1, "temperature thermal conductivity", 50.0) # k [W/(m∘C)]
    update!(p1, "temperature heat transfer coefficient", 10.0) # h [W/(m²∘C)]
    update!(p1, "temperature external temperature", 20.0)

    p2 = Problem(Dirichlet, "left boundary", 1, "temperature")
    e3 = Element(Poi1, [1])
    update!(e3, "geometry", X)
    update!(e3, "temperature 1", 100.0)
    push!(p2, e3)

    solver = LinearSolver(p1, p2)
    solver()
    T_min = minimum(p1.assembly.u)
    @test isapprox(T_max, T_acc(0.2); rtol=4.5e-2)
end
=#

@testset "compare simple 3d heat problem to code aster solution" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, "Hex8")
    element_sets = join(keys(mesh.element_sets), ", ")
    info("element sets: $element_sets")

    p1 = Problem(Heat, "rod", 1)
    rod = create_elements(mesh, "ROD")
    face2 = create_elements(mesh, "FACE2")
    face3 = create_elements(mesh, "FACE3")
    face4 = create_elements(mesh, "FACE4")
    face5 = create_elements(mesh, "FACE5")
    face6 = create_elements(mesh, "FACE6")
    update!(rod, "temperature thermal conductivity", 50.0)
    update!(face2, "temperature external temperature", 20.0)
    update!(face2, "temperature heat transfer coefficient", 60.0)
    update!(face3, "temperature external temperature", 30.0)
    update!(face3, "temperature heat transfer coefficient", 50.0)
    update!(face4, "temperature external temperature", 40.0)
    update!(face4, "temperature heat transfer coefficient", 40.0)
    update!(face5, "temperature external temperature", 50.0)
    update!(face5, "temperature heat transfer coefficient", 30.0)
    update!(face6, "temperature external temperature", 60.0)
    update!(face6, "temperature heat transfer coefficient", 20.0)
    push!(p1, rod, face2, face3, face4, face5, face6)

    p2 = Problem(Dirichlet, "left support T=100", 1, "temperature")
    push!(p2, create_elements(mesh, "FACE1"))
    update!(p2, "temperature 1", 100.0)

    solver = LinearSolver(p1, p2)
    solver()

    # fields extracted from Code Aster .resu file
    TEMP = Dict{Int64, Float64}(
        1 => 1.00000000000000E+02,
        2 => 1.00000000000000E+02,
        3 => 1.00000000000000E+02,
        4 => 1.00000000000000E+02,
        5 => 3.01613322896279E+01,
        6 => 3.01263406641066E+01,
        7 => 3.02559777927923E+01,
        8 => 3.02209215997131E+01)
    FLUX_ELGA = Dict{Int64, Vector{Float64}}(
        1 => [1.74565160615448E+04, -9.99903237329079E+01, -3.69874201221677E+01],
        2 => [1.74565160615448E+04, -3.73168968436642E+02, -1.38038931136833E+02],
        3 => [1.74428571293096E+04, -9.99903237329079E+01, -3.70268090662933E+01],
        4 => [1.74428571293096E+04, -3.73168968436642E+02, -1.38185932677561E+02],
        5 => [1.74615686370955E+04, -9.99509347888079E+01, -3.69874201221677E+01],
        6 => [1.74615686370955E+04, -3.73021966895897E+02, -1.38038931136833E+02],
        7 => [1.74479150854902E+04, -9.99509347888065E+01, -3.70268090662933E+01],
        8 => [1.74479150854901E+04, -3.73021966895874E+02, -1.38185932677561E+02])
    FLUX_NOEU = Dict{Int64, Vector{Float64}}(   
        1 => [1.74596669275930E+04,  7.55555618070503E-11,  3.68594044175552E-12],
        2 => [1.74684148339734E+04,  1.10418341137120E-11,  3.48876483258209E-12],
        3 => [1.74360055518019E+04,  7.91828824731056E-11,  1.95399252334028E-13],
        4 => [1.74447696000717E+04, -3.49587025993969E-12,  3.55271367880050E-13],
        5 => [1.74596669275931E+04, -4.73227515822099E+02, -1.74958127606525E+02],
        6 => [1.74684148339733E+04, -4.72904678032251E+02, -1.74958127606524E+02],
        7 => [1.74360055518019E+04, -4.73227515822118E+02, -1.75280965396335E+02],
        8 => [1.74447696000717E+04, -4.72904678032179E+02, -1.75280965396335E+02])

    postprocessor = Postprocessor(p1)
    flux = full(postprocessor())
    fluxd = Dict{Int64, Vector{Float64}}()
    for j=1:8
        fluxd[j] = vec(flux[j,:])
    end

    T = p1("temperature")

    for j in sort(collect(keys(T)))
        T1 = T[j][1]
        T2 = TEMP[j]
        rtol = norm(T1-T2)/max(T1,T2)*100.0
        @printf "node %i temp, JF: %e, CA: %e, rtol: %10.6f %%\n" j T1 T2 rtol
        @test rtol < 1.0e-9
    end

    for j=1:8
        q1 = get_integration_points(first(rod))[j]("heat flux", 0.0)
        q2 = FLUX_ELGA[j]
        rtol = norm(q1-q2)/max(norm(q1),norm(q2))*100.0
        @printf "ip   %i flux, JF: (% e,% e,% e), CA: (% e,% e,% e), rtol: %10.6f %%\n" j q1... q2... rtol
        # @test rtol < 0.05
        # testing in integration points makes no sense because they are in different order in CA
    end

    for j in sort(collect(keys(fluxd)))
        q1 = fluxd[j]
        q2 = FLUX_NOEU[j]
        rtol = norm(q1-q2)/max(norm(q1),norm(q2))*100.0
        @printf "node %i flux, JF: (% e,% e,% e), CA: (% e,% e,% e), rtol: %10.6f %%\n" j q1... q2... rtol
        @test rtol < 1.0e-9
    end

end

@testset "compare simple 3d heat problem to analytical solution" begin
    function calc_3d_heat_model(mesh_name)
        fn = Pkg.dir("JuliaFEM") * "/test/testdata/rod_short.med"
        mesh = aster_read_mesh(fn, mesh_name)
        p1 = Problem(Heat, "rod", 1)
        p2 = Problem(Dirichlet, "left support T=100", 1, "temperature")
        p1.elements = create_elements(mesh, "ROD", "FACE2")
        p2.elements = create_elements(mesh, "FACE1")
        update!(p1, "temperature thermal conductivity", 100.0)
        update!(p1, "temperature external temperature", 0.0)
        update!(p1, "temperature heat transfer coefficient", 1000.0)
        update!(p2, "temperature 1", 100.0)
        solver = LinearSolver(p1, p2)
        solver()
        T_min = minimum(p1.assembly.u)
        return T_min
    end
    for model in ["Tet4", "Tet10", "Hex8", "Hex20", "Hex27"]
        Tmin = calc_3d_heat_model(model)
        Tacc = 100/3
        rtol = norm(Tmin-Tacc)/max(Tmin,Tacc)*100.0
        @printf "%-10s : Tmin = % g, Tacc = % g, rtol = %g %%\n" model Tmin Tacc rtol
        @test isapprox(Tmin, 100/3)
    end
end

@testset "compare simple 3d heat problem to code aster solution" begin

    function calc_3d_heat_model(mesh_name)
        fn = Pkg.dir("JuliaFEM") * "/test/testdata/rod_short.med"
        mesh = aster_read_mesh(fn, mesh_name)
        element_sets = join(keys(mesh.element_sets), ", ")
        info("element sets: $element_sets")
        # x -> FACE1 ... FACE2
        # y -> FACE3 ... FACE4
        # z -> FACE5 ... FACE6
        # rod has longer dimension in x direction, first face comes
        # first in corresponding axis direction
        p1 = Problem(Heat, "rod", 1)
        rod = create_elements(mesh, "ROD")
        face2 = create_elements(mesh, "FACE2")
        face3 = create_elements(mesh, "FACE3")
        face4 = create_elements(mesh, "FACE4")
        face5 = create_elements(mesh, "FACE5")
        face6 = create_elements(mesh, "FACE6")
        update!(rod, "temperature thermal conductivity", 50.0)
        update!(face2, "temperature external temperature", 20.0)
        update!(face2, "temperature heat transfer coefficient", 60.0)
        update!(face3, "temperature external temperature", 30.0)
        update!(face3, "temperature heat transfer coefficient", 50.0)
        update!(face4, "temperature external temperature", 40.0)
        update!(face4, "temperature heat transfer coefficient", 40.0)
        update!(face5, "temperature external temperature", 50.0)
        update!(face5, "temperature heat transfer coefficient", 30.0)
        update!(face6, "temperature external temperature", 60.0)
        update!(face6, "temperature heat transfer coefficient", 20.0)
        push!(p1, rod, face2, face3, face4, face5, face6)
        p2 = Problem(Dirichlet, "left support T=100", 1, "temperature")
        p2.elements = create_elements(mesh, "FACE1")
        update!(p2, "temperature 1", 100.0)
        solver = LinearSolver(p1, p2)
        solver()
        return p1.assembly.u
    end

    CA_sol = Dict(
        "Tet4" => 3.01872246268290E+01,
        "Hex8" => 3.01263406641066E+01,
        "Tet10" => 4.38924023356612E+01,
        "Hex20" => 4.57539800177123E+01,
        "Hex27" => 4.57760386068096E+01)

    models = ["Tet4", "Hex8", "Hex20", "Hex27", "Tet10"]

    for model in models
        T = calc_3d_heat_model(model)
        T_min = minimum(T)
        T_ca = CA_sol[model]
        rtol = norm(T_min-T_ca)/max(T_min,T_ca)*100.0
        @printf "%-10s : T_min = % g, T_ca = % g, rtol = %g %%\n" model T_min T_ca rtol
        if rtol > 1.0e-9
            info("Solution vector")
            dump(T)
        end
        @test rtol < 1.0e-9
    end

end
