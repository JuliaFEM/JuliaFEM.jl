# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

function get_stress_tensor(element, ip, time)
    haskey(element, "displacement") || return nothing
    gradu = element("displacement", ip, time, Val{:Grad})
    eps = 0.5*(gradu' + gradu)
    E = element("youngs modulus", ip, time)
    nu = element("poissons ratio", ip, time)
    mu = E/(2.0*(1.0+nu))
    la = E*nu/((1.0+nu)*(1.0-2.0*nu))
    S = la*trace(eps)*I + 2.0*mu*eps
    return S
end

""" Return stress vector in "ABAQUS" order 11, 22, 33, 12, 23, 13. """
function get_stress(element, ip, time)
    S = get_stress_tensor(element, ip, time)
    return [S[1,1], S[2,2], S[3,3], S[1,2], S[2,3], S[1,3]]
end

""" Return principal stresses. """
function get_stress_principal(element, ip, time)
    S = get_stress_tensor(element, ip, time)
    return sort(eigvals(S))
end

""" Make least squares fit for some field to nodes. """
function lsq_fit(elements, field, time)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    volume = 0.0
    for element in elements
        gdofs = get_connectivity(element)
        # increase integration order by 1 from default
        for ip in get_integration_points(element, 1)
	        detJ = element(ip, time, Val{:detJ})
	        w = ip.weight*detJ
	        N = element(ip, time)
	        f = field(element, ip, time)
	        add!(A, gdofs, gdofs, w*kron(N', N))
	        for i=1:length(f)
                add!(b, gdofs, w*f[i]*N, i)
            end
            volume += w
	    end
    end
    info("Mass matrix for least-squares fit is assembled. Total volume to fit: $volume")
    A = sparse(A)
    b = sparse(b)
    A = 1/2*(A + A')
    
    SparseArrays.droptol!(A, 1.0e-6)
    SparseArrays.dropzeros!(A)
    nz = get_nonzero_rows(A)
    F = ldltfact(A[nz,nz])

    x = zeros(size(b)...)
    x[nz, :] = F \ b[nz, :]

    nodal_values = Dict(i => vec(x[i,:]) for i in nz)
    return nodal_values
end

#=
test subjects:
- surface pressure load in curved surface
- verification of elements wedge6 and wedge15

from Code Aster:
N38      -8.85861895037377E-01 -3.46944695195361E-18 -3.46944695195361E-18
coords of N38 = (1.0, 0.0, 0.0)

Analytical solution
uᵣ(r) = b³p/(2Er²(a³-b³)) * (a³(ν+1) + r³(-4ν+2)), where
a = inner surface radial distance, b = outer surface ...

if a=0.9, b=1.0, ν=1/3, E = 24580 and p = 7317 equation yields
-9/10 for radial displacement

http://mms2.ensmp.fr/emms_paris/plasticite3D/exercices/eSpherePress.pdf
=#
function test_wedge_sphere(model)
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, model)
    body = Problem(Elasticity, "hollow sphere 1/8 model", 3)
    body.elements = create_elements(mesh, "HOLLOWSPHERE8")
    update!(body, "youngs modulus", 24580.0)
    update!(body, "poissons ratio", 1/3)
    bc = Problem(Dirichlet, "symmetry bc", 3, "displacement")
    el1 = create_elements(mesh, "FACE1")
    update!(el1, "displacement 3", 0.0)
    el2 = create_elements(mesh, "FACE2")
    update!(el2, "displacement 2", 0.0)
    el3 = create_elements(mesh, "FACE3")
    update!(el3, "displacement 1", 0.0)
    bc.elements = [el1; el2; el3]
    lo = Problem(Elasticity, "pressure load", 3)
    lo.elements = create_elements(mesh, "OUTER")
    update!(lo, "surface pressure", 7317.0)
    solver = LinearSolver(body, bc, lo)
    solver()

    X = lo("geometry")
    u = lo("displacement")
    nids = sort(collect(keys(X)))
    umag = Float64[norm(u[id]) for id in nids]
    um = mean(umag)
    us = std(umag)
    rtol = norm(um - 0.9) / max(norm(um), 0.9) * 100.0
    info("mean umag = $um, std umag = $us, rtol = $rtol")
    @test rtol < 1.5 # percents
    
    # for linear model
    #=
    u_CA = [-8.85861895037377E-01, -3.46944695195361E-18, -3.46944695195361E-18]
    rtol = norm(u[38] - u_CA) / max(norm(u[38]), norm(u_CA)) * 100.0
    info("rel diff to CA = $rtol %")
    @test isapprox(u[38], u_CA)
    =#

    S = lsq_fit(body.elements, get_stress, 0.0)
    
    # Calculate principal stresses in nodes
    Sp = Dict()
    for (nid, s) in S
        # order is: 11, 22, 33, 12, 23, 13
        stress_tensor = [
            s[1] s[4] s[6]
            s[4] s[2] s[5]
            s[6] s[5] s[3]]
        Sp[nid] = sort(eigvals(stress_tensor))
    end
    
    node_ids = sort(collect(keys(Sp)))
    for (i, nid) in enumerate(node_ids)
        println("$nid -> $(Sp[nid])")
        if i > 9
            println("...")
            break
        end
    end
    
    for element in body.elements
        ip = [1/3, 1/3, 1/3]
        time = 0.0
	    stress_tensor = get_stress_tensor(element, ip, time)
        principal_stresses = sort(eigvals(stress_tensor))
        println("princial stresses = $principal_stresses")
    end
end

@testset """1/8 hollow sphere with surface load""" begin
    test_wedge_sphere("HOLLOWSPHERE8_WEDGE6")
    #test_wedge_sphere("HOLLOWSPHERE8_WEDGE15")
end

