# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test, SparseArrays, LinearAlgebra, Statistics

function get_stress_tensor(element, ip, time)
    haskey(element, "displacement") || return nothing
    gradu = element("displacement", ip, time, Val{:Grad})
    eps = 0.5*(gradu' + gradu)
    E = element("youngs modulus", ip, time)
    nu = element("poissons ratio", ip, time)
    mu = E/(2.0*(1.0+nu))
    la = E*nu/((1.0+nu)*(1.0-2.0*nu))
    S = la*tr(eps)*I + 2.0*mu*eps
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
    @info("Mass matrix for least-squares fit is assembled. Total volume to fit: $volume")
    A = sparse(A)
    b = sparse(b)
    A = 1/2*(A + A')

    SparseArrays.droptol!(A, 1.0e-6)
    SparseArrays.dropzeros!(A)
    nz = get_nonzero_rows(A)
    F = ldlt(A[nz,nz])

    x = zeros(size(b)...)
    x[nz, :] = F \ b[nz, :]

    nodal_values = Dict(i => vec(x[i,:]) for i in nz)
    return nodal_values
end

#=
test subjects:
- surface pressure load in curved surface
- verification of elements wedge6 and wedge15
- stress interpolation from gauss points to nodes

from Code Aster, for linear model
 N38      -8.85861895037377E-01 -3.46944695195361E-18 -3.46944695195361E-18
coords of N38 = (1.0, 0.0, 0.0)
 N49      -4.50934684240566E-01 -4.46908405333021E-01 -5.92329377847994E-01
coords of N49 = (0.521002, 0.515482, 0.68032) # quite middle of surface

Analytical solution
uᵣ(r) = b³p/(2Er²(a³-b³)) * (a³(ν+1) + r³(-4ν+2)), where
a = inner surface radial distance, b = outer surface ...

if a=0.9, b=1.0, ν=1/3, E = 24580 and p = 7317 equation yields
-9/10 for radial displacement

http://mms2.ensmp.fr/emms_paris/plasticite3D/exercices/eSpherePress.pdf
=#
function test_wedge_sphere(model, u_CA, S_CA)
    mesh_file = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, model)

    body = Problem(Elasticity, "hollow sphere 1/8 model", 3)
    body_elements = create_elements(mesh, "HOLLOWSPHERE8")
    update!(body_elements, "youngs modulus", 24580.0)
    update!(body_elements, "poissons ratio", 1/3)
    add_elements!(body, body_elements)

    bc = Problem(Dirichlet, "symmetry bc", 3, "displacement")
    el1 = create_elements(mesh, "FACE1")
    update!(el1, "displacement 3", 0.0)
    el2 = create_elements(mesh, "FACE2")
    update!(el2, "displacement 2", 0.0)
    el3 = create_elements(mesh, "FACE3")
    update!(el3, "displacement 1", 0.0)
    add_elements!(bc, el1, el2, el3)

    load = Problem(Elasticity, "pressure load", 3)
    load_elements = create_elements(mesh, "OUTER")
    update!(load_elements, "surface pressure", 7317.0)
    add_elements!(load, load_elements)

    analysis = Analysis(Linear)
    add_problems!(analysis, body, load, bc)
    run!(analysis)

    X = load("geometry", 0.0)
    u = load("displacement", 0.0)
    nids = sort(collect(keys(X)))
    umag = Float64[norm(u[id]) for id in nids]
    um = mean(umag)
    us = std(umag)
    rtol = norm(um - 0.9) / max(norm(um), 0.9) * 100.0
    @debug("Displacement field statistics", mean=um, std=us, rtol=rtol)
    @test rtol < 1.5 # percents

    @debug("Verifying displacement against Code Aster solution.. ")
    pass = true
    for nid in keys(u_CA)
        rtol = norm(u[nid] - u_CA[nid]) / max(norm(u[nid]), norm(u_CA[nid])) * 100.0
        @debug("Node id $nid, rel diff to CA = $rtol %")
        pass &= isapprox(u[nid], u_CA[nid])
    end
    @test pass

    S = lsq_fit(body.elements, get_stress, 0.0)

    @debug("Verifying stress against Code Aster solution.. ")
    for nid in keys(S_CA)
        rtol = norm(S[nid] - S_CA[nid]) / max(norm(S[nid]), norm(S_CA[nid])) * 100.0
        @debug("Node id $nid, S=$(S[nid]), S_CA=$(S_CA[nid]), rel diff to CA = $rtol %")
        # @test isapprox(S[nid], S_CA[nid])
        # http://code-aster.org/doc/default/en/man_r/r3/r3.06.03.pdf
        pass &= (rtol < 10.0) # percents
    end
    @test pass

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
        @debug("$nid -> $(Sp[nid])")
        if i > 9
            @debug("...")
            break
        end
    end
    # Stress state in M141 element
    S_CA_gp = Dict()
    S_CA_gp[1] = [-2.31497155149131E+04, -2.30873958541450E+04, -2.59731750416165E+04, 1.16262216932559E+04, 1.05883267606309E+04, 1.04255059462671E+04]
    S_CA_gp[2] = [-2.20268425477595E+04, -2.59602179285979E+04, -2.45141753387488E+04, 1.08344827948196E+04, 1.02393080182646E+04, 1.18221520216293E+04]
    S_CA_gp[3] = [-2.12175032863505E+04, -2.38106746858742E+04, -2.75848287641255E+04, 1.23383626529382E+04, 9.45231798421561E+03, 1.04600859263009E+04]
    S_CA_gp[4] = [-2.88674347639302E+04, -2.90047144010521E+04, -3.19545639967296E+04, 1.24436733999204E+04, 1.14033189825355E+04, 1.09837505652098E+04]
    S_CA_gp[5] = [-2.76741820176108E+04, -3.20576001061179E+04, -3.04041166591832E+04, 1.16023096459047E+04, 1.10324243354755E+04, 1.24679360552984E+04]
    S_CA_gp[6] = [-2.68141147396280E+04, -2.97733271271234E+04, -3.36672334497799E+04, 1.32004501509553E+04, 1.01961070966382E+04, 1.10204979599285E+04]

    time = 0.0
    for element in body.elements
        element.id == 141 || continue
        for (i, ip) in enumerate(get_integration_points(element))
            S = get_stress(element, ip, time)
            rtol = norm(S - S_CA_gp[i]) / max(norm(S), norm(S_CA_gp[i]))
            @debug("$i $S, rtol=$rtol")
            pass &= isapprox(S, S_CA_gp[i])
        end
    end
    @test pass
end

# 1/8 hollow sphere with surface load
u_CA = Dict()
u_CA[38] = [-8.85861895037377E-01, -3.46944695195361E-18, -3.46944695195361E-18]
u_CA[49] = [-4.50934684240566E-01, -4.46908405333021E-01, -5.92329377847994E-01]
S_CA = Dict()
S_CA[38] = [-1.94504819940510E+03, -3.47014655438479E+04, -3.40876135857114E+04, 1.70379805227866E+03, 2.16707698388864E+03, 4.32009983342141E-12]
S_CA[49] = [-2.36067010168718E+04, -2.33820775692753E+04, -1.80606705820104E+04, 8.85783922092890E+03, 1.12122431934777E+04, 1.14035796957343E+04]
test_wedge_sphere("HOLLOWSPHERE8_WEDGE6", u_CA, S_CA)
#test_wedge_sphere("HOLLOWSPHERE8_WEDGE15")
