using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Abaqus: create_surface_elements

Logging.configure(level=Logging.DEBUG)

function create_body(mesh, name, E, nu, rho)
    body = Problem(mesh, Elasticity, name, 3)
    update!(body, "youngs modulus", E)
    update!(body, "poissons ratio", nu)
    update!(body, "density", rho)
    return body
end

""" Create boundary condition from surface set. """
function create_bc_from_surface_set(mesh, name, u1, u2, u3)
    bc = Problem(Dirichlet, string(name), 3, "displacement")
    bc.elements = create_surface_elements(mesh, name)
    update!(bc, "displacement 1", u1)
    update!(bc, "displacement 2", u2)
    update!(bc, "displacement 3", u3)
    return bc
end

""" Create boundary condition from node set. """
function create_bc_from_node_set(mesh, name, u1, u2, u3)
    bc = Problem(Dirichlet, string(name), 3, "displacement")
    bc.elements = [Element(Poi1, [nid]) for nid in mesh.node_sets[name]]
    update!(bc, "geometry", mesh.nodes)
    update!(bc, "displacement 1", u1)
    update!(bc, "displacement 2", u2)
    update!(bc, "displacement 3", u3)
    return bc
end

function create_bc(mesh, name, u1=0.0, u2=0.0, u3=0.0)
    name = Symbol(name)
    if haskey(mesh.surface_sets, name)
        return create_bc_from_surface_set(mesh, name, u1, u2, u3)
    elseif haskey(mesh.node_sets, name)
        return create_bc_from_node_set(mesh, name, u1, u2, u3)
    else
        error("Mesh does not contain node or surface set $name")
    end
end

function create_interface(mesh, slave_surface::String, master_surface::String)
    interface = Problem(Mortar, "interface between $slave_surface and $master_surface", 3, "displacement")
    interface.properties.dual_basis = true
    slave_elements = create_surface_elements(mesh, Symbol(slave_surface))
    master_elements = create_surface_elements(mesh, Symbol(master_surface))
    nslaves = length(slave_elements)
    nmasters = length(master_elements)
    info("$nslaves slaves, $nmasters masters")
    update!(slave_elements, "master elements", master_elements)
    interface.elements = [slave_elements; master_elements]
    return interface
end

function create_interface(mesh, slave::Problem, master::Problem)
    slave_surface = slave.name * "_TO_" * master.name
    master_surface = master.name * "_TO_" * slave.name
    return create_interface(mesh, slave_surface, master_surface)
end

# start of simulation
mesh = abaqus_read_mesh("beam_3d_2nd_order_tetra_30mm.inp")
info("element sets = ", collect(keys(mesh.element_sets)))
info("surface sets = ", collect(keys(mesh.surface_sets)))

# parts
beam = create_body(mesh, "beam", 210.0e3, 0.3, 7.85e-9)

# boundary conditions
bc1 = create_bc(mesh, "fixed")

# load
load = Problem(Elasticity, "pressure load", 3)
load.elements = create_surface_elements(mesh, :load)
nload = length(load.elements)
info("$nload elements in load surface")
area = 0.0
time = 0.0
for element in load.elements
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        area += ip.weight*detJ
    end
end
info("load surface area: $area")

update!(load, "surface pressure", 20.0)

# solution
isfile("results.h5") && rm("results.h5")
isfile("results.xmf") && rm("results.xmf")
solver = Solver(Linear, beam, load)
solver.xdmf = Xdmf("results")
solver()

# calculate stresses in integration points and use least-squares fitting to move results to nodes

function get_stress(element, ip, time)
    haskey(element, "displacement") || return nothing
    gradu = element("displacement", ip, time, Val{:Grad})
    eps = 0.5*(gradu' + gradu)
    E = element("youngs modulus", ip, time)
    nu = element("poissons ratio", ip, time)
    mu = E/(2.0*(1.0+nu))
    la = E*nu/((1.0+nu)*(1.0-2.0*nu))
    S = la*trace(eps)*I + 2.0*mu*eps
    return [S[1,1], S[2,2], S[3,3], S[1,2], S[2,3], S[1,3]]
end

function lsq_fit(elements, field)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    for element in elements
	gdofs = get_connectivity(element)
	for ip in get_integration_points(element)
	    detJ = element(ip, time, Val{:detJ})
	    w = ip.weight*detJ
	    N = element(ip, time)
	    f = field(element, ip, time)
	    add!(A, gdofs, gdofs, w*kron(N', N))
	    for i=1:length(f)
		add!(b, gdofs, w*f[i]*N, i)
	    end
	end
    end
    A = sparse(A)
    b = sparse(b)
    A = 1/2*(A + A')
    
    SparseArrays.droptol!(A, 1.0e-6)
    nz = get_nonzero_rows(A)
    dropzeros!(A)
    #try
        F = ldltfact(A)

        x = zeros(size(b)...)
        x[nz, :] = F \ b[nz, :]
        nodal_values = Dict(i => vec(x[i,:]) for i=1:size(x,1))
    #catch
        err("Problem with zeros or tolerances")
        dump(nz)
    #end
    return nodal_values
end

S = lsq_fit(beam.elements, get_stress)

# store values to xml/h5
node_ids = sort(collect(keys(S)))
stress = hcat([S[nid] for nid in node_ids]...)

for (i, nid) in enumerate(node_ids)
    println("$nid -> $(S[nid])")
    if i > 3
        println("...")
        break
    end
end

using JuliaFEM: new_dataitem, new_child, set_attribute, add_child, save!

xdmf = get(solver.xdmf)
dataitem = new_dataitem(xdmf, "/Results/Time $(solver.time)/Nodal Fields/Stress", stress)
frame = read(xdmf, "/Domain/Grid/Grid")
attribute = new_child(frame, "Attribute")
set_attribute(attribute, "Name", "Stress")
set_attribute(attribute, "Center", "Node")
set_attribute(attribute, "AttributeType", "Tensor6")
add_child(attribute, dataitem)

save!(xdmf)
close(xdmf.hdf)

