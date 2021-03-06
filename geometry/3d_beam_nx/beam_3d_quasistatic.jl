# Quasistatic analysis for beam, including several timesteps

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

""" Convert Mesh object from quadratic to linear. """
function to_linear!(mesh)
    mapping = Dict(:Tet10 => :Tet4, :Tri6 => :Tri3)
    nnodes = Dict(:Tet4 => 4, :Tri3 => 3)
    for elid in keys(mesh.elements)
        eltype = mesh.element_types[elid]
        if haskey(mapping, eltype)
            mesh.element_types[elid] = mapping[eltype]
            nnodes_new = nnodes[mapping[eltype]]
            mesh.elements[elid] = mesh.elements[elid][1:nnodes_new]
        end
    end
end

# start of simulation
#mesh = abaqus_read_mesh("beam_3d_1st_order_tetra_30mm.inp")
mesh = abaqus_read_mesh("beam_3d_2nd_order_tetra_30mm.inp")
#to_linear!(mesh)
info("element sets = ", collect(keys(mesh.element_sets)))
info("surface sets = ", collect(keys(mesh.surface_sets)))

# parts
beam = create_body(mesh, "beam", 210.0e3, 0.3, 7.85e-9)

# boundary conditions
bc1 = create_bc(mesh, "fixed")

# load
load = Problem(Elasticity, "pressure load", 3)
load.elements = create_surface_elements(mesh, :load)
update!(load, "surface pressure", 0.0 => 0.0)
update!(load, "surface pressure", 1.0 => 20.0)

# give some information about surface load
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

# solution
isfile("results_quasistatic.h5") && rm("results_quasistatic.h5")
isfile("results_quasistatic.xmf") && rm("results_quasistatic.xmf")
solver = Solver(Linear, beam, bc1, load)
solver.xdmf = Xdmf("results")

for time in [0.0, 1.0]
    solver.time = time
    solver()
end

# calculate stresses in integration points and use least-squares fitting to
# extrapolate results to nodes

""" Return stress tensor. """
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

xdmf = get(solver.xdmf)

using JuliaFEM: new_dataitem, new_child, set_attribute, add_child, save!

""" Store values to xml/h5. """
function add_field_to_xdmf!(xdmf, field_name, xmlpath, xdmfpath, data::Dict; field_type="Vector")
    node_ids = sort(collect(keys(data)))
    info("Storing field $field_name to xdmfpath $xdmfpath, xmlpath $xmlpath")
    for (i, nid) in enumerate(node_ids)
        println("$nid -> $(data[nid])")
        if i > 2
            println("...")
            break
        end
    end
    datavec = hcat([data[nid] for nid in node_ids]...)
    dataitem = new_dataitem(xdmf, xdmfpath, datavec)
    frame = read(xdmf, xmlpath)
    attribute = new_child(frame, "Attribute")
    set_attribute(attribute, "Name", field_name)
    set_attribute(attribute, "Center", "Node")
    set_attribute(attribute, "AttributeType", field_type)
    add_child(attribute, dataitem)
end

# calculate and update principle stresses to xdmf file, also calculate minimum
# and maximum principle components over time

minp = Dict()
maxp = Dict()

for (i, time) in enumerate([0.0, 1.0])
    S = lsq_fit(beam.elements, get_stress, time)
    #Sp = lsq_fit(beam.elements, get_stress_principal)


    # Calculate principal stresses in nodes
    Sp = Dict()
    for (nid, s) in S
        # order is: 11, 22, 33, 12, 23, 13
        stress_tensor = [
            s[1] s[4] s[6]
            s[4] s[2] s[5]
            s[6] s[5] s[3]]
        principle = sort(eigvals(stress_tensor))
        Sp[nid] = principle

        # minimum and maximum of principle stress over time
        if haskey(minp, nid)
            minp[nid] = min(minp[nid], minimum(principle))
        else
            minp[nid] = minimum(principle)
        end
        if haskey(maxp, nid)
            maxp[nid] = max(maxp[nid], maximum(principle))
        else
            maxp[nid] = maximum(principle)
        end
    end

    xml_path = "/Domain/Grid/Grid[$i]"
    xdmf_path = "/Results/Time $time/Nodal Fields/Stress"
    add_field_to_xdmf!(xdmf, "Stress", xml_path, xdmf_path, S; field_type="Tensor6")
    xdmf_path = "/Results/Time $time/Nodal Fields/Principal Stress"
    add_field_to_xdmf!(xdmf, "Principal Stress", xml_path, xdmf_path, Sp; field_type="Vector")
end

save!(xdmf)
close(xdmf.hdf)

# calculate mean stress and stress amplitude

Sa = Dict()
Sm = Dict()
node_ids = keys(minp)
for nid in node_ids
    Sa[nid] = (maxp[nid] - minp[nid]) / 2
    Sm[nid] = (maxp[nid] + minp[nid]) / 2
end

# create a new xdmf result output and save mean stress and stress amplitude there

using JuliaFEM: new_element, get_all_elements, get_element_type,
                filter_by_element_type, get_element_id

isfile("results_cycle.h5") && rm("results_cycle.h5")
isfile("results_cycle.xmf") && rm("results_cycle.xmf")
xdmf = Xdmf("results_cycle")
domain = new_child(xdmf, "Domain")
frame = new_child(domain, "Grid")

# 1. save geometry
X_ = solver("geometry", solver.time)
node_ids = sort(collect(keys(X_)))
X = hcat([X_[nid] for nid in node_ids]...)
data_node_ids = new_dataitem(xdmf, "/Node IDs", node_ids)
geometry = new_element("Geometry", Dict("Type" => "XYZ"))
add_child(geometry, new_dataitem(xdmf, "/Geometry", X))
add_child(frame, geometry)

# 2. save topology
nid_mapping = Dict(j=>i for (i, j) in enumerate(node_ids))
all_elements = get_all_elements(solver)
nelements = length(all_elements)
debug("Saving topology: $nelements elements total.")
element_types = unique(map(get_element_type, all_elements))

xdmf_element_mapping = Dict(
    "Poi1" => "Polyvertex",
    "Seg2" => "Polyline",
    "Tri3" => "Triangle",
    "Quad4" => "Quadrilateral",
    "Tet4" => "Tetrahedron",
    "Pyramid5" => "Pyramid",
    "Wedge6" => "Wedge",
    "Hex8" => "Hexahedron",
    "Seg3" => "Edge_3",
    "Tri6" => "Tri_6",
    "Quad8" => "Quad_8",
    "Tet10" => "Tet_10",
    "Pyramid13" => "Pyramid_13",
    "Wedge15" => "Wedge_15",
    "Hex20" => "Hex_20")

for element_type in element_types
    elements = filter_by_element_type(element_type, all_elements)
    nelements = length(elements)
    info("Xdmf save: $nelements elements of type $element_type")
    sort!(elements, by=get_element_id)
    element_ids = map(get_element_id, elements)
    element_conn = map(element -> [nid_mapping[j]-1 for j in get_connectivity(element)], elements)
    element_conn = hcat(element_conn...)
    element_code = split(string(element_type), ".")[end]
    dataitem = new_dataitem(xdmf, "/Topology/$element_code/Element IDs", element_ids)
    dataitem = new_dataitem(xdmf, "/Topology/$element_code/Connectivity", element_conn)
    topology = new_element("Topology")
    set_attribute(topology, "TopologyType", xdmf_element_mapping[element_code])
    set_attribute(topology, "NumberOfElements", length(elements))
    add_child(topology, dataitem)
    add_child(frame, topology)
end

add_field_to_xdmf!(xdmf, "Stress (Amplitude)", "/Domain/Grid", "/Results/Stress (Amplitude)", Sa; field_type="Scalar")
add_field_to_xdmf!(xdmf, "Stress (Mean)", "/Domain/Grid", "/Results/Stress (Mean", Sm; field_type="Scalar")

save!(xdmf)
close(xdmf.hdf)

