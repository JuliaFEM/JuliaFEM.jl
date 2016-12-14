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

function create_interface(mesh, slave_surface::String, master_surface::String;
dual_basis=true)
    interface = Problem(Mortar, "interface between $slave_surface and $master_surface", 3, "displacement")
    interface.properties.dual_basis = dual_basis
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
mesh = abaqus_read_mesh("flange_coupling_tie_contact.inp")
#to_linear!(mesh)
info("element sets = ", collect(keys(mesh.element_sets)))
info("surface sets = ", collect(keys(mesh.surface_sets)))

# parts
male = create_body(mesh, "MALE", 210.0e3, 0.3, 7.85e-9)
female = create_body(mesh, "FEMALE", 210.0e3, 0.3, 7.85e-9)

# boundary conditions
bc1 = create_bc(mesh, "MALE_FIXED")

# load
load = Problem(Elasticity, "pressure load", 3)
load.elements = create_surface_elements(mesh, :FEMALE_FIXED)
update!(load, "surface pressure", 20.0)

contact = create_interface(mesh, "MALE_TO_FEMALE", "FEMALE_TO_MALE")

# solution
isfile("results.h5") && rm("results.h5")
isfile("results.xmf") && rm("results.xmf")
solver = Solver(Linear, male, female, bc1, load, contact)
solver.xdmf = Xdmf("results")
solver()

