# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar projection calculation for 2d

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master{S,M}(slave::Element{S}, master::Element{M}, xi1::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
#   slave_basis = get_basis(slave)

    # slave side geometry and normal direction at xi1
    X1 = slave("geometry", xi1, time)
    N1 = slave("nodal ntsys", xi1, time)[:,1]

    # master side geometry at xi2
    #master_basis = master.basis.data.basis
    #master_dbasis = master.basis.data.dbasis
    master_basis(xi) = get_basis(M, [xi])
    master_dbasis(xi) = get_dbasis(M, [xi])
    master_geometry = master("geometry")(time)

    function X2(xi2)
        N = master_basis(xi2)
        return sum([N[i]*master_geometry[i] for i=1:length(N)])
    end

    function dX2(xi2)
        dN = master_dbasis(xi2)
        return sum([dN[i]*master_geometry[i] for i=1:length(dN)])
    end

#    master_basis = get_basis(master)
#    X2(xi2) = master_basis("geometry", [xi2], time)
#    dX2(xi2) = dmaster_basis("geometry", xi2, time)

    # equation to solve
    R(xi2) = det([X2(xi2)-X1  N1]')
    dR(xi2) = det([dX2(xi2) N1]')
#   dR = ForwardDiff.derivative(R)

    # go!
    xi2 = 0.0
    for i=1:max_iterations
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < tol
            return Float64[xi2]
        end
    end
    error("find projection from slave to master: did not converge")
end

""" Find projection from master surface to slave point, i.e. find xi1 from slave
element corresponding to the xi2. """
function project_from_master_to_slave{S,M}(slave::Element{S}, master::Element{M}, xi2::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
#   slave_basis = get_basis(slave)

    # slave side geometry and normal direction at xi1

    slave_geometry = slave("geometry")(time)
    slave_normals = slave("nodal ntsys")(time)
    #slave_basis = slave.basis.data.basis
    #slave_dbasis = slave.basis.data.dbasis
    slave_basis(xi) = get_basis(S, [xi])
    slave_dbasis(xi) = get_dbasis(S, [xi])

    function X1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_geometry[i] for i=1:length(N)])
    end

    function dX1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_geometry[i] for i=1:length(dN)])
    end

    function N1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_normals[i] for i=1:length(N)])[:,1]
    end

    function dN1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_normals[i] for i=1:length(dN)])[:,1]
    end

    #X1(xi1) = slave_basis("geometry", [xi1], time)
    #N1(xi1) = slave_basis("nodal ntsys", [xi1], time)[:,1]

    #master_basis = get_basis(master)

    # master side geometry at xi2
    #X2 = master_basis("geometry", xi2, time)
    X2 = master("geometry", xi2, time)

    # equation to solve
    R(xi1) = det([X1(xi1)-X2  N1(xi1)]')
    dR(xi1) = det([dX1(xi1) N1(xi1)]') + det([X1(xi1)-X2 dN1(xi1)]')

    #=
    info("R(-1.0) = $(R(-1.0))")
    info("R( 0.0) = $(R(0.0))")
    info("R( 1.0) = $(R(1.0))")
    info("R( 1.5) = $(R(1.5))")
    info("dR(-1.0) = $(dR(-1.0))")
    info("dR( 0.0) = $(dR(0.0))")
    info("dR( 1.0) = $(dR(1.0))")
    info("dR( 1.5) = $(dR(1.5))")
    =#

    #dR = ForwardDiff.derivative(R)

    # go!
    xi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1) / dR(xi1)
        xi1 += dxi1
        #info("dxi1 = $dxi1, xi1 = $xi1, norm(dxi1) = $(norm(dxi1))")
        if norm(dxi1) < tol
            return Float64[xi1]
        end
    end
    error("find projection from master to slave: did not converge")
end

### Mortar problem

"""
Parameters
----------
node_csys
    coordinate system in node, normal + tangent + "binormal"
    in 3d 3x3 matrix, in 2d 2x2 matrix, respectively
"""
abstract MortarProblem <: AbstractProblem

function MortarProblem(parent_field_name, parent_field_dim, dim=1, elements=[])
    return BoundaryProblem{MortarProblem}(parent_field_name, parent_field_dim, dim, elements)
end

# Mortar assembly

function assemble!(assembly::Assembly, problem::BoundaryProblem{MortarProblem}, slave_element::Element, time::Number)
    
    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name

    slave_dofs = get_gdofs(slave_element, field_dim)

    for master_element in slave_element["master elements"]
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0])
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0])
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*(xi1[2]-xi1[1])
        if abs(l) < 1.0e-6
            warn("No contribution")
            continue # no contribution
        end
        master_dofs = get_gdofs(master_element, field_dim)
        for ip in get_integration_points(slave_element)
            w = ip.weight*det(slave_element, ip, time)*l

            # integration point on slave side segment
            xi_gauss = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point
            xi_projected = project_from_slave_to_master(slave_element, master_element, xi_gauss)

            # add contribution to left hand side 
            N1 = slave_element(xi_gauss, time)
            N2 = master_element(xi_projected, time)
            S = w*N1'*N1
            M = w*N1'*N2
            for i=1:field_dim
                sd = slave_dofs[i:field_dim:end]
                md = master_dofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, sd, sd, S)
                add!(assembly.stiffness_matrix, sd, md, -M)
            end

        end
    end
end

