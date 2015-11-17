# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar elements

# slave element = non-mortar element where integration happens
# master element = mortar element projected to non-mortar side

abstract MortarElement <: Element

type MSeg2 <: MortarElement
    connectivity :: Vector{Int}
    basis :: Basis
    fields :: FieldSet
    master_elements :: Vector{MortarElement}
end

function MSeg2(connectivity, master_elements=[], biorthogonal=false)
    basis(xi) = [(1-xi[1])/2 (1+xi[1])/2]
    dbasisdxi(xi) = [-1/2 1/2]
    return MSeg2(connectivity, Basis(basis, dbasisdxi), FieldSet(), master_elements)
end

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master(slave::MortarElement, master::MortarElement, xi1::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
    slave_basis = get_basis(slave)
    master_basis = get_basis(master)

    # slave side geometry and normal direction at xi1
    X1 = slave_basis("geometry", xi1, time)
    N1 = slave_basis("nodal ntsys", xi1, time)[:,1]

    # master side geometry at xi2
    X2(xi2) = master_basis("geometry", [xi2], time)
#   dX2(xi2) = dmaster_basis("geometry", xi2, time)

    # equation to solve
    R(xi2) = det([X2(xi2)-X1  N1]')
#   dR(xi2) = det([dX2(xi2) N1]')
    dR = ForwardDiff.derivative(R)

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

""" Find projection from master surface to slave point, i.e. find xi1 from slave element corresponding to the xi2. """
function project_from_master_to_slave(slave::MortarElement, master::MortarElement, xi2::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
    slave_basis = get_basis(slave)
    master_basis = get_basis(master)

    # slave side geometry and normal direction at xi1
    X1(xi1) = slave_basis("geometry", [xi1], time)
    N1(xi1) = slave_basis("nodal ntsys", [xi1], time)[:,1]

    # master side geometry at xi2
    X2 = master_basis("geometry", xi2, time)

    # equation to solve
    R(xi1) = det([X1(xi1)-X2  N1(xi1)]')
#   dR(xi1) = det([dX1(xi1) N1(xi1)]') + det([X1(xi1)-X2 dN1(xi1)]')
    dR = ForwardDiff.derivative(R)

    # go!
    xi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1) / dR(xi1)
        xi1 += dxi1
        if norm(dxi1) < tol
            return Float64[xi1]
        end
    end
    error("find projection from master to slave: did not converge")
end

