# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using AbaqusReader

function abaqus_read_mesh(fn::String)
    m = AbaqusReader.abaqus_read_mesh(fn)
    return Mesh(m)
end

""" 
    create_surface_elements(mesh::Mesh, surface_name::Symbol)

Create a set of surface elements from solid elements.

Notation follow what is defined in ABAQUS. For example, if solid elements
are Tet10, surface elements will be Tri6 and they can be used to define
boundary conditions.
"""
function create_surface_elements(mesh::Mesh, surface_name::Symbol)
    elements = Element[]
    for (elid, elsi) in mesh.surface_sets[surface_name]
        elty = mesh.element_types[elid]
        elco = mesh.elements[elid]
        chel, chcon = AbaqusReader.create_surface_element(elty, elsi, elco)
        ch = Element(getfield(JuliaFEM, chel), chcon)
        push!(elements, ch)
    end
    update!(elements, "geometry", mesh.nodes)
    return elements
end

""" 
    create_surface_elements(mesh::Mesh, surface_name::String)

Create a set of surface elements from solid elements.

Notation follow what is defined in ABAQUS. For example, if solid elements
are Tet10, surface elements will be Tri6 and they can be used to define
boundary conditions.
"""
function create_surface_elements(mesh::Mesh, surface_name::String)
    return create_surface_elements(mesh, Symbol(surface_name))
end
