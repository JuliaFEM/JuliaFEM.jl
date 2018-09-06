# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using AsterReader

"""
Nodes are different order in Code Aster compared to ABAQUS. This is yet
incomplete mapping between the permutations. Most of this is still a
great mystery.

# References
- http://onelab.info/pipermail/gmsh/2008/003850.html
- http://caelinux.org/wiki/index.php/Proj:UNVConvert

"""
const med_connectivity = Dict{Symbol, Vector{Int}}(
    :Tet4   => [4,3,1,2],
    :Tet10  => [4,3,1,2,10,7,8,9,6,5],
    :Pyr5   => [1,4,3,2,5],
    :Wedge6 => [4,5,6,1,2,3],
    :Hex8   => [4,8,7,3,1,5,6,2],
    :Hex20  => [4,8,7,3,1,5,6,2,20,15,19,11,12,16,14,10,17,13,18,9],
    :Hex27  => [4,8,7,3,1,5,6,2,20,15,19,11,12,16,14,10,17,13,18,9,24,25,26,23,21,22,27])

"""
Map element names used in in Code Aster to element names used in in JuliaFEM
"""
const med_element_names = Dict{Symbol, Symbol}(
    :PO1 => :Poi1,
    :SE2 => :Seg2,
    :SE3 => :Seg3,
    :SE4 => :Seg4,
    :TR3 => :Tri3,
    :TR6 => :Tri6,
    :TR7 => :Tri7,
    :QU4 => :Quad4,
    :QU8 => :Quad8,
    :QU9 => :Quad9,
    :TE4 => :Tet4,
    :T10 => :Tet10,
    :PE6 => :Wedge6,
    :P15 => :Wedge15,
    :P18 => :Wedge18,
    :HE8 => :Hex8,
    :H20 => :Hex20,
    :H27 => :Hex27,
    :PY5 => :Pyr5,
    :P13 => :Pyr13)

"""
    aster_read_mesh(filename, mesh_name=nothing; reorder_element_connectivity=true)

Read code aster mesh from file and return `Mesh` structure.

If mesh file contains several meshes, a name of mesh must be given. By default,
elements are reordered so that they match to the conventions used in JuliaFEM.
"""
function aster_read_mesh(filename::String, mesh_name=nothing; reorder_element_connectivity=true)
    m = AsterReader.aster_read_mesh(filename, mesh_name)
    mesh = Mesh(m)
    for (elid, eltype) in mesh.element_types
        mesh.element_types[elid] = med_element_names[eltype]
    end
    if reorder_element_connectivity
        reorder_element_connectivity!(mesh, med_connectivity)
    end
    nnodes = length(mesh.nodes)
    nelements = length(mesh.elements)
    @info("Mesh parsed from Code Aster file $filename.")
    @info("Mesh contains $nnodes nodes and $nelements elements.")
    for (elset_name, elset_elids) in mesh.element_sets
        content = Dict{Symbol, Int}()
        for elid in elset_elids
            eltype = mesh.element_types[elid]
            content[eltype] = get(content, eltype, 0) + 1
        end
        s = join(("$v x $k" for (k, v) in content), ", ")
        nels = length(elset_elids)
        @info("Element set $elset_name contains $nels elements ($s).")
    end
    return mesh
end
