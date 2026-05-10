# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

module JuliaFEMWriteVTKExt

using JuliaFEM
using WriteVTK

const _VTK = WriteVTK.VTKCellTypes

function _vtk_cell_type(::Type{T}) where {T}
    if T === Hex8
        return _VTK.VTK_HEXAHEDRON
    elseif T === Tet4
        return _VTK.VTK_TETRA
    elseif T === Quad4
        return _VTK.VTK_QUAD
    elseif T === Tri3
        return _VTK.VTK_TRIANGLE
    elseif T === Seg2
        return _VTK.VTK_LINE
    else
        throw(
            ArgumentError(
                "VTK export does not support topology $T. " *
                "Supported: Hex8, Tet4, Quad4, Tri3, Seg2.",
            ),
        )
    end
end

function _points_matrix(mesh::Mesh{N,T}) where {N,T}
    n = nnodes_total(mesh)
    pts = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        v = mesh.nodes[i]
        pts[1, i] = v[1]
        pts[2, i] = v[2]
        pts[3, i] = v[3]
    end
    return pts
end

function _mesh_cells(mesh::Mesh{N,T}) where {N,T}
    vtk_ct = _vtk_cell_type(T)
    ne = nelements(mesh)
    cells = Vector{MeshCell}(undef, ne)
    @inbounds for e in 1:ne
        conn = mesh.connectivity[e]
        cells[e] = MeshCell(vtk_ct, collect(Int, conn))
    end
    return cells
end

function _strip_vtu_extension(basepath::AbstractString)
    path = String(basepath)
    if endswith(lowercase(path), ".vtu")
        return path[1:(end - 4)]
    end
    return path
end

function _validate_point_data(mesh::Mesh, point_data::NamedTuple)
    n = nnodes_total(mesh)
    for (k, v) in pairs(point_data)
        if v isa AbstractVector
            length(v) == n ||
                throw(ArgumentError("point_data.$k: expected length $n (nnodes), got $(length(v))"))
        elseif v isa AbstractMatrix
            size(v, 2) == n ||
                throw(ArgumentError("point_data.$k: expected matrix with $n columns, got $(size(v))"))
            size(v, 1) in (1, 2, 3) ||
                throw(ArgumentError("point_data.$k: expected 1×n, 2×n, or 3×n matrix, got $(size(v))"))
        else
            throw(ArgumentError("point_data.$k: unsupported type $(typeof(v))"))
        end
    end
    return nothing
end

function _validate_cell_data(mesh::Mesh, cell_data::NamedTuple)
    ne = nelements(mesh)
    for (k, v) in pairs(cell_data)
        v isa AbstractVector ||
            throw(ArgumentError("cell_data.$k: expected AbstractVector, got $(typeof(v))"))
        length(v) == ne ||
            throw(ArgumentError("cell_data.$k: expected length $ne (nelements), got $(length(v))"))
    end
    return nothing
end

function JuliaFEM.write_vtu_mesh(
    basepath::AbstractString,
    mesh::Mesh{N,T};
    point_data::NamedTuple = (;),
    cell_data::NamedTuple = (;),
) where {N,T}
    _vtk_cell_type(T)
    root = _strip_vtu_extension(basepath)
    pts = _points_matrix(mesh)
    cells = _mesh_cells(mesh)
    isempty(point_data) || _validate_point_data(mesh, point_data)
    isempty(cell_data) || _validate_cell_data(mesh, cell_data)

    vtk = vtk_grid(root, pts, cells)
    for (name, vals) in pairs(point_data)
        if vals isa AbstractVector
            vtk[String(name), VTKPointData()] = collect(Float64, vals)
        else
            vtk[String(name), VTKPointData()] = Float64.(vals)
        end
    end
    for (name, vals) in pairs(cell_data)
        vtk[String(name), VTKCellData()] = collect(Float64, vals)
    end
    return first(vtk_save(vtk))
end

end # module
