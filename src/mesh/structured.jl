# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    create_structured_box_mesh(::Type{Hex8}; 
                               xmin=0.0, xmax=1.0, nx=1,
                               ymin=0.0, ymax=1.0, ny=1,
                               zmin=0.0, zmax=1.0, nz=1) -> Mesh{Hex8}

Create a structured box mesh with Hex8 elements.

# Arguments
- `::Type{Hex8}`: Structured brick topology only (no structured `Tet4` box mesh in this entry point)

# Keyword Arguments
- `xmin`, `xmax`: Domain bounds in X direction (default: 0.0, 1.0)
- `ymin`, `ymax`: Domain bounds in Y direction (default: 0.0, 1.0)
- `zmin`, `zmax`: Domain bounds in Z direction (default: 0.0, 1.0)
- `nx`: Number of elements in X direction (default: 1)
- `ny`: Number of elements in Y direction (default: 1)
- `nz`: Number of elements in Z direction (default: 1)

# Returns
- `Mesh{Hex8}`: Structured mesh with element sets and node sets for all boundaries

# Element Sets
- `:all`: All elements in the mesh

# Node Sets
- `:all`: All nodes in the mesh
- `:xmin`: Nodes on the x=xmin face
- `:xmax`: Nodes on the x=xmax face
- `:ymin`: Nodes on the y=ymin face
- `:ymax`: Nodes on the y=ymax face
- `:zmin`: Nodes on the z=zmin face
- `:zmax`: Nodes on the z=zmax face

# Examples

```julia
# Unit cube with 1 element
mesh = create_structured_box_mesh(Hex8)

# Cantilever beam: 10x2x2 domain with 10x2x2 elements
mesh = create_structured_box_mesh(Hex8, 
    xmin=0.0, xmax=10.0, nx=10,
    ymin=0.0, ymax=2.0, ny=2,
    zmin=0.0, zmax=2.0, nz=2)

# Fine mesh in one direction
mesh = create_structured_box_mesh(Hex8, 
    xmin=0.0, xmax=1.0, nx=20,
    ymin=0.0, ymax=1.0, ny=4,
    zmin=0.0, zmax=1.0, nz=4)

# Convergence study
for n in [2, 4, 8, 16]
    mesh = create_structured_box_mesh(Hex8, nx=n, ny=n, nz=n)
    result = solve_problem(mesh)
    println("Mesh \$(n)^3: error = \$(result.error)")
end
```

See also: [`Mesh`](@ref), [`refine`](@ref)
"""
function create_structured_box_mesh(
    ::Type{Hex8};
    xmin::Float64=0.0, xmax::Float64=1.0, nx::Int=1,
    ymin::Float64=0.0, ymax::Float64=1.0, ny::Int=1,
    zmin::Float64=0.0, zmax::Float64=1.0, nz::Int=1
)
    @assert nx ≥ 1 "Number of elements in X must be ≥ 1"
    @assert ny ≥ 1 "Number of elements in Y must be ≥ 1"
    @assert nz ≥ 1 "Number of elements in Z must be ≥ 1"
    @assert xmax > xmin "xmax must be > xmin"
    @assert ymax > ymin "ymax must be > ymin"
    @assert zmax > zmin "zmax must be > zmin"

    # Create structured grid of nodes
    nodes_x = range(xmin, xmax, length=nx + 1)
    nodes_y = range(ymin, ymax, length=ny + 1)
    nodes_z = range(zmin, zmax, length=nz + 1)

    # Generate nodes in IJK order
    nodes = Vec{3,Float64}[]
    for k in 1:(nz+1), j in 1:(ny+1), i in 1:(nx+1)
        push!(nodes, Vec(nodes_x[i], nodes_y[j], nodes_z[k]))
    end

    # Node indexing helper: (i,j,k) -> global node index
    # Nodes are stored in column-major order: i varies fastest, then j, then k
    function node_index(i::Int, j::Int, k::Int)
        return UInt32((k - 1) * (nx + 1) * (ny + 1) + (j - 1) * (nx + 1) + i)
    end

    # Generate Hex8 connectivity
    # Hex8 node ordering (local):
    #   8-------7
    #  /|      /|
    # 5-------6 |
    # | |     | |
    # | 4-----|-3
    # |/      |/
    # 1-------2
    #
    # Bottom face (z=zmin): 1-2-3-4
    # Top face (z=zmax):    5-6-7-8
    connectivity = NTuple{8,UInt32}[]

    for k in 1:nz, j in 1:ny, i in 1:nx
        # Element corners in IJK space
        n1 = node_index(i, j, k)    # Bottom-left-front
        n2 = node_index(i + 1, j, k)    # Bottom-right-front
        n3 = node_index(i + 1, j + 1, k)    # Bottom-right-back
        n4 = node_index(i, j + 1, k)    # Bottom-left-back
        n5 = node_index(i, j, k + 1)  # Top-left-front
        n6 = node_index(i + 1, j, k + 1)  # Top-right-front
        n7 = node_index(i + 1, j + 1, k + 1)  # Top-right-back
        n8 = node_index(i, j + 1, k + 1)  # Top-left-back

        push!(connectivity, (n1, n2, n3, n4, n5, n6, n7, n8))
    end

    # Create element sets
    element_sets = Dict{Symbol,Set{UInt32}}(
        :all => Set(UInt32(1):UInt32(length(connectivity)))
    )

    # Create node sets for boundary faces
    node_sets = Dict{Symbol,Set{UInt32}}()

    # All nodes
    node_sets[:all] = Set(UInt32(1):UInt32(length(nodes)))

    # X boundaries
    xmin_nodes = Set{UInt32}()
    xmax_nodes = Set{UInt32}()
    for k in 1:(nz+1), j in 1:(ny+1)
        push!(xmin_nodes, node_index(1, j, k))      # i=1
        push!(xmax_nodes, node_index(nx + 1, j, k))   # i=nx+1
    end
    node_sets[:xmin] = xmin_nodes
    node_sets[:xmax] = xmax_nodes

    # Y boundaries
    ymin_nodes = Set{UInt32}()
    ymax_nodes = Set{UInt32}()
    for k in 1:(nz+1), i in 1:(nx+1)
        push!(ymin_nodes, node_index(i, 1, k))      # j=1
        push!(ymax_nodes, node_index(i, ny + 1, k))   # j=ny+1
    end
    node_sets[:ymin] = ymin_nodes
    node_sets[:ymax] = ymax_nodes

    # Z boundaries
    zmin_nodes = Set{UInt32}()
    zmax_nodes = Set{UInt32}()
    for j in 1:(ny+1), i in 1:(nx+1)
        push!(zmin_nodes, node_index(i, j, 1))      # k=1
        push!(zmax_nodes, node_index(i, j, nz + 1))   # k=nz+1
    end
    node_sets[:zmin] = zmin_nodes
    node_sets[:zmax] = zmax_nodes

    return Mesh{Hex8}(nodes, connectivity; element_sets=element_sets, node_sets=node_sets)
end

"""
    create_unit_cube_mesh(::Type{Hex8}; nx=1, ny=1, nz=1) -> Mesh{Hex8}

Create a structured unit cube mesh [0,1]^3 with Hex8 elements.

Convenience wrapper around `create_structured_box_mesh` for the common case
of a unit cube domain.

# Arguments
- `::Type{Hex8}`: Element topology type

# Keyword Arguments
- `nx`: Number of elements in X direction (default: 1)
- `ny`: Number of elements in Y direction (default: 1)
- `nz`: Number of elements in Z direction (default: 1)

# Returns
- `Mesh{Hex8}`: Structured mesh of unit cube with boundary node sets

# Example
```julia
# Single element unit cube
mesh = create_unit_cube_mesh(Hex8)

# Fine discretization: 10x10x10 elements
mesh = create_unit_cube_mesh(Hex8, nx=10, ny=10, nz=10)

# Anisotropic mesh: fine in X, coarse in Y and Z
mesh = create_unit_cube_mesh(Hex8, nx=20, ny=4, nz=4)
```

See also: [`create_structured_box_mesh`](@ref)
"""
function create_unit_cube_mesh(::Type{Hex8}; nx::Int=1, ny::Int=1, nz::Int=1)
    return create_structured_box_mesh(Hex8,
        xmin=0.0, xmax=1.0, nx=nx,
        ymin=0.0, ymax=1.0, ny=ny,
        zmin=0.0, zmax=1.0, nz=nz)
end

"""
    create_cantilever_mesh(::Type{Hex8}; 
                          length=10.0, width=2.0, height=2.0,
                          nx=10, ny=2, nz=2) -> Mesh{Hex8}

Create a structured mesh for a cantilever beam.

This is a convenience function that creates a rectangular box mesh with
dimensions suitable for cantilever beam problems. The mesh is oriented
along the X-axis (length direction).

# Arguments
- `::Type{Hex8}`: Element topology type

# Keyword Arguments
- `length`: Length in X direction (default: 10.0)
- `width`: Width in Y direction (default: 2.0)
- `height`: Height in Z direction (default: 2.0)
- `nx`: Number of elements along length (default: 10)
- `ny`: Number of elements along width (default: 2)
- `nz`: Number of elements along height (default: 2)

# Returns
- `Mesh{Hex8}`: Structured mesh with boundary node sets
  - `:xmin` typically used for fixed boundary condition
  - `:xmax` typically used for applied load

# Example
```julia
# Standard cantilever: L/h = 5 with moderate discretization
mesh = create_cantilever_mesh(Hex8, 
    length=10.0, width=2.0, height=2.0,
    nx=20, ny=4, nz=4)

# Apply boundary conditions
fixed_nodes = get_nodes_in_set(mesh, :xmin)    # Fixed end
loaded_nodes = get_nodes_in_set(mesh, :xmax)   # Free end (apply load)

# Convergence study
for n in [5, 10, 20, 40]
    mesh = create_cantilever_mesh(Hex8, nx=n, ny=n÷5, nz=n÷5)
    result = solve_elasticity(mesh)
    println("nx=\$n: tip deflection = \$(result.tip_displacement)")
end
```

See also: [`create_structured_box_mesh`](@ref), [`create_unit_cube_mesh`](@ref)
"""
function create_cantilever_mesh(
    ::Type{Hex8};
    length::Float64=10.0,
    width::Float64=2.0,
    height::Float64=2.0,
    nx::Int=10,
    ny::Int=2,
    nz::Int=2
)
    return create_structured_box_mesh(Hex8,
        xmin=0.0, xmax=length, nx=nx,
        ymin=0.0, ymax=width, ny=ny,
        zmin=0.0, zmax=height, nz=nz)
end

"""
    create_thin_plate_mesh(::Type{Hex8};
                          length=10.0, width=10.0, thickness=0.1,
                          nx=10, ny=10, nz=1) -> Mesh{Hex8}

Create a structured mesh for a thin plate.

This creates a rectangular box mesh with small thickness dimension (Z),
suitable for plate bending problems or thin-walled structures.

# Arguments
- `::Type{Hex8}`: Element topology type

# Keyword Arguments
- `length`: Length in X direction (default: 10.0)
- `width`: Width in Y direction (default: 10.0)
- `thickness`: Thickness in Z direction (default: 0.1)
- `nx`: Number of elements along length (default: 10)
- `ny`: Number of elements along width (default: 10)
- `nz`: Number of elements through thickness (default: 1)

# Returns
- `Mesh{Hex8}`: Structured mesh with boundary node sets

# Example
```julia
# Square plate 10x10x0.1
mesh = create_thin_plate_mesh(Hex8,
    length=10.0, width=10.0, thickness=0.1,
    nx=20, ny=20, nz=1)

# Use `:zmin` and `:zmax` for top/bottom surfaces
bottom_nodes = get_nodes_in_set(mesh, :zmin)
top_nodes = get_nodes_in_set(mesh, :zmax)
```

See also: [`create_structured_box_mesh`](@ref)
"""
function create_thin_plate_mesh(
    ::Type{Hex8};
    length::Float64=10.0,
    width::Float64=10.0,
    thickness::Float64=0.1,
    nx::Int=10,
    ny::Int=10,
    nz::Int=1
)
    return create_structured_box_mesh(Hex8,
        xmin=0.0, xmax=length, nx=nx,
        ymin=0.0, ymax=width, ny=ny,
        zmin=0.0, zmax=thickness, nz=nz)
end
