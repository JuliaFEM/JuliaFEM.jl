# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
End-to-end multi-field tests for the new DOF system.

Exercises the path

    @DOFSet → Element{K, P, S, N} → local_dof_layout → DOFHandler → create_elements!
            → DOFConnectivity → field_dof_range / element_dofs

for two representative coupled DOF specifications:

1. Thermo-mechanical    `(T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex})`
2. Mixed u-p (Stokes)   `(u::DOF{Displacement{3}, Vertex}, p::DOF{Float64, Cell})`

Both verify:
- compile-time DOF count matches `local_dof_layout` length,
- block-ordering of DOFs (all field-1 DOFs first, then field-2),
- per-entity DOF starts in the handler,
- per-element global DOF indices match handler's `field_starts`,
- `field_dof_range(elem, :name)` and `element_dofs(elem, :name)` agree
  with `local_dof_layout`,
- `DOFConnectivity` is consistent with the element's DOF list.
"""

using Test
using JuliaFEM
using JuliaFEM: DOFLayoutEntry, local_dof_layout,
                field_idx, entity_local, component
using JuliaFEM: DOFHandler, create_elements!, @DOFSet, DOF,
                Displacement, Temperature, Vertex, Cell
using JuliaFEM: DOFConnectivity, elem_id, local_dof_idx
using Tensors

# ----------------------------------------------------------------------------
# Tiny meshes used by both testsets
# ----------------------------------------------------------------------------

"Single Tet4 with arbitrary geometry (geometry irrelevant for DOF tests)."
function _tet4_mesh()
    nodes = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.0, 1.0, 0.0)),
        Vec{3}((0.0, 0.0, 1.0)),
    ]
    conn = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]
    return Mesh{Tetrahedron{4}}(nodes, conn)
end

"Two Tet4s sharing the face (2,3,4) so node-shared DOFs are exercised."
function _two_tet4_mesh()
    nodes = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),  # node 1 — only in elem 1
        Vec{3}((1.0, 0.0, 0.0)),  # node 2 — shared
        Vec{3}((0.0, 1.0, 0.0)),  # node 3 — shared
        Vec{3}((0.0, 0.0, 1.0)),  # node 4 — shared
        Vec{3}((1.0, 1.0, 1.0)),  # node 5 — only in elem 2
    ]
    conn = [
        (UInt32(1), UInt32(2), UInt32(3), UInt32(4)),
        (UInt32(2), UInt32(3), UInt32(4), UInt32(5)),
    ]
    return Mesh{Tetrahedron{4}}(nodes, conn)
end

# ============================================================================
# Test 1 — Thermo-mechanical: (T::Vertex, u::Vertex)
# ============================================================================
@testset "Multi-field DOFs: thermo-mechanical (T+u, both Vertex)" begin
    mesh = _two_tet4_mesh()
    n_nodes = length(mesh.nodes)              # 5
    n_elems = length(mesh.connectivity)       # 2

    S  = @DOFSet{T::DOF{Temperature, Vertex},
                 u::DOF{Displacement{3}, Vertex}}
    ET = Element{Tetrahedron{4}, Lagrange{1}, S}

    elements, handler = create_elements!(mesh, ET)
    ET_concrete = eltype(elements)

    # ----------------------------------------------------------------------
    # 1.1 Element template: ndofs and local_dof_layout
    # ----------------------------------------------------------------------
    @test n_element_dofs(elements[1]) == 16   # 4*1 + 4*3
    @test ndofs(Tetrahedron{4}, S) == 16

    layout = local_dof_layout(ET_concrete)
    @test layout isa NTuple{16, DOFLayoutEntry}

    # First 4 entries: field T (idx 1), one component, vertices 1..4
    for k in 1:4
        e = layout[k]
        @test field_idx(e)     == 1
        @test entity_local(e)  == k
        @test component(e)     == 1
    end
    # Remaining 12 entries: field u (idx 2), 3 components per vertex 1..4
    for k in 1:4, c in 1:3
        e = layout[4 + 3*(k-1) + c]
        @test field_idx(e)     == 2
        @test entity_local(e)  == k
        @test component(e)     == c
    end

    # ----------------------------------------------------------------------
    # 1.2 Handler block ordering: total = nT + nu, field_starts contiguous
    # ----------------------------------------------------------------------
    nT = n_nodes * 1
    nU = n_nodes * 3
    @test handler.total_dofs == nT + nU
    @test length(handler.field_starts) == 2

    starts_T = handler.field_starts[1]
    starts_U = handler.field_starts[2]
    @test length(starts_T) == n_nodes
    @test length(starts_U) == n_nodes
    # Field T occupies DOFs 1..nT, one per node
    @test starts_T == collect(1:nT)
    # Field u immediately follows, three per node
    @test starts_U == collect((nT + 1):3:(nT + nU))

    # ----------------------------------------------------------------------
    # 1.3 Per-element DOFs match handler starts
    # ----------------------------------------------------------------------
    for (eid, elem) in enumerate(elements)
        conn = mesh.connectivity[eid]
        dofs = element_dofs(elem)
        @test length(dofs) == 16

        # First 4 = field T at the four nodes of this element
        for k in 1:4
            @test Int(dofs[k]) == starts_T[Int(conn[k])]
        end
        # Next 12 = field u, three components per node
        for k in 1:4, c in 1:3
            @test Int(dofs[4 + 3*(k-1) + c]) == starts_U[Int(conn[k])] + (c - 1)
        end
    end

    # ----------------------------------------------------------------------
    # 1.4 field_dof_range / element_dofs(elem, :field)
    # ----------------------------------------------------------------------
    elem1 = elements[1]
    @test field_dof_range(elem1, :T) == 1:4
    @test field_dof_range(elem1, :u) == 5:16
    @test element_dofs(elem1, :T) == elem1.dof_indices[1:4]
    @test element_dofs(elem1, :u) == elem1.dof_indices[5:16]

    # ----------------------------------------------------------------------
    # 1.5 Type stability of local_dof_layout
    # ----------------------------------------------------------------------
    GC.gc()
    @test (@allocated local_dof_layout(ET_concrete)) == 0

    # ----------------------------------------------------------------------
    # 1.6 DOFConnectivity round-trips through the element's flat DOF list
    # ----------------------------------------------------------------------
    @test handler.dof_connectivity isa DOFConnectivity
    dc = handler.dof_connectivity
    @test dc.n_total_dofs == handler.total_dofs

    for (eid, elem) in enumerate(elements)
        for (li, gdof) in enumerate(elem.dof_indices)
            conns = dc.dof_to_elements[Int(gdof)]
            # The element must appear with the matching local index
            hit = any(c -> elem_id(c) == eid && local_dof_idx(c) == li, conns)
            @test hit
        end
    end
end

# ============================================================================
# Test 2 — Mixed u-p: (u::Vertex, p::Cell), Stokes/incompressible flavour
# ============================================================================
@testset "Multi-field DOFs: mixed u-p (Vertex + Cell)" begin
    mesh = _two_tet4_mesh()
    n_nodes = length(mesh.nodes)              # 5
    n_elems = length(mesh.connectivity)       # 2

    S  = @DOFSet{u::DOF{Displacement{3}, Vertex},
                 p::DOF{Float64, Cell}}
    ET = Element{Tetrahedron{4}, Lagrange{1}, S}

    elements, handler = create_elements!(mesh, ET)
    ET_concrete = eltype(elements)

    # 13 = 4*3 (u at vertices) + 1*1 (p at cell)
    @test n_element_dofs(elements[1]) == 13
    @test ndofs(Tetrahedron{4}, S) == 13

    # ----------------------------------------------------------------------
    # 2.1 Layout: 12 u entries (field 1, vertices 1..4, comps 1..3),
    #             then 1 p entry (field 2, entity_local=1, comp=1).
    # ----------------------------------------------------------------------
    layout = local_dof_layout(ET_concrete)
    @test layout isa NTuple{13, DOFLayoutEntry}
    for k in 1:4, c in 1:3
        e = layout[3*(k-1) + c]
        @test field_idx(e)    == 1
        @test entity_local(e) == k
        @test component(e)    == c
    end
    p_entry = layout[13]
    @test field_idx(p_entry)    == 2
    @test entity_local(p_entry) == 1
    @test component(p_entry)    == 1

    # ----------------------------------------------------------------------
    # 2.2 Handler block ordering: u block first (15 dofs), then p (2 dofs)
    # ----------------------------------------------------------------------
    nU = n_nodes * 3   # 15
    nP = n_elems * 1   # 2
    @test handler.total_dofs == nU + nP
    starts_U = handler.field_starts[1]
    starts_P = handler.field_starts[2]
    @test length(starts_U) == n_nodes
    @test length(starts_P) == n_elems
    @test starts_U == collect(1:3:nU)
    @test starts_P == collect((nU + 1):(nU + nP))

    # ----------------------------------------------------------------------
    # 2.3 Per-element DOFs: u dofs come from per-vertex starts, p dof comes
    #     from per-element start.
    # ----------------------------------------------------------------------
    for (eid, elem) in enumerate(elements)
        conn = mesh.connectivity[eid]
        dofs = element_dofs(elem)

        for k in 1:4, c in 1:3
            @test Int(dofs[3*(k-1) + c]) == starts_U[Int(conn[k])] + (c - 1)
        end
        @test Int(dofs[13]) == starts_P[eid]
    end

    # ----------------------------------------------------------------------
    # 2.4 field_dof_range / element_dofs(elem, :field) for mixed entities
    # ----------------------------------------------------------------------
    elem1 = elements[1]
    @test field_dof_range(elem1, :u) == 1:12
    @test field_dof_range(elem1, :p) == 13:13
    @test element_dofs(elem1, :u) == elem1.dof_indices[1:12]
    @test element_dofs(elem1, :p) == elem1.dof_indices[13:13]

    # ----------------------------------------------------------------------
    # 2.5 The two cell DOFs are *not* shared — even though the elements
    #     share three vertices.
    # ----------------------------------------------------------------------
    p_dof_e1 = Int(elements[1].dof_indices[13])
    p_dof_e2 = Int(elements[2].dof_indices[13])
    @test p_dof_e1 != p_dof_e2

    # The shared u DOFs *are* shared — verify a few node-shared components
    shared_nodes = (2, 3, 4)
    for ns in shared_nodes
        # u-x DOF at node `ns` for both elements must coincide
        i_in_e1 = findfirst(==(UInt32(ns)), mesh.connectivity[1])
        i_in_e2 = findfirst(==(UInt32(ns)), mesh.connectivity[2])
        @test elements[1].dof_indices[3*(i_in_e1 - 1) + 1] ==
              elements[2].dof_indices[3*(i_in_e2 - 1) + 1]
    end

    # ----------------------------------------------------------------------
    # 2.6 Type stability + zero-allocation lookup
    # ----------------------------------------------------------------------
    GC.gc()
    @test (@allocated local_dof_layout(ET_concrete)) == 0
end
