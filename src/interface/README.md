# src/interface/

First-class interface geometry and DOF numbering for coupled problems (mortar
contact, FSI traces, interface reactions, surface chemistry), kept separate from
volume `Mesh` / `DOFHandler`.

Types:

- `AbstractInterfaceMesh`, `InterfaceMesh` — embedded polygonal or segment grids
  (`Seg2` first) plus per-segment `InterfaceVolumeCoupling` to slave/master volume
  elements.
- `InterfaceDOFHandler`, `create_interface_elements!` — same assignment discipline as
  volume handlers (`Vertex` / `Cell` on the interface mesh only; no `Edge`/`Face`
  until facet maps on interfaces exist).

Mortar quadrature, projection kernels, and composition with volume operators are
implemented on top of these structures in future work.
