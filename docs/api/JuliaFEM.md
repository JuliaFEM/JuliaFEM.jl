# JuliaFEM

## Exported

---

<a id="method__get_coordinates.1" class="lexicon_definition"></a>
#### get_coordinates(el::JuliaFEM.Element) [¶](#method__get_coordinates.1)
Return coordinates of element in array of size dim x nnodes


*source:*
[JuliaFEM/src/elements.jl:29](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L29)

---

<a id="method__integrate.2" class="lexicon_definition"></a>
#### integrate(f::Function) [¶](#method__integrate.2)
This version returns a function which must be operated with element e


*source:*
[JuliaFEM/src/math.jl:163](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L163)

---

<a id="method__integrate.3" class="lexicon_definition"></a>
#### integrate(f::Function,  el::JuliaFEM.Element) [¶](#method__integrate.3)
Integrate f over element using Gaussian quadrature rules.

Parameters
----------
el::Element
    well defined element
f::Function
    Function to integrate


*source:*
[JuliaFEM/src/math.jl:143](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L143)

---

<a id="method__interpolate.1" class="lexicon_definition"></a>
#### interpolate(field::Float64,  basis::Function,  ip::Array{Float64, 1}) [¶](#method__interpolate.1)
Interpolate field variable using basis functions f for point ip.
This function tries to be as general as possible and allows interpolating
lot of different fields.

Parameters
----------
field :: Array{Number, dim}
  Field variable
basis :: Function
  Basis functions
ip :: Array{Number, 1}
  Point to interpolate


*source:*
[JuliaFEM/src/math.jl:26](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L26)

---

<a id="method__linearize.2" class="lexicon_definition"></a>
#### linearize(f::Function,  el::JuliaFEM.Element,  field::ASCIIString) [¶](#method__linearize.2)
Linearize function f w.r.t some given field, i.e. calculate dR/du

Parameters
----------
f::Function
    (possibly) nonlinear function to linearize
field::ASCIIString
    field variable

Returns
-------
Array{Float64, 2}
    jacobian / "tangent stiffness matrix"



*source:*
[JuliaFEM/src/math.jl:85](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L85)

---

<a id="method__linearize.3" class="lexicon_definition"></a>
#### linearize(f::Function,  field::ASCIIString) [¶](#method__linearize.3)
This version returns another function which can be then evaluated against field


*source:*
[JuliaFEM/src/math.jl:100](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L100)

---

<a id="method__set_coordinates.1" class="lexicon_definition"></a>
#### set_coordinates(el::JuliaFEM.Element,  coordinates) [¶](#method__set_coordinates.1)
Set coordinates for element


*source:*
[JuliaFEM/src/elements.jl:36](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L36)

## Internal

---

<a id="method__add_handler.1" class="lexicon_definition"></a>
#### add_handler(section,  function_name) [¶](#method__add_handler.1)
Register new handler for parser


*source:*
[JuliaFEM/src/abaqus_reader.jl:13](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/abaqus_reader.jl#L13)

---

<a id="method__create_lagrange_element.1" class="lexicon_definition"></a>
#### create_lagrange_element(element_name,  X,  P,  dP) [¶](#method__create_lagrange_element.1)
Create new Lagrange element

FIXME: this is not working

LoadError: error compiling anonymous: type definition not allowed inside a local scope

It's the for loop which is causing problems. See
https://github.com/JuliaLang/julia/issues/10555



*source:*
[JuliaFEM/src/elements.jl:64](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L64)

---

<a id="method__get_dbasisdx.1" class="lexicon_definition"></a>
#### get_dbasisdX(el::JuliaFEM.Element,  xi) [¶](#method__get_dbasisdx.1)
Evaluate partial derivatives of basis function w.r.t
material description X, i.e. dbasis/dX


*source:*
[JuliaFEM/src/elements.jl:20](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L20)

---

<a id="method__get_element_id.1" class="lexicon_definition"></a>
#### get_element_id(el::JuliaFEM.Element) [¶](#method__get_element_id.1)
Get element id


*source:*
[JuliaFEM/src/elements.jl:43](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L43)

---

<a id="method__get_jacobian.1" class="lexicon_definition"></a>
#### get_jacobian(el::JuliaFEM.Element,  xi) [¶](#method__get_jacobian.1)
Get jacobian of element evaluated at point xi


*source:*
[JuliaFEM/src/elements.jl:9](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L9)

---

<a id="method__integrate.1" class="lexicon_definition"></a>
#### integrate!(f::Function,  el::JuliaFEM.Element,  target) [¶](#method__integrate.1)
This version saves results inplace to target, garbage collection free


*source:*
[JuliaFEM/src/math.jl:178](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L178)

---

<a id="method__linearize.1" class="lexicon_definition"></a>
#### linearize!(f::Function,  el::JuliaFEM.Element,  field::ASCIIString,  target::ASCIIString) [¶](#method__linearize.1)
In-place version, no additional garbage collection.


*source:*
[JuliaFEM/src/math.jl:118](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/math.jl#L118)

---

<a id="method__xdmf_new_model.1" class="lexicon_definition"></a>
#### xdmf_new_model() [¶](#method__xdmf_new_model.1)
Build a new model for outout

Parameters
----------

Example
-------
    >>> xdmf_new_model()


*source:*
[JuliaFEM/src/xdmf.jl:48](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/xdmf.jl#L48)

---

<a id="method__xdmf_new_model.2" class="lexicon_definition"></a>
#### xdmf_new_model(xdmf_version) [¶](#method__xdmf_new_model.2)
Build a new model for outout

Parameters
----------

Example
-------
    >>> xdmf_new_model()


*source:*
[JuliaFEM/src/xdmf.jl:48](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/xdmf.jl#L48)

---

<a id="type__point1.1" class="lexicon_definition"></a>
#### JuliaFEM.Point1 [¶](#type__point1.1)
1 node point element


*source:*
[JuliaFEM/src/elements.jl:112](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L112)

---

<a id="type__quad4.1" class="lexicon_definition"></a>
#### JuliaFEM.Quad4 [¶](#type__quad4.1)
4 node bilinear quadrangle element


*source:*
[JuliaFEM/src/elements.jl:155](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L155)

---

<a id="type__seg2.1" class="lexicon_definition"></a>
#### JuliaFEM.Seg2 [¶](#type__seg2.1)
3 node quadratic line element


*source:*
[JuliaFEM/src/elements.jl:139](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L139)

---

<a id="type__tet10.1" class="lexicon_definition"></a>
#### JuliaFEM.Tet10 [¶](#type__tet10.1)
10 node quadratic tethahedron


*source:*
[JuliaFEM/src/elements.jl:195](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elements.jl#L195)

