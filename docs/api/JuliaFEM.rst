JuliaFEM
========

Exported
--------

 .. function:: get_coordinates(el::JuliaFEM.Element)

**source**
[JuliaFEM/src/elements.jl:29]

 .. function:: integrate(f::Function)

**source**
[JuliaFEM/src/math.jl:163]

 .. function:: integrate(f::Function,  el::JuliaFEM.Element)

    Integrate f over element using Gaussian quadrature rules.

    :param el: Element
    :param f: Function
**source**
[JuliaFEM/src/math.jl:143]

 .. function:: interpolate(field::Float64,  basis::Function,  ip::Array{Float64, 1})

    Interpolate field variable using basis functions f for point ip.
    This function tries to be as general as possible and allows interpolating
    lot of different fields.

    :param field :  Array{Number, dim}
    :param basis :  Function
    :param ip :  Array{Number, 1}
**source**
[JuliaFEM/src/math.jl:26]

 .. function:: linearize(f::Function,  el::JuliaFEM.Element,  field::ASCIIString)

    Linearize function f w.r.t some given field, i.e. calculate dR/du

    :param f: Function
    :param field: ASCIIString
    :returns: Array{Float64, 2}
                  jacobian / "tangent stiffness matrix"
**source**
[JuliaFEM/src/math.jl:85]

 .. function:: linearize(f::Function,  field::ASCIIString)

**source**
[JuliaFEM/src/math.jl:100]

 .. function:: set_coordinates(el::JuliaFEM.Element,  coordinates)

**source**
[JuliaFEM/src/elements.jl:36]

Internal
--------

 .. function:: add_handler(section,  function_name)

**source**
[JuliaFEM/src/abaqus_reader.jl:13]

 .. function:: create_lagrange_element(element_name,  X,  P,  dP)

**source**
[JuliaFEM/src/elements.jl:64]

 .. function:: get_dbasisdX(el::JuliaFEM.Element,  xi)

**source**
[JuliaFEM/src/elements.jl:20]

 .. function:: get_element_id(el::JuliaFEM.Element)

**source**
[JuliaFEM/src/elements.jl:43]

 .. function:: get_jacobian(el::JuliaFEM.Element,  xi)

**source**
[JuliaFEM/src/elements.jl:9]

 .. function:: integrate!(f::Function,  el::JuliaFEM.Element,  target)

**source**
[JuliaFEM/src/math.jl:178]

 .. function:: linearize!(f::Function,  el::JuliaFEM.Element,  field::ASCIIString,  target::ASCIIString)

**source**
[JuliaFEM/src/math.jl:118]

 .. function:: xdmf_new_model()

    Build a new model for outout

**source**
[JuliaFEM/src/xdmf.jl:48]

 .. function:: xdmf_new_model(xdmf_version)

    Build a new model for outout

**source**
[JuliaFEM/src/xdmf.jl:48]

 .. function:: JuliaFEM.Point1

**source**
[JuliaFEM/src/elements.jl:112]

 .. function:: JuliaFEM.Quad4

**source**
[JuliaFEM/src/elements.jl:155]

 .. function:: JuliaFEM.Seg2

**source**
[JuliaFEM/src/elements.jl:139]

 .. function:: JuliaFEM.Tet10

**source**
[JuliaFEM/src/elements.jl:195]

