# JuliaFEM

## Exported

---

<a id="method__add_elements.1" class="lexicon_definition"></a>
#### add_elements(model,  elements) [¶](#method__add_elements.1)
Add new elements to model.
Parameters
----------
list of dicts, dict = {element_type => elcode, elids => [node ids..]}

Examples
--------
Create two tet4 element and add them:

>>> m = new_model()
>>> const TET4 = 0x6
>>> el1 = Dict("element_type" => TET4, "node_ids" => [1, 2, 3, 4])
>>> el2 = Dict("element_type" => TET4, "node_ids" => [4, 3, 2, 1])

In dict key means element id

>>> elements = Dict(1 => el1, 2 => el2)
>>> add_elements(m, elements)



*source:*
[JuliaFEM/src/JuliaFEM.jl:102](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/JuliaFEM.jl#L102)

---

<a id="method__get_elements.1" class="lexicon_definition"></a>
#### get_elements(model,  element_ids) [¶](#method__get_elements.1)
Get subset of elements from model.
Parameters
----------
element_ids : list of ints
    Element id numbers

Returns
-------
Dict
{element_type = XXX, node_ids = [a, b, c, d, e, ..., n]}


*source:*
[JuliaFEM/src/JuliaFEM.jl:126](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/JuliaFEM.jl#L126)

---

<a id="method__get_field.1" class="lexicon_definition"></a>
#### get_field(field_type,  field_name) [¶](#method__get_field.1)
Get field from model.

Parameters
----------
field_type : Dict()
    Target topology (model.model, model.nodes, model.elements,
                     model.element_nodes, model.element_gauss
field_name : str
    Field name

create_if_doesnt_exist : bool, optional
    If field doesn't exists, create one and return empty field

Raises
------
Error, if field not found and create_if_doesnt_exist == false


*source:*
[JuliaFEM/src/JuliaFEM.jl:67](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/JuliaFEM.jl#L67)

---

<a id="method__new_model.1" class="lexicon_definition"></a>
#### new_model() [¶](#method__new_model.1)
Initialize empty model.

Parameters
----------
None

Returns
-------
New model struct


*source:*
[JuliaFEM/src/JuliaFEM.jl:41](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/JuliaFEM.jl#L41)

---

<a id="type__model.1" class="lexicon_definition"></a>
#### JuliaFEM.Model [¶](#type__model.1)
Basic model


*source:*
[JuliaFEM/src/JuliaFEM.jl:19](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/JuliaFEM.jl#L19)

