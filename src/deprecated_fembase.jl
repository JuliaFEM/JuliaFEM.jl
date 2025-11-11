# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

"""
    length(element)

Return the length of basis (number of nodes).
"""
function length(element::AbstractElement)
    return length(element.connectivity)
end

"""
    size(element)

Return the size of basis (dim, nnodes).
"""
function size(element::AbstractElement)
    # Return (dimension, number of nodes)
    # For now, infer dimension from basis type
    conn = element.connectivity
    nnodes = length(conn)
    # This is a simplified version - proper dimension should come from basis
    return (3, nnodes)  # Default to 3D for now
end

function getindex(element::AbstractElement, field_name::String)
    return get_field(element, Symbol(field_name))
end

function setindex!(element::AbstractElement, data::T, field_name) where T<:AbstractField
    element.fields[field_name] = data
end

function setindex!(element::AbstractElement, data::Function, field_name)
    if hasmethod(data, Tuple{AbstractElement,Vector,Float64})
        # create enclosure to pass element as argument
        element.fields[field_name] = field((ip, time) -> data(element, ip, time))
    else
        element.fields[field_name] = field(data)
    end
end

function setindex!(element::AbstractElement, data, field_name)
    element.dfields[field_name] = field(data)
end

function setindex!(fieldset::Dict{Symbol,AbstractField}, field_data, field_name::String)
    setindex!(fieldset, field_data, Symbol(field_name))
end

function has_field(element, field_name::String)
    return has_field(element, Symbol(field_name))
end

#""" Return a Field object from element.
#Examples
#--------
#>>> element = Element(Seg2, [1, 2])
#>>> data = Dict(1 => 1.0, 2 => 2.0)
#>>> update!(element, "my field", data)
#>>> element("my field")
#"""
function (element::Element)(field_name::String)
    return element[field_name]
end

function size(element::AbstractElement, dim)
    return size(element)[dim]
end

""" Check existence of field. """
function haskey(element::AbstractElement, field_name)
    return has_field(element, field_name)
end

# will be deprecated
function assemble!(::Assembly, ::Problem{P}, ::AbstractElement, ::Any) where P
    @warn("One must define assemble! function for problem of type $P. " *
          "Not doing anything.")
    return nothing
end

# will be deprecated
function assemble!(assembly::Assembly, problem::Problem{P},
    elements::Vector{Element}, time) where P
    @warn("This is default assemble! function. Decreased performance can be " *
          "expected without preallocation of memory. One should implement " *
          "`assemble_elements!(problem, assembly, elements, time)` function.")
    for element in elements
        assemble!(assembly, problem, element, time)
    end
    return nothing
end

# generally a bad idea to have functions like update! and interpolate, which
# are not explicitly giving targe?

function update!(element::AbstractElement, field_name, field_data)
    error("""
    update!() is DEPRECATED for immutable Element architecture!

    Elements are now immutable. Use update() instead, which returns a NEW element:

    OLD (mutable):
        element = Element(Seg2, (1, 2))
        update!(element, "geometry", X)  # Mutates element

    NEW (immutable):
        element = Element(Seg2, (1, 2))
        element = update(element, :geometry => X)  # Returns new element

    OR create element with fields from the start:
        element = Element(Seg2, (1, 2), fields=(geometry=X, temperature_1=0.0))

    See docs/book/migration-guide-element-api.md for complete migration guide.
    """)
end

function update!(field::AbstractField, data)
    update_field!(field, data)
end

function update!(element::AbstractElement, field_name::String, field_data)
    # Convert string to symbol and call the error-throwing version
    update!(element, Symbol(field_name), field_data)
end

function update!(elements::Vector{Element}, field_name::String, field_data)
    update_field!(elements, Symbol(field_name), field_data)
end

function update!(elements::Vector{Element{T}}, field_name::String, field_data) where T
    update_field!(convert(Vector{Element}, elements), field_name, field_data)
end

function interpolate(element::AbstractElement, field_name::String, time)
    interpolate_field(element, Symbol(field_name), time)
end

function interpolate(element::AbstractElement, field_name::String, ip, time)
    interpolate(element, Symbol(field_name), ip, time)
end

# OLD: fields is now dfields - DEPRECATED with new Element{N,NIP,F,B} architecture
# New Element struct has `fields::F` directly, no need for redirection
# function Base.getproperty(element::Element, sym::Symbol)
#     if sym === :fields  # Use === instead of == (which is overridden by fields.jl)
#         return getfield(element, :dfields)
#     else
#         return getfield(element, sym)
#     end
# end

# getindex, when someone asks key element.fields["f"] => element.fields[:f]
function Base.getindex(fieldset::Dict{Symbol,AbstractField}, field_name::String)
    return getindex(fieldset, Symbol(field_name))
end

