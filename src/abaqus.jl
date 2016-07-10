# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

importall Base
using JuliaFEM.Preprocess

global const ABAQUS_SECTIONS = [
    "HEADING", "NODE", "ELEMENT", "SOLID SECTION",
    "MATERIAL", "NSET", "SURFACE", "STEP"]

global const ABAQUS_SUBSECTIONS = [
    "ELASTIC", "DENSITY", "SPECIFIC HEAT", "CONDUCTIVITY",
    "STATIC", "BOUNDARY", "DSLOAD", "OUTPUT", "NODE FILE",
    "RESTART", "END STEP"]

abstract AbstractMaterial
abstract AbstractProperty
abstract AbstractStep

type Model
    mesh :: Mesh
    materials :: Dict
    properties :: Vector
    steps :: Vector
end

function Model()
    return Model(Mesh(), Dict(), Vector(), Vector())
end

function push!(model::Model, property::AbstractProperty)
    push!(model.properties, property)
end

function push!(model::Model, step::AbstractStep)
    push!(model.steps, step)
end

###

type Keyword
    name
    options
end

function Keyword()
    return Keyword(nothing, nothing)
end

function getindex(kw::Keyword, s)
    return parse(Dict(kw.options)[s])
end

type AbaqusReaderState
    section
    subsection
    material
    property
    step
    data
end

function parse(s::AbaqusReaderState)
    data = []
    for row in s.data
        col = split(row, ',')
        col = map(parse, col)
        push!(data, col)
    end
    return data
end

function AbaqusReaderState()
    return AbaqusReaderState(Keyword(), Keyword(), nothing, nothing, nothing, [])
end


function is_comment(line)
    return startswith(line, "**")
end

function is_keyword(line)
    return startswith(line, "*") && !is_comment(line)
end

function parse_keyword(line; uppercase_keyword=true)
    args = split(line, ",")
    args = map(strip, args)
    keyword_name = strip(args[1], '*')
    if uppercase_keyword
        keyword_name = uppercase(keyword_name)
    end
    keyword_options = []
    for option in args[2:end]
        pair = split(option, "=")
        if uppercase_keyword
            pair[1] = uppercase(pair[1])
        end
        if length(pair) == 1
            push!(keyword_options, pair)
        elseif length(pair) == 2
            push!(keyword_options, pair[1] => pair[2])
        else
            error("Keyword failure: $line, $option, $pair")
        end
    end
    return Keyword(keyword_name, keyword_options)
end

function is_new_section(line)
    is_keyword(line) || return false
    section = parse_keyword(line)
    section.name in ABAQUS_SECTIONS || return false
    return true
end

function is_new_subsection(line)
    is_keyword(line) || return false
    subsection = parse_keyword(line)
    subsection.name in ABAQUS_SUBSECTIONS || return false
    return true
end

function maybe_open_section!(model, state)
    section_name = Val{Symbol(state.section.name)}
    args = Tuple{Model, AbaqusReaderState, Type{section_name}}
    if method_exists(open_section!, args)
        info("Opening section $(state.section.name)")
        open_section!(model, state, section_name)
    end
end

function maybe_close_section!(model, state)
    section_name = Val{Symbol(state.section.name)}
    args = Tuple{Model, AbaqusReaderState, Type{section_name}}
    if method_exists(close_section!, args)
        info("Closing section $(state.section.name)")
        close_section!(model, state, section_name)
    end
end

function new_section!(model, state, line::AbstractString)
    maybe_close_subsection!(model, state)
    maybe_close_section!(model, state)
    state.data = []
    state.section = parse_keyword(line)
    state.subsection = Keyword()
    info("New section: $(state.section.name) with options $(state.section.options)")
    maybe_open_section!(model, state)
end

function maybe_open_subsection!(model, state)
    section_name = Val{Symbol(state.section.name)}
    subsection_name = Val{Symbol(state.subsection.name)}
    args = Tuple{Model, AbaqusReaderState, Type{section_name}, Type{subsection_name}}
    if method_exists(open_subsection!, args)
        info("Opening subsection $(state.section.name) / $(state.subsection.name)")
        open_subsection!(model, state, section_name, subsection_name)
    end
end

function maybe_close_subsection!(model, state)
    section_name = Val{Symbol(state.section.name)}
    subsection_name = Val{Symbol(state.subsection.name)}
    args = Tuple{Model, AbaqusReaderState, Type{section_name}, Type{subsection_name}}
    if method_exists(close_subsection!, args)
        info("Closing subsection $(state.section.name) / $(state.subsection.name)")
        close_subsection!(model, state, section_name, subsection_name)
    end
end

function new_subsection!(model, state, line::AbstractString)
    maybe_close_subsection!(model, state)
    state.data = []
    state.subsection = parse_keyword(line)
    info("New subsection: $(state.subsection.name) with options $(state.subsection.options)")
    maybe_open_subsection!(model, state)
end

# open_section! and open_subsection! are called right after keyword is found
function open_section! end
function open_subsection! end

# close_section! and close_subsection! are called at the end or section or before new keyword
function close_section! end
function close_subsection! end

function process_line!(model, state, line)
    if state.section.name == nothing
        info("unknown section, line = $line")
        return
    end
    if is_keyword(line)
        warn("missing keyword..? $line")
        info("($(state.section.name), $(state.subsection.name)) => $line")
        return
    end
    push!(state.data, line)
end

function abaqus_read_model(fn; read_mesh=true)

    model = Model()

    if read_mesh
        model.mesh = abaqus_read_mesh(fn)
    else
        model.mesh = Mesh()
    end

    state = AbaqusReaderState()

    fid = open(fn)
    for line in eachline(fid)
        line = strip(line)
        is_comment(line) && continue
        if is_new_section(line)
            new_section!(model, state, line)
        elseif is_new_subsection(line)
            new_subsection!(model, state, line)
        else
            process_line!(model, state, line)
        end
    end
    maybe_close_subsection!(model, state)
    maybe_close_section!(model, state)
    close(fid)

    return model
end

### Model parse start

## Properties

type SolidSection <: AbstractProperty
    element_set
    material
end

function close_section!(model, state, ::Type{Val{Symbol("SOLID SECTION")}})
    property = SolidSection(state.section["ELSET"], state.section["MATERIAL"])
    push!(model, property)
end

## Materials

abstract MaterialProperty

type Elastic <: MaterialProperty
    E
    nu
end

type Material <: AbstractMaterial
    name
    properties
end

function Material(name)
    return Material(name, [])
end

function push!(material::Material, property::MaterialProperty)
    push!(material.properties, property)
end

function open_section!(model, state, ::Type{Val{:MATERIAL}})
    state.material = Material(state.section["NAME"])
end

function close_subsection!(model, state, ::Type{Val{:MATERIAL}}, ::Type{Val{:ELASTIC}})
    E, nu = parse(state)[1]
    push!(state.material, Elastic(E, nu))
end

function close_section!(model, state, ::Type{Val{:MATERIAL}})
    material_name = state.material.name
    if haskey(model.materials, material_name)
        warn("Material $material_name already exists in model, skipping definition.")
    else
        model.materials[material_name] = state.material
    end
end

## Steps

type Step <: AbstractStep
    content :: Vector
end

function push!(step::Step, data)
    push!(step.content, data)
end

abstract AbstractBoundaryCondition

type Boundary <: AbstractBoundaryCondition
    data :: Vector
end

function getindex(b::Boundary, j::Int64)
    return b.data[j]
end

type DSLoad <: AbstractBoundaryCondition
    data :: Vector
end

function getindex(l::DSLoad, j::Int64)
    return l.data[j]
end

function open_section!(model, state, ::Type{Val{:STEP}})
    state.step = Step([])
end

function close_subsection!(model, state, ::Type{Val{:STEP}}, ::Type{Val{:BOUNDARY}})
   push!(state.step, Boundary(parse(state)))
end

function close_subsection!(model, state, ::Type{Val{:STEP}}, ::Type{Val{:DSLOAD}})
   push!(state.step, DSLoad(parse(state)))
end

function close_section!(model, state, ::Type{Val{:STEP}})
    push!(model.steps, state.step)
end

### model parse end

# when model is called, run simulation

function determine_problem_type(model, element_set_name)
    return Elasticity
end

function determine_problem_dimension(model, element_set_name)
    return 3
end

function call(model::Model)
    info("Starting JuliaFEM-ABAQUS solver.")
    # 1. create field problems and add elements
    field_problems = []
    for (element_set_name, element_ids) in model.mesh.element_sets
        problem_type = determine_problem_type(model, element_set_name)
        problem_name = "BODY $element_set_name"
        problem_dimension = determine_problem_dimension(model, element_set_name)
        problem = Problem(problem_type, problem_name, problem_dimension)
        problem.elements = create_elements(model.mesh, element_set_name)
        section = get_element_section(element_set_name)
        material = get_material(section.material)
        update!(problem, material)
        push!(field_problems, problem)
    end
    # 2. loop steps
    for step in model.steps
        boundary_problems = []
        for bc in step.content
            if isa(bc, Boundary)
                problem = Problem(Dirichlet, "fix nodes", 3, "displacement")
                for (bc_name, dof) in bc.data
                    nodes = model.mesh.node_sets[bc_name]
                    for node in nodes
                        fix_node!(problem, node => dof)
                    end
                end
                push!(boundary_problems, problem)
            end
            if isa(bc, DSLoad)
                problem = Problem(Elasticity, "pressure load", 3, "displacement")
                for (bc_name, bc_type, pressure) in bc.data
                    elements = create_elements_from_surface_set(bc_name)
                    update!(elements, "surface pressure", pressure)
                    problem.elements = [problem.elements; elements]
                end
                push!(boundary_problems, problem)
            end
        end
        all_problems = [problems; boundary_problems]
        solver = Solver(solver_type, solver_description, all_problems...)
        solver()
    end
end

