# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

importall Base

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess

### Model definitions for ABAQUS data model

abstract AbstractMaterial
abstract AbstractMaterialProperty
abstract AbstractProperty
abstract AbstractStep
abstract AbstractBoundaryCondition
abstract AbstractOutputRequest

type Model
    path :: AbstractString
    name :: AbstractString
    mesh :: Mesh
    materials :: Dict{Symbol, AbstractMaterial}
    properties :: Vector{AbstractProperty}
    boundary_conditions :: Vector{AbstractBoundaryCondition}
    steps :: Vector{AbstractStep}
    problems :: Vector{Problem}
end

type SolidSection <: AbstractProperty
    element_set :: Symbol
    material_name :: Symbol
end

type Material <: AbstractMaterial
    name :: Symbol
    properties :: Vector{AbstractMaterialProperty}
end

type Elastic <: AbstractMaterialProperty
    E :: Float64
    nu :: Float64
end

type Step <: AbstractStep
    kind :: Nullable{Symbol} # STATIC, ...
    boundary_conditions :: Vector{AbstractBoundaryCondition}
    output_requests :: Vector{AbstractOutputRequest}
end

type BoundaryCondition <: AbstractBoundaryCondition
    kind :: Symbol # BOUNDARY, CLOAD, DLOAD, DSLOAD, ...
    data :: Vector
    options :: Dict
end

type OutputRequest <: AbstractOutputRequest
    kind :: Symbol # NODE, EL, SECTION, ...
    data :: Vector
    options :: Dict
    target :: Symbol # PRINT, FILE
end

### Utility functions to parse ABAQUS .inp file to data model

type Keyword
    name :: AbstractString
    options :: Vector{Union{AbstractString, Pair}}
end

function getindex(kw::Keyword, s)
    return parse(Dict(kw.options)[s])
end

type AbaqusReaderState
    section :: Nullable{Keyword}
    material :: Nullable{AbstractMaterial}
    property :: Nullable{AbstractProperty}
    step :: Nullable{AbstractStep}
    data :: Vector{AbstractString}
end

function get_data(state::AbaqusReaderState)
    data = []
    for row in state.data
        row = strip(row, [' ', ','])
        col = split(row, ',')
        col = map(parse, col)
        push!(data, col)
    end
    return data
end

function get_options(state::AbaqusReaderState)
    return Dict(get(state.section).options)
end

function get_option(state::AbaqusReaderState, what::AbstractString)
    return get_options(state)[what]
end

function length(state::AbaqusReaderState)
    return length(state.data)
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
    keyword = Keyword(keyword_name, [])
    for option in args[2:end]
        pair = split(option, "=")
        if uppercase_keyword
            pair[1] = uppercase(pair[1])
        end
        if length(pair) == 1
            push!(keyword.options, pair[1])
        elseif length(pair) == 2
            push!(keyword.options, pair[1] => pair[2])
        else
            error("Keyword failure: $line, $option, $pair")
        end
    end
    return keyword
end

macro register_abaqus_keyword(keyword)
    underscored = Symbol(replace(keyword, " ", "_"))
    quote
        global is_abaqus_keyword_registered
        typealias $underscored Type{Val{Symbol($keyword)}}
        is_abaqus_keyword_registered(::Type{Val{Symbol($keyword)}}) = true
    end
end

function is_abaqus_keyword_registered(s::AbstractString)
    return is_abaqus_keyword_registered(Val{Symbol(s)})
end

function is_abaqus_keyword_registered(others)
    return false
end

function is_new_section(line)
    is_keyword(line) || return false
    section = parse_keyword(line)
    is_abaqus_keyword_registered(section.name) || return false
    return true
end

function maybe_close_section!(model, state; verbose=true)
    isnull(state.section) && return
    section_name = get(state.section).name
    verbose && info("Close section: $section_name")
    args = Tuple{Model, AbaqusReaderState, Type{Val{Symbol(section_name)}}}
    if method_exists(close_section!, args)
        close_section!(model, state, Val{Symbol(section_name)})
    else
        verbose && warn("no close_section! found for $section_name")
    end
    state.section = nothing
end

function maybe_open_section!(model, state; verbose=true)
    section_name = get(state.section).name
    section_options = get(state.section).options
    verbose && info("New section: $section_name with options $section_options")
    args = Tuple{Model, AbaqusReaderState, Type{Val{Symbol(section_name)}}}
    if method_exists(open_section!, args)
        open_section!(model, state, Val{Symbol(section_name)})
    else
        verbose && warn("no open_section! found for $section_name")
    end
end

function new_section!(model, state, line::AbstractString; verbose=true)
    maybe_close_section!(model, state; verbose=verbose)
    state.data = []
    state.section = parse_keyword(line)
    maybe_open_section!(model, state; verbose=verbose)
end

# open_section! is called right after keyword is found
function open_section! end

# close_section! is called at the end or section or before new keyword
function close_section! end

function process_line!(model, state, line; verbose=false)
    if isnull(state.section)
        verbose && info("section = nothing! line = $line")
        return
    end
    if is_keyword(line)
        warn("missing keyword? line = $line")
        # close section, this is probably keyword and collecting data should stop.
        maybe_close_section!(model, state)
        return
    end
    push!(state.data, line)
end

function abaqus_read_model(fn; read_mesh=true)

    model_path = dirname(fn)
    model_name = first(splitext(basename(fn)))
    model = Model(model_path, model_name, Mesh(), Dict(), [], [], [], [])

    if read_mesh
        model.mesh = abaqus_read_mesh(fn)
    end

    state = AbaqusReaderState(nothing, nothing, nothing, nothing, [])

    fid = open(fn)
    for line in eachline(fid)
        line = strip(line)
        is_comment(line) && continue
        if is_new_section(line)
            new_section!(model, state, line)
        else
            process_line!(model, state, line)
        end
    end
    close(fid)
    maybe_close_section!(model, state)

    return model
end

### Code to parse ABAQUS .inp to data model

# add here only keywords when planning to define open_section! and/or
# close_section!, i.e. actually parse keyword to model. little bit of
# magic is happening here, but after calling macro there is typealias
# defined i.e. typealias SOLID_SECTION Type{Val{Symbol("SOLID_SECTION")}}
# and also is_keyword_registered("SOLID SECTION") returns true after
# registration, also notice underscoring

@register_abaqus_keyword("SOLID SECTION")

@register_abaqus_keyword("MATERIAL")
@register_abaqus_keyword("ELASTIC")

@register_abaqus_keyword("STEP")
@register_abaqus_keyword("STATIC")
@register_abaqus_keyword("END STEP")

@register_abaqus_keyword("BOUNDARY")
@register_abaqus_keyword("CLOAD")
@register_abaqus_keyword("DLOAD")
@register_abaqus_keyword("DSLOAD")
typealias BOUNDARY_CONDITIONS Union{BOUNDARY, CLOAD, DLOAD, DSLOAD}

@register_abaqus_keyword("NODE PRINT")
@register_abaqus_keyword("EL PRINT")
@register_abaqus_keyword("SECTION PRINT")
typealias OUTPUT_REQUESTS Union{NODE_PRINT, EL_PRINT, SECTION_PRINT}

## Properties

function open_section!(model, state, ::SOLID_SECTION)
    element_set = get_option(state, "ELSET")
    material_name = get_option(state, "MATERIAL")
    property = SolidSection(element_set, material_name)
    state.property = property
    push!(model.properties, property)
end

function close_section!(model, state, ::SOLID_SECTION)
    state.property = nothing
end

## Materials

function open_section!(model, state, ::MATERIAL)
    material_name = Symbol(get_option(state, "NAME"))
    material = Material(material_name, [])
    state.material = material
    if haskey(model.materials, material_name)
        warn("Material $material_name already exists in model, skipping definition.")
    else
        model.materials[material_name] = material
    end
end

function close_section!(model, state, ::ELASTIC)
    # FIXME
    @assert length(state) == 1
    E, nu = first(get_data(state))
    material_property = Elastic(E, nu)
    material = get(state.material)
    push!(material.properties, material_property)
end

## Steps

function open_section!(model, state, ::STEP)
    step = Step(nothing, Vector(), Vector())
    state.step = step
    push!(model.steps, step)
end

function open_section!(model, state, ::STATIC)
    isnull(state.step) && error("*STATIC outside *STEP ?")
    get(state.step).kind = :STATIC
end

function open_section!(model, state, ::END_STEP)
    state.step = nothing
end

## Steps -- boundary conditions

function close_section!(model, state, ::BOUNDARY_CONDITIONS)
    kind = Symbol(get(state.section).name)
    data = get_data(state)
    options = get_options(state)
    bc = BoundaryCondition(kind, data, options)
    if isnull(state.step)
        push!(model.boundary_conditions, bc)
    else
        step = get(state.step)
        push!(step.boundary_conditions, bc)
    end
end

## Steps -- output requests

function close_section!(model, state, ::OUTPUT_REQUESTS)
    kind, target = map(parse, split(get(state.section).name, " "))
    data = get_data(state)
    options = get_options(state)
    request = OutputRequest(kind, data, options, target)
    step = get(state.step)
    push!(step.output_requests, request)
end


### Code to use JuliaFEM to run ABAQUS data model

function determine_problem_type(model::Model)
    # FIXME
    return Elasticity
end

function determine_problem_dimension(model::Model)
    # FIXME
    return 3
end

function get_element_section(model::Model, element_set_name::Symbol)
    sections = filter(s -> s.element_set == element_set_name, model.properties)
    length(sections) == 1 || error("Multiple sections found for element set $element_set_name")
    return sections[1]
end

function get_material(model::Model, material_name)
    return model.materials[material_name]
end

function create_problem(model::Model, element_set_name::Symbol; verbose=true)
    problem_type = determine_problem_type(model)
    problem_name = "$problem_type $element_set_name"
    problem_dimension = determine_problem_dimension(model)
    problem = Problem(problem_type, problem_name, problem_dimension)
    problem.elements = create_elements(model.mesh, element_set_name)
    section = get_element_section(model, element_set_name)
    material = get_material(model, section.material_name)
    for mp in material.properties
        if isa(mp, Elastic)
            verbose && info("$element_set_name: elastic material, E = $(mp.E), nu = $(mp.nu)")
            update!(problem.elements, "youngs modulus", mp.E)
            update!(problem.elements, "poissons ratio", mp.nu)
        end
    end
    return problem
end

""" Dirichlet boundary condition. """
function create_boundary_problem(model::Model, bc::AbstractBoundaryCondition, ::BOUNDARY; verbose=true)
    dim = determine_problem_dimension(model)
    problem = Problem(Dirichlet, "Dirichlet boundary *BOUNDARY", dim, "displacement")
    for row in bc.data

        if isa(row[1], AbstractString) # node set given
            nodes = model.mesh.node_sets[bc_name]
        else # single node given
            nodes = [row[1]]
        end

        elements = [Element(Poi1, [id]) for id in nodes]
        update!(elements, "geometry", model.mesh.nodes)

        for dof in row[2]:row[end]
            # FIXME
            val = 0.0
            update!(elements, "displacement $dof", val)
            verbose && info("Nodes ", join(nodes, ", "), " dof $dof => $val")
        end

        push!(problem, elements)
    end
    return problem
end

""" Distributed surface load (DSLOAD). """
function create_boundary_problem(model::Model, bc::AbstractBoundaryCondition, ::DSLOAD; verbose=false)
    dim = determine_problem_dimension(model)
    problem = Problem(Elasticity, "Distributed surface load *DSLOAD", dim)
    for row in bc.data
        bc_name, bc_type, pressure = row
        bc_type == :P || error("bc_type = $bc_type != :P")
        elements = []
        for (parent_element_id, parent_element_side) in model.mesh.surface_sets[bc_name]
            parent_element_type = model.mesh.element_types[parent_element_id]
            parent_element_connectivity = model.mesh.elements[parent_element_id]

            child_element_type, child_element_lconn, child_element_connectivity = 
                get_child_element(parent_element_type, parent_element_side,
                parent_element_connectivity)

            verbose && info("parent element : $parent_element_id, $parent_element_type, $parent_element_connectivity, $parent_element_side")
            verbose && info("child element  : $child_element_type, $child_element_connectivity")

            child_element = Element(JuliaFEM.(child_element_type), child_element_connectivity)
            push!(elements, child_element)
        end
        update!(elements, "geometry", model.mesh.nodes)
        update!(elements, "surface pressure", pressure)
        push!(problem, elements)
    end
    return problem
end

""" Distributed load (DLOAD). """
function create_boundary_problem(model::Model, bc::AbstractBoundaryCondition, ::DLOAD; verbose=false)
    dim = determine_problem_dimension(model)
    problem = Problem(Elasticity, "Distributed load *DLOAD", dim)
    for row in bc.data
        parent_element_id, parent_element_side, pressure = row
        parent_element_type = model.mesh.element_types[parent_element_id]
        parent_element_connectivity = model.mesh.elements[parent_element_id]

        child_element_type, child_element_lconn, child_element_connectivity = 
            get_child_element(parent_element_type, parent_element_side,
            parent_element_connectivity)

        verbose && info("parent element : $parent_element_id, $parent_element_type, $parent_element_connectivity, $parent_element_side")
        verbose && info("child element  : $child_element_type, $child_element_connectivity")

        child_element = Element(getfield(JuliaFEM, child_element_type), child_element_connectivity)
        update!(child_element, "geometry", model.mesh.nodes)
        update!(child_element, "surface pressure", -pressure)
        push!(problem.elements, child_element)
    end
    return problem
end

""" Concentrated load (CLOAD). """
function create_boundary_problem(model::Model, bc::AbstractBoundaryCondition, ::CLOAD; verbose=false)
    dim = determine_problem_dimension(model)
    problem = Problem(Elasticity, "Concentrated load *CLOAD", dim)
    nodes = sort(unique([row[1] for row in bc.data]))
    elements = Dict()
    for node in nodes
        element = Element(Poi1, [node])
        update!(element, "geometry", model.mesh.nodes)
        elements[node] = element
    end
    for row in bc.data
        node, dof, load = row
        update!(elements[node], "concentrated force $dof", load)
    end
    problem.elements = collect(values(elements))
    return problem
end

function create_boundary_problem(model::Model, bc::AbstractBoundaryCondition)
    create_boundary_problem(model, bc, Val{bc.kind})
end

""" Given element code, element side and global connectivity, determine boundary
element. E.g. for Tet4 we have 4 sides S1..S4 and boundary element is of type Tri3.
"""
function get_child_element(element_type::Symbol, element_side::Symbol,
                           element_connectivity::Vector{Int64})

    element_mapping = Dict(
        :Tet4 => Dict(
            :S1 => (:Tri3, [1, 3, 2]),
            :S2 => (:Tri3, [1, 2, 4]),
            :S3 => (:Tri3, [2, 3, 4]),
            :S4 => (:Tri3, [1, 4, 3])),
        :Tet10 => Dict(
            :S1 => (:Tri6, [1, 3, 2, 7, 6, 5]),
            :S2 => (:Tri6, [1, 2, 4, 5, 9, 8]),
            :S3 => (:Tri6, [2, 3, 4, 6, 10, 9]),
            :S4 => (:Tri6, [1, 4, 3, 8, 10, 7])),
        :Hex8 => Dict(
            :P1 => (:Quad4, [1, 2, 3, 4]),
            :P2 => (:Quad4, [5, 8, 7, 6]),
            :P3 => (:Quad4, [1, 5, 6, 2]),
            :P4 => (:Quad4, [2, 6, 7, 3]),
            :P5 => (:Quad4, [3, 7, 8, 4]),
            :P6 => (:Quad4, [4, 8, 5, 1]))
        )

        if !haskey(element_mapping, element_type)
            error("Unable to find child element for element of type $element_type for side $element_side, check mapping.")
        end

        if !haskey(element_mapping[element_type], element_side)
            error("Unable to find child element side mapping for element of type $element_type for side $element_side, check mapping.")
        end

        child_element, child_element_lconn = element_mapping[element_type][element_side]
        child_element_gconn = element_connectivity[child_element_lconn]
        return child_element, child_element_lconn, child_element_gconn
    end

    function determine_solver_type(model::Model, step::AbstractStep)
        # FIXME
        return Linear
    end

    function process_output_request(model::Model, solver::Solver, output_request::AbstractOutputRequest)
        kind = Val{output_request.kind}
        target = Val{output_request.target}
    process_output_request(model, solver, output_request, kind, target)
end

function process_output_request(model::Model, solver::Solver, output_request::AbstractOutputRequest,
                                ::Type{Val{:NODE}}, ::Type{Val{:PRINT}})
    data = output_request.data
    options = output_request.options
    code_mapping = Dict(
        :COORD => "geometry",
        :U => "displacement",
        :CF => "concentrated force",
        :RF => "reaction force")
    abbr_mapping = Dict(:COORD => :COOR)
    for row in data
        info(repeat("-", 80))
        codes = join(row, ", ")
        info("*NODE PRINT request, with fields $codes")
        if length(options) != 0
            info("Additional options: $options")
        end
        info(repeat("-", 80))
        tables = Any[]
        for code in row
            haskey(code_mapping, code) || continue
            field_name = code_mapping[code]
            abbr = get(abbr_mapping, code, code)
            table = solver(DataFrame, field_name, abbr, solver.time)
            push!(tables, table)
        end
        length(tables) != 0 || continue
        results = join(tables..., on=:NODE, kind=:outer)
        sort!(results, cols=[:NODE])
        println()
        println(results)
        println()
    end
end

function process_output_request(model::Model, solver::Solver, output_request::AbstractOutputRequest,
                                ::Type{Val{:EL}}, ::Type{Val{:PRINT}})
    data = output_request.data
    options = output_request.options
    code_mapping = Dict(
        :COORD => "geometry",
        :S => "stress",
        :E => "strain")
    abbr_mapping = Dict(:COORD => :COOR)
    for row in data
        info(repeat("-", 80))
        codes = join(row, ", ")
        info("*EL PRINT request, with fields $codes")
        if length(options) != 0
            info("Additional options: $options")
        end
        info(repeat("-", 80))
        #= to be fixed
        tables = Any[]
        for code in row
            haskey(code_mapping, code) || continue
            field_name = code_mapping[code]
            abbr = get(abbr_mapping, code, code)
            table = solver(DataFrame, solver.time, Val{code})
            push!(tables, table)
        end
        length(tables) != 0 || continue
        results = first(tables)
        if length(tables) > 1
            for i=2:length(tables)
                results = join(results, tables[i], on=:ELEMENT, kind=:outer)
            end
        end
        #sort!(results; cols=[:ELEMENT, :IP])
        # filter out elements with id -1, they are automatically created boundary elements
        fel = find(results[:ELEMENT] .!= Symbol("E-1"))
        results = results[fel, :]
        println()
        println(results)
        println()
        =#
    end
end

function process_output_request(model::Model, solver::Solver, output_request::AbstractOutputRequest,
                                ::Type{Val{:SECTION}}, ::Type{Val{:PRINT}})
    data = output_request.data
    options = output_request.options
    info("SECTION PRINT output request, with data $data and options $options")
end

function (model::Model)()
    info("Starting JuliaFEM-ABAQUS solver.")

    # 1. create field problems and add elements
    element_sets = collect(keys(model.mesh.element_sets))
    info("Creating problems for element sets ", join(element_sets, ", "))
    model.problems = [create_problem(model, elset) for elset in element_sets]

    # 2. create boundary problems (the ones defined before *STEP)
    info("Boundary conditions defined before *STEP")
    for (i, bc) in enumerate(model.boundary_conditions)
        info("$i $(bc.kind)")
    end
    boundary_problems = [create_boundary_problem(model, bc) for bc in model.boundary_conditions]

    # 3. loop steps
    for step in model.steps
        # 3.1 add boundary conditions defined inside *STEP
        step_problems = [create_boundary_problem(model, bc) for bc in step.boundary_conditions]
        # 3.2 create solver and solve set of problems
        solver_type = determine_solver_type(model, step)
        solver_description = "$solver_type solver"
        solver = Solver(solver_type, solver_description)
        all_problems = [model.problems; boundary_problems; step_problems]
        push!(solver, all_problems...)
        solver()
        info(repeat("-", 80))
        info("Simulation ready, processing output requests")
        info(repeat("-", 80))
        # 3.3 postprocessing based on output requests
        for output_request in step.output_requests
            process_output_request(model, solver, output_request)
        end
    end

    return 0
end

function abaqus_download(name)
    fn = "$name.inp"
    if !haskey(ENV, "ABAQUS_DOWNLOAD_URL")
        info("""
        ABAQUS input file $fn not found and ABAQUS_DOWNLOAD_URL not set, unable to
        download file. To enable automatic model downloading, set environment variable
        ABAQUS_URL to point url to models.""")
        return 1
    end
    url = ENV["ABAQUS_DOWNLOAD_URL"]
    if haskey(ENV, "ABAQUS_DOWNLOAD_DIR")
        fn = rstrip(ENV["ABAQUS_DOWNLOAD_DIR"], '/') * "/" * fn
    end
    if !isfile(fn)
        info("Downloading model $name ...")
        download("$url/$name.inp", fn)
    end
    return 0
end

""" Return input file name. """
function abaqus_input_file_name(name)
    fn = "$name.inp"
    isfile(fn) && return fn
    if haskey(ENV, "ABAQUS_DOWNLOAD_DIR")
        fn = rstrip(ENV["ABAQUS_DOWNLOAD_DIR"], '/') * "/" * fn
    end
    isfile(fn) && return fn
    return ""
end

function abaqus_input_file_path(name)
    return dirname(abaqus_input_file_name(name))
end

function abaqus_open_results(name)
    path = abaqus_input_file_path(name)
    result_file = "$path/$name.xmf"
    return XDMF(result_file)
end

### JuliaFEM-ABAQUS interface entry point

"""
Run ABAQUS model. If input file is not found, attempt to fetch it from internet
if fetch is set to true and ABAQUS_DOWNLOAD_URL is set. Return exit code 0 if
execution of model is success.
"""
function abaqus_run_model(name; fetch=false, verbose=false)

    if !isfile("$name.inp") && fetch
        status = abaqus_download(name)
        status == 0 || return status # download failed
    end

    fn = abaqus_input_file_name(name)

    if verbose
        println(repeat("-", 80))
        println("Running ABAQUS model $name from file $fn")
        println(repeat("-", 80))
        println(readstring(fn))
        println(repeat("-", 80))
    end

    model = abaqus_read_model(fn)
    status = model()
    return status
end


""" 
This function gerates surface elements from solid elements

slave = create_surface_elements(mesh, :slave_surf) 
master = create_surface_elements(mesh, :master_surf) 
"""
function create_surface_elements(mesh::Mesh, surface_name::Symbol)
   elements = []
   for (parent_element_id, parent_element_side) in mesh.surface_sets[surface_name]
       parent_element_type = mesh.element_types[parent_element_id]
       parent_element_connectivity = mesh.elements[parent_element_id]

       child_element_type, child_element_lconn, child_element_connectivity =
           get_child_element(parent_element_type, parent_element_side,
           parent_element_connectivity)

       child_element = Element(getfield(JuliaFEM, child_element_type), child_element_connectivity)
       push!(elements, child_element)
   end
   update!(elements, "geometry", mesh.nodes)
   return elements
end

function create_surface_elements(mesh::Mesh, surface_name::String)
    return create_surface_elements(mesh, Symbol(surface_name))
end

