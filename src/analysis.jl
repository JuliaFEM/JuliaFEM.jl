# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

abstract type AbstractAnalysis end
abstract type AbstractResultsWriter end

mutable struct Analysis{A<:AbstractAnalysis}
    name :: String
    problems :: Vector{Problem}
    fields :: Dict{String, AbstractField}
    results_writers :: Vector{AbstractResultsWriter}
    properties :: A
end

"""
    Analysis(A, analysis_name)

Create a new analysis of type `A`, where `A` is a subtype of `AbstractAnalysis`.
Analysis can be e.g. `Linear` for linear quasistatic analysis, `Nonlinear` for
nonlinear quasistatic analysis or `Modal` for natural frequency analysis.

# Examples

```julia
analysis = Analysis(Linear, "linear quasistatic analysis of beam structure")
```
"""
function Analysis(::Type{A}, name::String="$A Analysis") where A<:AbstractAnalysis
    analysis = Analysis{A}(name, [], Dict(), [], A())
    @info("Creating a new analysis of type $A with name `$name`.")
    return analysis
end

function add_problem!(analysis::Analysis, problems::Problem...)
    for problem in problems
        @info("Adding problem `$(problem.name)` to analysis `$(analysis.name)`.")
        push!(analysis.problems, problem)
    end
    return nothing
end

add_problems!(analysis::Analysis, problems::Union{Vector, Tuple}) = add_problem!(analysis, problems...)

"""
    add_problems!(analysis, problem...)

Add problem(s) to analysis.

# Examples

Add two problems, `beam` and `bc`, to linear quasistatic analysis:
```julia
beam = Problem(Beam, "beam structure", 6)
bc = Problem(Dirichlet, "fix", 6, "displacement")
analysis = Analysis(Linear, "linear quasistatic analysis")
add_problems!(analysis, beam, bc)
```
"""
add_problems!(analysis::Analysis, problems...) = add_problem!(analysis, problems...)

function get_problems(analysis::Analysis)
    return analysis.problems
end

function get_problem(analysis::Analysis, problem_name)
    problems = filter(p -> p.name == problem_name, get_problems(analysis))
    if length(problems) != 1
        error("Several problem with name $problem_name found from analysis $(analysis.name).")
    end
    return first(problems)
end

function add_results_writer!(analysis::Analysis, writer::W) where W<:AbstractResultsWriter
    push!(analysis.results_writers, writer)
    return nothing
end

function get_results_writers(analysis::Analysis)
    return analysis.results_writers
end

function run!(::Analysis{A}) where A<:AbstractAnalysis
    @info("This is a placeholder function for running an analysis $A for a set of problems.")
end

function write_results!(::Analysis{A}, ::W) where {A<:AbstractAnalysis, W<:AbstractResultsWriter}
    @info("Writing the results of analysis $A is not supported by a results writer $W")
    return nothing
end

function write_results!(analysis)
    results_writers = get_results_writers(analysis)
    if isempty(results_writers)
        @info("No result writers attached to the analysis $(analysis.name). " *
              "In order to get results of the analysis stored to the disk, one " *
              "must attach some results writer to the analysis using " *
              "add_results_writer!, e.g. xdmf_writer = Xdmf(\"results\"); " *
              "add_results_writer!(analysis, xdmf_writer)")
        return nothing
    end
    for results_writer in results_writers
        write_results!(analysis, results_writer)
    end
    return nothing
end
