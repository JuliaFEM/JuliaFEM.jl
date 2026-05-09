using Documenter
using JuliaFEM

makedocs(
    sitename = "JuliaFEM.jl",
    authors  = "JuliaFEM Team",
    modules  = [JuliaFEM],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = nothing,
        size_threshold_warn = nothing,
    ),
    pages    = [
        "Home"                         => "index.md",
        "Elements and multiphysics"    => "elements_multiphysics_teaser.md",
        "Thermo-elastic walkthrough"   => "thermo_elastic_walkthrough.md",
        "Choosing an assembler"        => "assembler_choice.md",
        "Legacy module"                => "legacy.md",
        "Developer"                    => [
            "Architecture layers" => "developer/architecture_layers.md",
            "Repository layout"   => "repository_layout.md",
        ],
        "API Reference"                => "api.md",
    ],
    checkdocs = :none,
    doctest   = false,
    linkcheck = false,
    warnonly  = [:missing_docs, :cross_references],
)
