# Modern Documenter.jl build script for JuliaFEM API documentation
# Generates markdown files for Quarto integration

using Documenter
using DocumenterMarkdown
using JuliaFEM

# Get directories
juliafem_dir = dirname(@__DIR__)
website_dir = joinpath(juliafem_dir, "juliafem.github.io")
build_dir = joinpath(website_dir, "api")  # Build markdown directly to api/ directory

# Simple pages structure - just API documentation
pages = [
    "Home" => "index.md",
    "API Reference" => "api.md",
]

# Build API documentation as markdown
makedocs(
    sitename = "JuliaFEM API",
    authors = "JuliaFEM Team",
    modules = [JuliaFEM],
    
    # Source and build directories
    source = "api",  # Use docs/api/ for source
    build = build_dir,  # Build markdown to website api/ directory
    
    # Format: Generate markdown instead of HTML
    format = DocumenterMarkdown.Markdown(),
    
    # Documentation options
    checkdocs = :none,
    doctest = false,
    linkcheck = false,
    warnonly = [:missing_docs, :cross_references],
)
