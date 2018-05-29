# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Documenter, JuliaFEM

if !haskey(Pkg.installed(), "Literate")
    Pkg.add("Literate")
end
using Literate

"""
    copy_docs(pkg_name)

Copy documentation of some package `pkg_name` to `docs/src/pkg_name`,
where `pkg_name` is the name of package. Return true if success, false
otherwise.
"""
function copy_docs(pkg_name)
    src = joinpath(Pkg.dir(pkg_name), "docs", "src")
    if !isdir(src)
        warn("Cannot copy documentation of package $pkg_name from $src: ",
             "No such directory exists. (Is the package in REQUIRE of JuliaFEM?)")
        return false
    end
    dst = joinpath(Pkg.dir("JuliaFEM"), "docs", "src", pkg_name)
    cp(src, dst; remove_destination=true)
    info("Copied documentation of package $pkg_name from $src succesfully.")
    return true
end

"""
    add_page!(part, page)

Add new page to documentation. Part can be USER_GUIDE, DEVELOPER_GUIDE or
PACKAGES. `page` is a string containing path to the page, or `Pair`
where key is the name of the page and value is the path to the page.

# Examples

If page starts with `#`, i.e. having title, page is automatically included
to documentation with that name:

```julia
add_page!(PACKAGES, "MyPackage/index.md")
```

Title can be changed using `Pair`, i.e.

```julia
add_page!(PACKAGES, "Theory" => "MyPackage/theory.md")
```

"""
function add_page!(dst, src)
    file = isa(src, Pair) ? src.second : src
    if !isfile(joinpath(Pkg.dir("JuliaFEM"), "docs", "src", file))
        warn("Cannot add page $file: no such file")
        return false
    end
    push!(dst, src)
    return true
end

#=
if haskey(ENV, "TRAVIS")
    println("inside TRAVIS, installing PyPlot + matplotlib")
    Pkg.add("PyPlot")
    run(`pip install matplotlib`)
end
=#

USER_GUIDE = []

# Developer's guide is published in FEMBase.jl
DEVELOPER_GUIDE = []
if copy_docs("FEMBase")
    add_page!(DEVELOPER_GUIDE, "FEMBase/mesh.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/fields.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/basis.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/integration.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/elements.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/problems.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/solvers.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/postprocessing.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/results.md")
    add_page!(DEVELOPER_GUIDE, "FEMBase/materials.md")
end

# Let's construct here some description for packages
PACKAGES = []

if copy_docs("FEMQuad")
    add_page!(PACKAGES, "FEMQuad/index.md")
end

if copy_docs("FEMBasis")
    add_page!(PACKAGES, "FEMBasis/index.md")
end

if copy_docs("HeatTransfer")
    add_page!(PACKAGES, "HeatTransfer/index.md")
end

# Generate examples using Literate.jl

function license_stripper(s)
    return replace(s, r"# This file is a part of JuliaFEM.*?\n\n"sm, "")
end
EXAMPLES = []
docs_src = Pkg.dir("JuliaFEM", "docs", "src")
ex_src = Pkg.dir("JuliaFEM", "examples")
ex_dst = joinpath(docs_src, "examples")
for ex_file in readdir(ex_src)
    dst_file = Literate.markdown(joinpath(ex_src, ex_file), ex_dst;
                                 documenter=true, preprocess=license_stripper)
    push!(EXAMPLES, relpath(dst_file, docs_src))
end

# API documentation
LIBRARY = ["api.md"]

# Collect all together
PAGES = []
push!(PAGES, "Home" => "index.md")
#push!(PAGES, "User's guide" => USER_GUIDE)
push!(PAGES, "Developer's guide" => DEVELOPER_GUIDE)
push!(PAGES, "Description of packages" => PACKAGES)
push!(PAGES, "Examples" => EXAMPLES)
#push!("API documentation" => LIBRARY)

println(PAGES)

makedocs(modules=[JuliaFEM],
         format = :html,
         checkdocs = :all,
         sitename = "JuliaFEM",
         analytics = "UA-83590644-1",
         pages = PAGES)
