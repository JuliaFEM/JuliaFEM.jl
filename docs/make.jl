# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Documenter, JuliaFEM

if !haskey(Pkg.installed(), "Literate")
    Pkg.add("Literate")
end
using Literate

if haskey(ENV, "TRAVIS") && get(ENV, "INSTALL_PYPLOT", "false") == "true"
    info("inside TRAVIS, installing PyPlot + matplotlib")
    Pkg.add("PyPlot")
    run(`pip install matplotlib`)
end

"""
    copy_docs(pkg_name)

Copy documentation of some package `pkg_name` to `docs/src/pkg_name`,
where `pkg_name` is the name of package. Return true if success, false
otherwise.

If package is undocumented, i.e. directory `docs/src` is missing,
but there still exists `README.md`, copy that file to
`docs/src/pkg_name/index.md`.
"""
function copy_docs(pkg_name)

    src_dir = Pkg.dir(pkg_name, "docs", "src")
    pkg_dir = Pkg.dir("JuliaFEM", "docs", "src", "packages")
    dst_dir = joinpath(pkg_dir, pkg_name)
    isdir(pkg_dir) || mkpath(pkg_dir)

    # if can find pkg_name/docs/src =>
    # copy that to docs/src/packages/pkg_name
    if isdir(src_dir)
        cp(src_dir, dst_dir; remove_destination=true)
        info("Copied documentation of package $pkg_name from $src_dir succesfully.")
        return true
    end

    # if can find pkg_name/README.md =>
    # copy that to docs/src/packages/pkg_name/index.md
    readme_file = Pkg.dir(pkg_name, "README.md")
    if isfile(readme_file)
        isdir(dst_dir) || mkpath(dst_dir)
        cp(readme_file, joinpath(dst_dir, "README.md"))
        info("Copied README.md of package $pkg_name from $readme_file succesfully.")
        return true
    end

    warn("Cannot copy documentation of package $pkg_name from $src_dir: ",
         "No such directory exists. (Is the package in REQUIRE of JuliaFEM?)")

    return false

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
    src_dir = Pkg.dir("JuliaFEM", "docs", "src")
    if isfile(joinpath(src_dir, file))
        push!(dst, src)
        return true
    end
    if isfile(joinpath(src_dir, "packages", file))
        push!(dst, joinpath("packages", src))
        return true
    end
    warn("Cannot add page $file: no such file")
    return false
end

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
copy_docs("FEMBase")                && add_page!(PACKAGES, "FEMBase/index.md")
copy_docs("FEMBasis")               && add_page!(PACKAGES, "FEMBasis/index.md")
copy_docs("FEMQuad")                && add_page!(PACKAGES, "FEMQuad/index.md")
copy_docs("FEMSparse")              && add_page!(PACKAGES, "FEMSparse/index.md")
copy_docs("Materials")              && add_page!(PACKAGES, "Materials/index.md")
copy_docs("AsterReader")            && add_page!(PACKAGES, "AsterReader/index.md")
copy_docs("AbaqusReader")           && add_page!(PACKAGES, "AbaqusReader/index.md")
copy_docs("LinearImplicitDynamics") && add_page!(PACKAGES, "LinearImplicitDynamics/index.md")
copy_docs("HeatTransfer")           && add_page!(PACKAGES, "HeatTransfer/index.md")
copy_docs("PlaneElasticity")        && add_page!(PACKAGES, "PlaneElasticity/index.md")
copy_docs("FEMBeam")                && add_page!(PACKAGES, "FEMBeam/index.md")
copy_docs("FEMCoupling")            && add_page!(PACKAGES, "FEMCoupling/index.md")
copy_docs("FEMTruss")               && add_page!(PACKAGES, "FEMTruss/index.md")
copy_docs("Mortar2D")               && add_page!(PACKAGES, "Mortar2D/index.md")
copy_docs("Mortar3D")               && add_page!(PACKAGES, "Mortar3D/index.md")
copy_docs("MortarContact2D")        && add_page!(PACKAGES, "MortarContact2D/index.md")
copy_docs("MortarContact2DAD")      && add_page!(PACKAGES, "MortarContact2DAD/index.md")
copy_docs("OptoMechanics")          && add_page!(PACKAGES, "OptoMechanics/index.md")
copy_docs("Miniball")               && add_page!(PACKAGES, "Miniball/index.md")
copy_docs("ModelReduction")         && add_page!(PACKAGES, "ModelReduction/index.md")
copy_docs("NodeNumbering")          && add_page!(PACKAGES, "NodeNumbering/index.md")
#copy_docs("Xdmf")                   && add_page!(PACKAGES, "Xdmf/index.md")
copy_docs("UMAT")                   && add_page!(PACKAGES, "UMAT/index.md")

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
    add_page!(EXAMPLES, relpath(dst_file, docs_src))
end

# API documentation
LIBRARY = ["api.md"]

# Collect all together
PAGES = []
push!(PAGES, "Home" => "index.md")
#push!(PAGES, "User's guide" => USER_GUIDE)
push!(PAGES, "Examples" => EXAMPLES)
push!(PAGES, "Developer's guide" => DEVELOPER_GUIDE)
push!(PAGES, "Description of packages" => PACKAGES)
#push!("API documentation" => LIBRARY)

makedocs(modules=[JuliaFEM],
         format = :html,
         checkdocs = :all,
         sitename = "JuliaFEM",
         analytics = "UA-83590644-1",
         pages = PAGES)
