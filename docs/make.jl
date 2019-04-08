# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Pkg, Documenter, Literate, JuliaFEM

# really?
juliafem_dir = abspath(joinpath(dirname(pathof(JuliaFEM)), ".."))

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

    local pkg_path
    try
        pkg = getfield(JuliaFEM, Symbol(pkg_name))
        pkg_path = abspath(joinpath(pathof(pkg), "..", ".."))
    catch
        @info("Could not find package $pkg_name from JuliaFEM namespace, pkg_path unknown.")
        return false
    end

    src_dir = joinpath(pkg_path, "docs", "src")
    dst_dir = joinpath(juliafem_dir, "docs", "src", "packages", pkg_name)
    pkg_dir = joinpath(juliafem_dir, "docs", "src", "packages")
    isdir(pkg_dir) || mkpath(pkg_dir)

    # if can find pkg_name/docs/src =>
    # copy that to docs/src/packages/pkg_name
    if isdir(src_dir)
        isdir(dst_dir) || cp(src_dir, dst_dir)
        @info("Copied documentation of package $pkg_name from $src_dir succesfully.")
        return true
    end

    # if can find pkg_name/README.md =>
    # copy that to docs/src/packages/pkg_name/index.md
    readme_file = joinpath(pkg_path, "README.md")
    if isfile(readme_file)
        isdir(dst_dir) || mkpath(dst_dir)
        dst_file = joinpath(dst_dir, "index.md")
        isfile(dst_file) || cp(readme_file, dst_file)
        @info("Copied README.md of package $pkg_name from $readme_file succesfully.")
        return true
    end

    @warn("Cannot copy documentation of package $pkg_name from $src_dir: " *
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
    src_dir = joinpath(juliafem_dir, "docs", "src")
    if isfile(joinpath(src_dir, file))
        push!(dst, src)
        return true
    end
    if isfile(joinpath(src_dir, "packages", file))
        push!(dst, joinpath("packages", src))
        return true
    end
    @warn("Cannot add page $file: no such file")
    return false
end



"""
    generate_developers_guide()

Generate Developer's guide.
"""
function generate_developers_guide()
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
    return DEVELOPER_GUIDE
end


"""
    generate_packages()

Generate single page description for packages by fetching the README.md, index.md
or similar from the package documentation. Returns a vector containing pages for
a documentation.
"""
function generate_packages()
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
    return PACKAGES
end

"""
    license_stripper(s)

Strip license strings away from source file.
"""
function license_stripper(s)
    # return replace(s, r"# This file is a part of JuliaFEM.*?\n\n"sm, "")
    lines = split(s, '\n')
    function myfilt(line)
        occursin(line, "# This file is a part of JuliaFEM.") && return false
        occursin(line, "# License is MIT") && return false
        return true
    end
    new_s = join(filter(myfilt, lines), '\n')
    return new_s
end


"""
    generate_example(example_src, dst_dir)

Given excutable Julia script file, use Literate.jl to render file to Markdown
file. If there exists a directory with the same name than script, but without
extension, it will be copied to destination also. Function returns a path to
the generated example file

# Example

If `example_src` is `examples/my_example_analysis.jl`, and there exists a
directory `examples/my_example_analysis`, there will be a rendered file
`dst_dir/my_example_analysis.md` and additional directory
`dst_dir/my_example_analysis` containing complementary material like mesh,
result files and so on.
"""
function generate_example(example_src, dst_dir)
    dst_file = Literate.markdown(example_src, dst_dir;
                                 documenter=true,
                                 preprocess=license_stripper)

    # ex_dir = directory for complementary material
    complementary_dir = first(splitext(basename(example_src)))
    src_ex_dir = joinpath(juliafem_dir, "examples", complementary_dir)
    dst_ex_dir = joinpath(dst_dir, complementary_dir)
    if isdir(src_ex_dir) && !isdir(dst_ex_dir)
        cp(src_ex_dir, dst_ex_dir)
    end
    docs_src = joinpath(juliafem_dir, "docs", "src")
    return relpath(dst_file, docs_src)
end

"""
    get_example_files(ex_src)

Return all example files from directory `examples`.
"""
function get_example_files(ex_src)
    ex_flt(s) = endswith(s, ".jl") && !startswith(s, "test_")
    return filter(ex_flt, readdir(ex_src))
end

"""
    generate_examples()

Generate examples using Literate.jl. Returns a vector containing pages for
a documentation.
"""
function generate_examples()
    EXAMPLES = []
    ex_src = joinpath(juliafem_dir, "examples")
    ex_dst = joinpath(juliafem_dir, "docs", "src", "examples")
    for ex_file in get_example_files(ex_src)
        example_file = joinpath(ex_src, ex_file)
        markdown_file = generate_example(example_file, ex_dst)
        add_page!(EXAMPLES, markdown_file)
    end
    return EXAMPLES
end

# Collect all together
USER_GUIDE = []
EXAMPLES = generate_examples()
DEVELOPER = generate_developers_guide()
PACKAGES = generate_packages()
APIDOC = ["api.md"]

PAGES = [
         "Home" => "index.md",
         "User's guide" => USER_GUIDE,
         "Examples" => EXAMPLES,
         "Developer's guide" => DEVELOPER,
         "Description of packages" => PACKAGES,
         "API documentation" => APIDOC
        ]

@info("Pages in documentation", PAGES)

makedocs(modules=[JuliaFEM],
         format = Documenter.HTML(analytics="UA-83590644-1"),
         checkdocs = :all,
         sitename = "JuliaFEM.jl",
         pages = PAGES)

include("deploy.jl")
