using Docile, Lexicon, JuliaFEM

const api_directory = "api"
const modules = [JuliaFEM, JuliaFEM.elasticity_solver]

# main_folder = dirname(dirname(@__FILE__))
# this_folder = dirname(@__FILE__)

# file_ = "README.md"
# run(`cp $main_folder/$file_ $this_folder`)

cd(dirname(@__FILE__)) do
    # Run the doctests *before* we start to generate *any* documentation.
    for m in modules
        failures = failed(doctest(m))
        if !isempty(failures.results)
            println("\nDoctests failed, aborting commit.\n")
            display(failures)
            exit(1) # Bail when doctests fail.
        end
    end
    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    for mod in modules
        Lexicon.update!(index, save(joinpath(api_directory, "$(mod).md"), mod))
    end
    save(joinpath(api_directory, "index.md"), index; md_subheader = :category)

    # Add a reminder not to edit the generated files.
    open(joinpath(api_directory, "README.md"), "w") do f
        print(f, """
        Files in this directory are generated using the `build.jl` script. Make
        all changes to the originating docstrings/files rather than these ones.
        """)
    end

    # info("Adding all documentation changes in $(api_directory) to this commit.")
    # success(`git add $(api_directory)`) || exit(1)

end

