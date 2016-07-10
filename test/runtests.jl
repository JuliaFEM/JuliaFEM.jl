# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Testing

function run_tests(; verbose=true)

    maybe_test_files = readdir(Pkg.dir("JuliaFEM")*"/test")
    is_test_file(fn) = startswith(fn, "test_") & endswith(fn, ".jl")
    test_files = filter(is_test_file, maybe_test_files)
    #test_files = ["test_nodal_constraints.jl"]

    verbose && info("Test files:")
    for (i, test_file) in enumerate(test_files)
        verbose && info("$i   $test_file")
    end

    t0 = Base.time()

    body = quote
        @testset "JuliaFEM" begin
            for fn in $test_files
                @testset "$fn" begin include(fn) end
            end
        end
    end
    eval(body)

    t1 = round(Base.time()-t0, 2)
    info("Testing completed in $t1 seconds.")
end

run_tests()

#=
facts("Testing if somebody used print, println(), @sprint in src directory") do
  # TODO: make better reqular expression. Currently it will match all print words
  lines_with_print = Dict()
  src_dir = joinpath(Pkg.dir("JuliaFEM"),"src")
  src = readdir(src_dir)
  for file_name in src
    fil = open(joinpath(src_dir,file_name),"r")
    for (line_number,line) in enumerate(readlines(fil))
      if ismatch(r"print",line)
        lines_with_print[file_name * ":$line_number"] = "print found in line: $line_number"
      end
    end
    close(fil)
  end
  @fact lines_with_print => isempty "Instead of println() use Logging.jl package"
end

facts("Testing if we have non ascii characters in the src files") do
  # TODO: we should allow Greeck letters in documentation. Thus this test should skip docstrings
  lines_with_non_ascii = []
  src_dir = joinpath(Pkg.dir("JuliaFEM"),"src")
  src = readdir(src_dir)
  for file_name in src
    fil = open(joinpath(src_dir,file_name),"r")
    for (line_number,line) in enumerate(readlines(fil))
      if ismatch(r"[^\x00-\x7F]",line)
        push!(lines_with_non_ascii, file_name * ":line $line_number")
      end
    end
    close(fil)
  end
  @fact lines_with_non_ascii => isempty "non ascii charecters found in src -> test is failing"
end

facts("Looking the [src,test] folders *.jl files header information") do
  files_no_license = []
  pkg_dir = Pkg.dir("JuliaFEM")
  dirs = ["test";"src"]
  for folder in dirs
    for file_name in readdir(joinpath(pkg_dir,folder))
      if file_name[end-2:end] == ".jl"
        fil = open(joinpath(pkg_dir,folder,file_name),"r")
        head = readall(fil)
      else
        continue
      end
      if ~ismatch(r"This file is a part of JuliaFEM",head)
        push!(files_no_license,"$file_name is missing JuliaFEM statement")
      end
      if ~ismatch(r"MIT",head)
        push!(files_no_license,"$file_name is missing MIT statement")
      end
      if ~ismatch(r"https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md",head)
        push!(files_no_license,"$file_name is missing reference to LICENSE.md")
      end
      close(fil)
    end
  end
  out_str = """License information missing or incorrect. Please use these two lines in the beginning of each file:

# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

  """
  @fact files_no_license => isempty out_str
end

#=
test_files = readdir(Pkg.dir("JuliaFEM")*"/test")
for test_file in test_files
  Logging.info("checking is $test_file is real test file")
  if (startswith(test_file, "test_")) & (endswith(test_file, ".jl"))
    Logging.info("Running tests from file $test_file")
    include(test_file)
  end
end
=#

# Keep this at the end of this file (include statements above this)
@Logging.configure(level=DEBUG)
@debug("Let's print the summary of the tests")
for dic in FactCheck.getstats()
  @debug(dic[1], ": ",dic[2])
end
=#


